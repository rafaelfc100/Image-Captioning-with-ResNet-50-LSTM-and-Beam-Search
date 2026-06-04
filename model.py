"""
model.py
────────
- EncoderCNN   : ResNet-50 preentrenada (últimas 2 capas descongeladas)
- DecoderRNN   : LSTM con inyección visual en CADA paso (fix principal)
- ImageCaptioner: wrapper con greedy y beam search
"""

import torch
import torch.nn as nn
import torchvision.models as models

import config


# ════════════════════════════════════════════════════════════════
# Encoder
# ════════════════════════════════════════════════════════════════

class EncoderCNN(nn.Module):
    def __init__(self, embed_dim: int = config.EMBED_DIM):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Congela todo primero
        for p in resnet.parameters():
            p.requires_grad_(False)

        # Descongela layer4
        children_layer4 = list(resnet.layer4.children())
        for p in children_layer4[-1].parameters():   # solo el último bloque
            p.requires_grad_(True)

        modules      = list(resnet.children())[:-1]   # hasta avgpool inclusive
        self.resnet  = nn.Sequential(*modules)
        self.project = nn.Linear(config.ENCODER_DIM, embed_dim)
        self.bn      = nn.BatchNorm1d(embed_dim, momentum=0.01)
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        # images: (B, 3, H, W)
        features = self.resnet(images)               # (B, 2048, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (B, 2048)
        features = self.dropout(self.bn(self.project(features)))  # (B, E)
        return features


# ════════════════════════════════════════════════════════════════
# Decoder  — inyección visual en cada paso
# ════════════════════════════════════════════════════════════════

class DecoderRNN(nn.Module):
    """
    El vector visual se concatena al embedding de CADA token antes
    de entrar al LSTM: input_size = embed_dim + embed_dim.
    Esto evita que el modelo "olvide" la imagen tras el primer paso.
    """

    def __init__(self, embed_dim: int, hidden_dim: int,
                 vocab_size: int, num_layers: int = config.NUM_LAYERS,
                 dropout: float = config.DROPOUT):
        super().__init__()
        self.embed_dim  = embed_dim
        self.hidden_dim = hidden_dim

        self.embed   = nn.Embedding(vocab_size, embed_dim)
        # Input: [token_emb | visual_feat]  →  2 * embed_dim
        self.lstm    = nn.LSTM(embed_dim * 2, hidden_dim, num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.fc      = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    # ── Entrenamiento (teacher forcing) ──────────────────────────
    def forward(self, features, captions):
        """
        features : (B, embed_dim)
        captions : (B, T)   — incluye <SOS> y <EOS>
        returns  : (B, T-1, vocab_size)
        """
        B, T = captions.shape

        # Embeddings de los tokens de entrada (sin <EOS>)
        tok_emb = self.dropout(self.embed(captions[:, :-1]))   # (B, T-1, E)

        # Replica el vector visual para cada paso temporal
        vis = features.unsqueeze(1).expand(B, T - 1, self.embed_dim)  # (B, T-1, E)

        # Concatena por la dimensión de features
        inputs = torch.cat([tok_emb, vis], dim=2)              # (B, T-1, 2E)

        hiddens, _ = self.lstm(inputs)                         # (B, T-1, H)
        out        = self.fc(hiddens)                          # (B, T-1, V)
        return out

    # ── Greedy decoding ──────────────────────────────────────────
    @torch.no_grad()
    def generate_greedy(self, feature, vocab, max_len: int = 50):
        """feature: (1, embed_dim)"""
        sos    = vocab.stoi["<SOS>"]
        eos    = vocab.stoi["<EOS>"]
        token  = torch.tensor([[sos]], device=feature.device)
        states = None
        words  = []

        for _ in range(max_len):
            tok_emb = self.dropout(self.embed(token))          # (1,1,E)
            vis     = feature.unsqueeze(0)                     # (1,1,E)
            inp     = torch.cat([tok_emb, vis], dim=2)        # (1,1,2E)
            out, states = self.lstm(inp, states)
            pred    = self.fc(out.squeeze(1)).argmax(-1)
            idx     = pred.item()
            if idx == eos:
                break
            words.append(vocab.itos[idx])
            token = pred.unsqueeze(1)

        return " ".join(words)

    # ── Beam search decoding ─────────────────────────────────────
    @torch.no_grad()
    def generate_beam(self, feature, vocab, beam_size: int = 5,
                      max_len: int = 50):
        """
        feature   : (1, embed_dim)
        beam_size : ancho del beam (3-5 es suficiente)
        """
        sos = vocab.stoi["<SOS>"]
        eos = vocab.stoi["<EOS>"]
        dev = feature.device

        # Cada candidato: (score, token_ids, lstm_states)
        token  = torch.tensor([[sos]], device=dev)
        tok_emb = self.embed(token)                            # (1,1,E)
        vis     = feature.unsqueeze(0)                         # (1,1,E)
        inp     = torch.cat([tok_emb, vis], dim=2)
        out, states = self.lstm(inp, None)
        log_probs   = torch.log_softmax(self.fc(out.squeeze(1)), dim=-1)

        topk_scores, topk_idx = log_probs.topk(beam_size, dim=-1)
        # Lista de (score, [tokens], states)
        beams = [
            (topk_scores[0, k].item(), [topk_idx[0, k].item()], states)
            for k in range(beam_size)
        ]

        completed = []

        for _ in range(max_len - 1):
            new_beams = []
            for score, tokens, st in beams:
                if tokens[-1] == eos:
                    completed.append((score, tokens))
                    continue
                token   = torch.tensor([[tokens[-1]]], device=dev)
                tok_emb = self.embed(token)
                inp     = torch.cat([tok_emb, feature.unsqueeze(0)], dim=2)
                out, new_st = self.lstm(inp, st)
                lp      = torch.log_softmax(self.fc(out.squeeze(1)), dim=-1)
                topk_s, topk_i = lp.topk(beam_size, dim=-1)
                for k in range(beam_size):
                    new_score  = score + topk_s[0, k].item()
                    new_tokens = tokens + [topk_i[0, k].item()]
                    new_beams.append((new_score, new_tokens, new_st))

            # Mantén solo los beam_size mejores
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_size]

            if len(completed) >= beam_size:
                break

        # Añade los beams activos como candidatos
        for score, tokens, _ in beams:
            completed.append((score, tokens))

        # Elige el de mayor score (normalizado por longitud)
        best_score, best_tokens = max(
            completed,
            key=lambda x: x[0] / max(len(x[1]), 1)
        )

        words = []
        for idx in best_tokens:
            if idx == eos:
                break
            if idx != sos:
                words.append(vocab.itos[idx])
        return " ".join(words)


# ════════════════════════════════════════════════════════════════
# Wrapper
# ════════════════════════════════════════════════════════════════

class ImageCaptioner(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.encoder = EncoderCNN(config.EMBED_DIM)
        self.decoder = DecoderRNN(config.EMBED_DIM, config.HIDDEN_DIM,
                                  vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        return self.decoder(features, captions)

    @torch.no_grad()
    def caption(self, image_tensor, vocab,
                beam_size: int = 5, max_len: int = 50):
        """
        image_tensor : (1, 3, H, W)
        beam_size    : 1 = greedy, >1 = beam search
        """
        feature = self.encoder(image_tensor)
        if beam_size <= 1:
            return self.decoder.generate_greedy(feature, vocab, max_len)
        return self.decoder.generate_beam(feature, vocab, beam_size, max_len)