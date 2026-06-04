"""
dataset.py
──────────
- Vocabulary: construida desde captions.txt
- FlickrDataset: __getitem__ devuelve (imagen, caption_tensor)
- get_loaders: devuelve train / val / test DataLoaders
"""

import os
import pickle
import random
from collections import Counter

import pandas as pd
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config


# ════════════════════════════════════════════════════════════════
# Vocabulario
# ════════════════════════════════════════════════════════════════

class Vocabulary:
    def __init__(self, freq_threshold: int = config.MIN_WORD_FREQ):
        self.freq_threshold = freq_threshold
        self.itos = {0: config.PAD_TOKEN, 1: config.SOS_TOKEN,
                     2: config.EOS_TOKEN, 3: config.UNK_TOKEN}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text: str):
        return text.lower().strip().split()

    def build(self, sentences):
        counter = Counter()
        for sentence in sentences:
            counter.update(self.tokenize(sentence))
        idx = len(self.itos)
        for word, freq in counter.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx]  = word
                idx += 1
        print(f"[Vocabulary] size = {len(self)}")

    def numericalize(self, text: str):
        unk = self.stoi[config.UNK_TOKEN]
        return [self.stoi.get(w, unk) for w in self.tokenize(text)]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)


# ════════════════════════════════════════════════════════════════
# Dataset
# ════════════════════════════════════════════════════════════════

class FlickrDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: Vocabulary,
                 transform=None):
        self.df        = df.reset_index(drop=True)
        self.vocab     = vocab
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(os.path.join(config.IMAGE_DIR, row["image"])).convert("RGB")
        if self.transform:
            img = self.transform(img)

        caption = [self.vocab.stoi[config.SOS_TOKEN]]
        caption += self.vocab.numericalize(row["caption"])
        caption += [self.vocab.stoi[config.EOS_TOKEN]]
        caption  = torch.tensor(caption, dtype=torch.long)

        return img, caption


# ════════════════════════════════════════════════════════════════
# Collate
# ════════════════════════════════════════════════════════════════

class CaptionCollate:
    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs     = torch.stack([item[0] for item in batch])
        captions = pad_sequence([item[1] for item in batch],
                                batch_first=False,
                                padding_value=self.pad_idx)
        return imgs, captions


# ════════════════════════════════════════════════════════════════
# Transforms
# ════════════════════════════════════════════════════════════════

def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE + 32, config.IMG_SIZE + 32)),
            transforms.RandomCrop(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])


# ════════════════════════════════════════════════════════════════
# Función pública: get_loaders
# ════════════════════════════════════════════════════════════════

def get_loaders(vocab_cache: str = "vocab.pkl"):
    """Devuelve (train_loader, val_loader, test_loader, vocab)."""

    df = pd.read_csv(config.CAPTIONS_FILE, sep="|")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"image_name": "image", "comment": "caption"})
    df = df[["image", "caption"]].dropna()
    df["caption"] = df["caption"].astype(str).str.strip()

    # ── Split por imagen (no por fila) para evitar data-leakage ──
    all_images = df["image"].unique().tolist()
    random.seed(42)
    random.shuffle(all_images)

    n       = len(all_images)
    n_train = int(n * config.TRAIN_RATIO)
    n_val   = int(n * config.VAL_RATIO)

    train_imgs = set(all_images[:n_train])
    val_imgs   = set(all_images[n_train:n_train + n_val])
    test_imgs  = set(all_images[n_train + n_val:])

    train_df = df[df["image"].isin(train_imgs)]
    val_df   = df[df["image"].isin(val_imgs)]
    test_df  = df[df["image"].isin(test_imgs)]

    print(f"[dataset] imágenes  train={len(train_imgs)}  "
          f"val={len(val_imgs)}  test={len(test_imgs)}")
    print(f"[dataset] filas     train={len(train_df)}  "
          f"val={len(val_df)}  test={len(test_df)}")

    # Guardar lista de test para evaluate.py
    test_df.to_csv("test_split.csv", index=False)

    # ── Vocabulario ──
    if os.path.exists(vocab_cache):
        vocab = Vocabulary.load(vocab_cache)
        print(f"[dataset] Vocabulario cargado desde {vocab_cache}")
    else:
        vocab = Vocabulary()
        vocab.build(train_df["caption"].tolist())
        vocab.save(vocab_cache)
        print(f"[dataset] Vocabulario guardado en {vocab_cache}")

    pad_idx = vocab.stoi[config.PAD_TOKEN]

    # ── DataLoaders ──
    train_ds = FlickrDataset(train_df, vocab, get_transforms(train=True))
    val_ds   = FlickrDataset(val_df,   vocab, get_transforms(train=False))
    test_ds  = FlickrDataset(test_df,  vocab, get_transforms(train=False))

    collate = CaptionCollate(pad_idx)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                          shuffle=True,  collate_fn=collate,
                          num_workers=config.NUM_WORKERS,
                          pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE,
                          shuffle=False, collate_fn=collate,
                          num_workers=config.NUM_WORKERS,
                          pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE,
                            shuffle=False, collate_fn=collate,
                            num_workers=config.NUM_WORKERS,
                            pin_memory=False)

    return train_loader, val_loader, test_loader, vocab