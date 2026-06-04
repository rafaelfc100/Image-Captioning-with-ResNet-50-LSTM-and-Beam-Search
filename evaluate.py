"""
evaluate.py
───────────
Evalúa el modelo en el conjunto de test con BLEU-1..4.
Usa beam search por defecto (beam_size=5).

Uso:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/best.pth --beam 5
    python evaluate.py --beam 1   # greedy (más rápido)
"""

import argparse
import os

import pandas as pd
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from PIL import Image
from tqdm import tqdm

import config
from dataset import FlickrDataset, Vocabulary, get_transforms, CaptionCollate
from model import ImageCaptioner
from torch.utils.data import DataLoader


def evaluate(checkpoint_path: str, beam_size: int = 5):
    # ── Vocabulario ──
    if not os.path.exists("vocab.pkl"):
        raise FileNotFoundError("vocab.pkl no encontrado. Ejecuta train.py primero.")
    vocab   = Vocabulary.load("vocab.pkl")
    pad_idx = vocab.stoi[config.PAD_TOKEN]

    # ── Test split ──
    if not os.path.exists("test_split.csv"):
        raise FileNotFoundError("test_split.csv no encontrado.")
    test_df = pd.read_csv("test_split.csv")

    # ── Modelo ──
    model = ImageCaptioner(len(vocab)).to(config.DEVICE)
    ckpt  = torch.load(checkpoint_path, map_location=config.DEVICE,
                       weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Checkpoint cargado : {checkpoint_path}")
    print(f"Vocabulario        : {len(vocab)} tokens")
    print(f"Beam size          : {beam_size} {'(greedy)' if beam_size <= 1 else ''}")

    # ── Referencias múltiples por imagen ──
    all_images   = test_df["image"].unique().tolist()
    refs_by_img  = {}
    for _, row in test_df.iterrows():
        refs_by_img.setdefault(row["image"], []).append(
            vocab.tokenize(row["caption"]))

    transform    = get_transforms(train=False)
    hypotheses   = []
    references   = []

    print("\nGenerando captions …")
    with torch.no_grad():
        for img_name in tqdm(all_images, ncols=80):
            img_path   = os.path.join(config.IMAGE_DIR, img_name)
            img_tensor = transform(
                Image.open(img_path).convert("RGB")
            ).unsqueeze(0).to(config.DEVICE)

            caption = model.caption(img_tensor, vocab, beam_size=beam_size)
            hypotheses.append(vocab.tokenize(caption))
            references.append(refs_by_img[img_name])

    # ── BLEU ──
    smooth = SmoothingFunction().method1
    b1 = corpus_bleu(references, hypotheses, weights=(1,0,0,0),      smoothing_function=smooth)
    b2 = corpus_bleu(references, hypotheses, weights=(.5,.5,0,0),    smoothing_function=smooth)
    b3 = corpus_bleu(references, hypotheses, weights=(.33,.33,.33,0), smoothing_function=smooth)
    b4 = corpus_bleu(references, hypotheses, weights=(.25,.25,.25,.25), smoothing_function=smooth)

    print("\n" + "=" * 44)
    print(f"  BLEU-1 : {b1:.4f}")
    print(f"  BLEU-2 : {b2:.4f}")
    print(f"  BLEU-3 : {b3:.4f}")
    print(f"  BLEU-4 : {b4:.4f}")
    print("=" * 44)

    print("\nEjemplos (primeros 8):")
    for i, img_name in enumerate(all_images[:8]):
        print(f"\n  Imagen : {img_name}")
        print(f"  Hyp    : {' '.join(hypotheses[i])}")
        for j, ref in enumerate(references[i][:2]):
            print(f"  Ref[{j}] : {' '.join(ref)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        default=os.path.join(config.CHECKPOINT_DIR, "best.pth"))
    parser.add_argument("--beam", type=int, default=5,
                        help="Beam size (1 = greedy)")
    args = parser.parse_args()
    evaluate(args.checkpoint, beam_size=args.beam)