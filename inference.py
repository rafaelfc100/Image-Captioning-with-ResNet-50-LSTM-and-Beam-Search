"""
inference.py
────────────
Genera captions para imágenes nuevas.

Uso:
    python inference.py --image foto.jpg
    python inference.py --image img1.jpg img2.jpg --beam 5
    python inference.py --image foto.jpg --beam 1   # greedy
"""

import argparse
import os

import torch
from PIL import Image

import config
from dataset import Vocabulary, get_transforms
from model import ImageCaptioner


def load_model(checkpoint_path: str, vocab: Vocabulary):
    model = ImageCaptioner(len(vocab)).to(config.DEVICE)
    ckpt  = torch.load(checkpoint_path, map_location=config.DEVICE,
                       weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", nargs="+", required=True)
    parser.add_argument("--checkpoint",
                        default=os.path.join(config.CHECKPOINT_DIR, "best.pth"))
    parser.add_argument("--vocab",  default="vocab.pkl")
    parser.add_argument("--beam",   type=int, default=5,
                        help="Beam size (1 = greedy)")
    args = parser.parse_args()

    vocab     = Vocabulary.load(args.vocab)
    model     = load_model(args.checkpoint, vocab)
    transform = get_transforms(train=False)

    print(f"\nDispositivo : {config.DEVICE}")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Vocabulario : {len(vocab)} tokens")
    print(f"Beam size   : {args.beam}")
    print("─" * 50)

    for img_path in args.image:
        if not os.path.exists(img_path):
            print(f"[!] No encontrado: {img_path}")
            continue
        img    = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            caption = model.caption(tensor, vocab, beam_size=args.beam)
        print(f"Imagen  : {img_path}")
        print(f"Caption : {caption}")
        print("─" * 50)


if __name__ == "__main__":
    main()