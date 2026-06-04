"""
train.py
────────
Entrena el modelo ImageCaptioner y guarda checkpoints.

Reanuda desde el último checkpoint periódico (epoch_XXX.pth) si existe,
o desde best.pth como fallback.

Al finalizar (o con Ctrl+C) guarda las gráficas de entrenamiento en:
    checkpoints/training_curves.png

Uso:
    python train.py
    python train.py --epochs 60   # sobreescribe NUM_EPOCHS de config
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import glob
import json
import time

import matplotlib
matplotlib.use("Agg")          # sin GUI, funciona en servidor
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import config
from dataset import get_loaders
from model import ImageCaptioner


# ════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ════════════════════════════════════════════════════════════════

def save_checkpoint(state: dict, filename: str):
    torch.save(state, filename)
    print(f"  ✓ Checkpoint guardado → {filename}")


def find_latest_periodic(ckpt_dir: str):
    """Devuelve el path al epoch_XXX.pth más reciente, o None."""
    pattern = os.path.join(ckpt_dir, "epoch_*.pth")
    files   = sorted(glob.glob(pattern))
    return files[-1] if files else None


def load_checkpoint(path: str, model, optimizer=None):
    ckpt = torch.load(path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    history = ckpt.get("history", {"train": [], "val": [], "lr": []})
    return ckpt.get("epoch", 0), ckpt.get("best_val_loss", float("inf")), history


# ════════════════════════════════════════════════════════════════
# Gráficas
# ════════════════════════════════════════════════════════════════

def save_plots(history: dict, out_path: str):
    """Genera y guarda training_curves.png."""
    epochs     = range(1, len(history["train"]) + 1)
    train_loss = history["train"]
    val_loss   = history["val"]
    lrs        = history["lr"]

    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor("#0f0f17")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # ── Paleta ──
    C_TRAIN = "#7c6af7"
    C_VAL   = "#e05fe0"
    C_LR    = "#4ade9f"
    C_TEXT  = "#c8c8d8"
    C_GRID  = "#2a2a3a"
    C_BG    = "#13131a"

    def style_ax(ax, title):
        ax.set_facecolor(C_BG)
        ax.set_title(title, color=C_TEXT, fontsize=13, fontweight="bold", pad=10)
        ax.tick_params(colors=C_TEXT, labelsize=9)
        ax.xaxis.label.set_color(C_TEXT)
        ax.yaxis.label.set_color(C_TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(C_GRID)
        ax.grid(True, color=C_GRID, linewidth=0.6, linestyle="--", alpha=0.7)

    # ── Panel 1: Loss ──
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(epochs, train_loss, color=C_TRAIN, linewidth=2,
             marker="o", markersize=3, label="Train loss")
    ax1.plot(epochs, val_loss,   color=C_VAL,   linewidth=2,
             marker="o", markersize=3, label="Val loss", linestyle="--")
    best_ep  = int(min(range(len(val_loss)), key=lambda i: val_loss[i])) + 1
    best_val = min(val_loss)
    ax1.axvline(best_ep, color=C_VAL, linewidth=1, alpha=0.4, linestyle=":")
    ax1.annotate(f"best {best_val:.3f}\n(ep {best_ep})",
                 xy=(best_ep, best_val),
                 xytext=(best_ep + max(1, len(epochs)//10), best_val + (max(val_loss)-min(val_loss))*0.1),
                 color=C_VAL, fontsize=8,
                 arrowprops=dict(arrowstyle="->", color=C_VAL, lw=1))
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Cross-Entropy Loss")
    legend = ax1.legend(facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT, fontsize=9)
    style_ax(ax1, "Loss de entrenamiento y validación")

    # ── Panel 2: Learning rate ──
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(epochs, lrs, color=C_LR, linewidth=2, marker="o", markersize=3)
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Learning Rate")
    ax2.set_yscale("log")
    style_ax(ax2, "Learning Rate (escala log)")

    # Sombra entre train/val
    ax1.fill_between(epochs, train_loss, val_loss,
                     alpha=0.07, color=C_VAL)

    plt.suptitle("Training Curves — Image Captioning",
                 color=C_TEXT, fontsize=15, fontweight="bold", y=1.02)

    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  📊 Gráfica guardada → {out_path}")


# ════════════════════════════════════════════════════════════════
# epoch_pass
# ════════════════════════════════════════════════════════════════

def epoch_pass(loader, model, criterion, optimizer, pad_idx,
               train: bool, epoch_label: str):
    model.train(train)
    total_loss = 0.0
    total_tok  = 0
    phase = "train" if train else "val"

    bar = tqdm(loader, desc=f"  {phase:5s}", leave=False,
               ncols=90, colour="blue" if train else "magenta")

    with torch.set_grad_enabled(train):
        for imgs, captions in bar:
            imgs     = imgs.to(config.DEVICE)
            captions = captions.to(config.DEVICE).permute(1, 0)   # (B, T)

            outputs = model(imgs, captions)       # (B, T-1, V) — decoder ya recorta
            B, T1, V = outputs.shape
            out = outputs.reshape(-1, V)          # (B*(T-1), V)
            tgt = captions[:, 1:].reshape(-1)

            loss = criterion(out, tgt)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                optimizer.step()

            mask       = tgt != pad_idx
            n_tok      = mask.sum().item()
            total_loss += loss.item() * n_tok
            total_tok  += n_tok

            bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_tok, 1)


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS,
                        help="Total de épocas a entrenar")
    args = parser.parse_args()
    NUM_EPOCHS = args.epochs

    print("=" * 60)
    print(f"  Dispositivo : {config.DEVICE}")
    print(f"  Épocas      : {NUM_EPOCHS}")
    print("=" * 60)

    # ── Datos ──
    train_loader, val_loader, _, vocab = get_loaders()
    vocab_size = len(vocab)
    pad_idx    = vocab.stoi[config.PAD_TOKEN]

    # ── Modelo ──
    model = ImageCaptioner(vocab_size).to(config.DEVICE)

    # ── Optimizador & scheduler ──
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LR,
        weight_decay=1e-4
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=2,
        threshold=0.005,       # mejora real > 0.5% para resetear contador
        threshold_mode="rel",
        min_lr=1e-5
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # ── Busca el checkpoint más reciente ──
    start_epoch   = 0
    best_val_loss = float("inf")
    history       = {"train": [], "val": [], "lr": []}

    best_ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best.pth")
    last_periodic  = find_latest_periodic(config.CHECKPOINT_DIR)

    resume_path = last_periodic or (best_ckpt_path if os.path.exists(best_ckpt_path) else None)

    if resume_path:
        print(f"Reanudando desde {resume_path} …")
        start_epoch, best_val_loss, history = load_checkpoint(
            resume_path, model, optimizer)
        start_epoch += 1
        print(f"  → época {start_epoch + 1}  |  mejor val loss hasta ahora: {best_val_loss:.4f}")
        print(f"  → historial cargado: {len(history['train'])} épocas previas")
    else:
        print("Sin checkpoint previo — entrenando desde cero.")

    plot_path = os.path.join(config.CHECKPOINT_DIR, "training_curves.png")

    # ── Bucle ──
    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            t0          = time.time()
            epoch_label = f"[{epoch+1:03d}/{NUM_EPOCHS}]"

            train_loss = epoch_pass(train_loader, model, criterion,
                                    optimizer, pad_idx, train=True,
                                    epoch_label=epoch_label)
            val_loss   = epoch_pass(val_loader,   model, criterion,
                                    optimizer, pad_idx, train=False,
                                    epoch_label=epoch_label)

            # ── Scheduler PRIMERO, luego leer LR actualizado ──
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            elapsed    = time.time() - t0

            history["train"].append(train_loss)
            history["val"].append(val_loss)
            history["lr"].append(current_lr)

            print(f"Epoch {epoch_label}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"lr={current_lr:.2e}  ({elapsed:.1f}s)")

            # ── Mejor modelo ──
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint({
                    "epoch"         : epoch,
                    "model"         : model.state_dict(),
                    "optimizer"     : optimizer.state_dict(),
                    "best_val_loss" : best_val_loss,
                    "vocab_size"    : vocab_size,
                    "history"       : history,
                }, best_ckpt_path)

            # ── Checkpoint periódico cada 5 épocas ──
            if (epoch + 1) % 5 == 0:
                periodic_path = os.path.join(
                    config.CHECKPOINT_DIR, f"epoch_{epoch+1:03d}.pth")
                save_checkpoint({
                    "epoch"         : epoch,
                    "model"         : model.state_dict(),
                    "optimizer"     : optimizer.state_dict(),
                    "best_val_loss" : best_val_loss,
                    "vocab_size"    : vocab_size,
                    "history"       : history,
                }, periodic_path)

                # Actualiza gráfica en cada checkpoint
                save_plots(history, plot_path)

    except KeyboardInterrupt:
        print("\n\nInterrumpido por el usuario.")

    finally:
        # Siempre guarda la gráfica al terminar
        if history["train"]:
            save_plots(history, plot_path)
            print(f"\nEntrenamiento finalizado.")
            print(f"  Mejor val loss : {best_val_loss:.4f}")
            print(f"  Gráfica        : {plot_path}")


if __name__ == "__main__":
    main()