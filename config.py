import torch
import os

# ── Rutas ──────────────────────────────────────────────────────────────────────
DATA_DIR = "data"

IMAGE_DIR = r"D:\flickr30k_images\flickr30k_images"

CAPTIONS_FILE = os.path.join(DATA_DIR, "captions.csv")

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Split ──────────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

# ── Vocabulario ────────────────────────────────────────────────────────────────
MIN_WORD_FREQ = 2          # ← era 5; bajar reduce <UNK> significativamente
PAD_TOKEN  = "<PAD>"
SOS_TOKEN  = "<SOS>"
EOS_TOKEN  = "<EOS>"
UNK_TOKEN  = "<UNK>"

# ── Imagen ─────────────────────────────────────────────────────────────────────
IMG_SIZE = 224

# ── Modelo ─────────────────────────────────────────────────────────────────────
EMBED_DIM    = 256
HIDDEN_DIM   = 512
NUM_LAYERS   = 2
DROPOUT      = 0.5
ENCODER_DIM  = 2048        # salida del avgpool de ResNet-50

# ── Entrenamiento ──────────────────────────────────────────────────────────────
BATCH_SIZE   = 32
NUM_EPOCHS   = 1         
LR           = 4e-4
GRAD_CLIP    = 5.0
NUM_WORKERS  = 2

# ── Hardware ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[config] Using device: {DEVICE}")