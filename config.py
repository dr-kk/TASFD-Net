import os

ROOT = "."

DATASET = os.path.join(ROOT, "datasets")

VSUMM = os.path.join(DATASET, "VSUMM")
TVSUM = os.path.join(DATASET, "TVSum")
SUMME = os.path.join(DATASET, "SumMe")
ACCIDENT = os.path.join(DATASET, "Accident")

# Video settings
IMG_SIZE = 224
NUM_FRAMES = 16
FPS = 1

# ===== ADD THESE =====
MODEL_NAME = "MCG-NJU/videomae-base"

BATCH_SIZE = 2
EPOCHS = 30
LR = 1e-4

DEVICE = "cpu"   # Abhi CPU hai
# =====================

CHECKPOINT = "checkpoints"
RESULT = "results"

os.makedirs(CHECKPOINT, exist_ok=True)
os.makedirs(RESULT, exist_ok=True)