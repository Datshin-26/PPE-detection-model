"""
Central configuration for the PPE Detection System.
Keep all tunable constants here so nothing is hardcoded in the app pages.
"""
import os
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATABASE_DIR = BASE_DIR / "database"
REPORTS_DIR = BASE_DIR / "reports"
ASSETS_DIR = BASE_DIR / "assets"
DB_PATH = DATABASE_DIR / "ppe_detections.db"

# --- Model ---
# Custom-trained weights (train.py produces this after you fine-tune on a
# PPE dataset, e.g. exported from Roboflow). If this file doesn't exist,
# the app falls back to a stock COCO-pretrained model in DEMO MODE, which
# can only detect generic "person" boxes (no helmet/vest/mask classes).
CUSTOM_WEIGHTS_PATH = MODELS_DIR / "ppe_best.pt"
DEMO_WEIGHTS_NAME = "yolov8n.pt"  # auto-downloaded by ultralytics on first run

# Class names your custom model should be trained on (edit to match your
# Roboflow dataset's data.yaml order exactly).
PPE_CLASS_NAMES = [
    "person",
    "helmet",
    "no_helmet",
    "vest",
    "no_vest",
    "mask",
    "no_mask",
]

# Which classes count as "mandatory" for a worker to be marked Compliant
MANDATORY_PPE = ["helmet", "vest", "mask"]

CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.45

# --- UI ---
APP_TITLE = "PPE Detection System"
APP_ICON = "🦺"
COMPLIANT_COLOR = (0, 200, 0)      # green (BGR for OpenCV)
NON_COMPLIANT_COLOR = (0, 0, 230)  # red (BGR for OpenCV)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)
