# src/xai/config.py
"""
Configuration for XAI scripts (paths, filenames, seeds).
Adjust the paths to your local / cluster environment.
"""

from pathlib import Path

# Base directory for the Spain Trials dataset
# TODO: change this to your actual base path
DATA_DIR = Path(
    "/content/drive/MyDrive/AI in healthcare/Laryngeal paralysis/Spain Dataset/Trials"
)

# Features table (Agati features)
FEATURES_XLSX = DATA_DIR / "Agati features.xlsx"

# Weights for the trained models
# TODO: update to your actual weights paths
WEIGHTS_2CLASS = Path(
    "/content/drive/MyDrive/AI in healthcare/Laryngeal paralysis/Weights/"
    "combined_model_weights_due_classi.pth"
)

WEIGHTS_3CLASS = Path(
    "/content/drive/MyDrive/AI in healthcare/Laryngeal paralysis/Weights/"
    "best_model_three.pth"
)

# Where to save attributions
ATTRIBUTIONS_2CLASS = Path(
    "/content/drive/MyDrive/AI in healthcare/Laryngeal paralysis/Weights/"
    "attributions_bin.npy"
)

ATTRIBUTIONS_3CLASS = Path(
    "/content/drive/MyDrive/AI in healthcare/Laryngeal paralysis/Weights/"
    "attributions.npy"
)

# Random seed (kept consistent with the notebooks)
SEED = 42
