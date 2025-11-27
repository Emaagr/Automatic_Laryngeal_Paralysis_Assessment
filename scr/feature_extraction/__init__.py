# src/feature_extraction/__init__.py
"""
Feature extraction module for the Automatic Laryngeal Paralysis Assessment project.

This package provides functions to:
- Extract frame-wise and trial-wise features from video-laryngoscopy data.
- Post-process and clean extracted features (e.g., outlier filtering).
- Ensure a consistent folder structure for saving derived data.

Submodules
----------
- feature_extraction : core feature extraction routines.
- utils              : general-purpose utilities for preprocessing and I/O.
"""

from .feature_extraction import feature_extract, ggm
from .utils import ensure_dir, filter_outliers, extract_frame

__all__ = [
    "feature_extract",
    "ggm",
    "ensure_dir",
    "filter_outliers",
    "extract_frame",
]

