# src/xai/__init__.py
"""
Explainable AI (XAI) module for the Automatic Laryngeal Paralysis Assessment project.

This package provides:
- 2-class XAI pipeline (Normal vs Paralyzed).
- 3-class XAI pipeline (Normal / Monolateral / Bilateral).
- Dataset classes tailored for integrated gradients over images and additional features.
"""

from .xai_utils import XAIDatasetBinary, XAIDatasetMulticlass
from .xai_2class import run_xai_2class
from .xai_3class import run_xai_3class

__all__ = [
    "XAIDatasetBinary",
    "XAIDatasetMulticlass",
    "run_xai_2class",
    "run_xai_3class",
]
