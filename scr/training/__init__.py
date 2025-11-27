# src/training/__init__.py
"""
Training module for the Automatic Laryngeal Paralysis Assessment project.

This package contains:
- Training pipelines for binary and 3-class classification.
- Common model components (CNN backbone, MLP head, combined model).
- Dataset and dataloader utilities tailored to Agati-based features and images.

Submodules
----------
- train_2class     : training script for binary classification (Normal vs Paralyzed).
- train_3class     : training script for 3-class classification (Normal / Monolateral / Bilateral).
- training_utils   : shared dataset, model, and device utilities.
"""

from .train_2class import my_train as my_train_2class
from .train_3class import my_train_3class
from .training_utils import (
    CustomDataset,
    DeviceDataLoader,
    CombinedModel,
    MLPModule,
    ModifiedResNet18,
)

__all__ = [
    "my_train_2class",
    "my_train_3class",
    "CustomDataset",
    "DeviceDataLoader",
    "CombinedModel",
    "MLPModule",
    "ModifiedResNet18",
]

