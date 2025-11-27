# src/ablation/__init__.py
"""
Ablation module for the Automatic Laryngeal Paralysis Assessment project.

This package groups utilities and scripts to run ablation studies on:
- Binary classification (Normal vs Paralyzed)
- 3-class classification (Normal / Monolateral / Bilateral)

Submodules
----------
- ablation_2class : main script for 2-class ablation.
- ablation_3class : main script for 3-class ablation.
- ablation_utils  : shared utilities (He init, reinitialisation, training loop, Shapley, etc.).
"""

from . import ablation_2class
from . import ablation_3class
from .ablation_utils import (
    he_init,
    reinit_weights,
    select_additional_indices,
    my_train,
    shapley_value,
)

__all__ = [
    "ablation_2class",
    "ablation_3class",
    "he_init",
    "reinit_weights",
    "select_additional_indices",
    "my_train",
    "shapley_value",
]

