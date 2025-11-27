# src/xai/xai_utils.py
"""
Utility components for XAI:
- Datasets for 2-class and 3-class XAI
- Helpers to build the CombinedModel (CNN + MLP) for 2-class and 3-class setups.
"""

from typing import Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from ..training.training_utils import (
    device,
    composer,          # image transforms (Resize+Gray+Normalize)
    set_global_seed,
    ModifiedResNet18,
    MLPModule,
    CombinedModel,
)


class XAIDatasetBinary(Dataset):
    """
    Dataset for 2-class XAI (Normal vs Paralyzed).

    Expects:
    - `subjects`: list of trial names (values from 'Name' column)
    - `features_dataframe`: DataFrame containing at least 'Name', 'Path', 'Class',
      and additional features between 'Name' and 'Class'.

    Returns a dict with:
        - 'image': transformed image tensor (for model input)
        - 'additional_features': handcrafted features (float32 tensor)
        - 'labels': binary label (0/1, where Class > 0 -> 1), float32
        - 'original': resized image tensor (C, H, W) for visualization
    """

    def __init__(self, subjects, features_dataframe, transform=None):
        self.subjects = subjects
        self.features_dataframe = features_dataframe
        self.transform = transform if transform is not None else composer
        self.resize = transforms.Resize((250, 250))
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int):
        name = self.subjects[idx]
        row = self.features_dataframe.loc[self.features_dataframe["Name"] == name].iloc[0]

        # Image
        image_path = row["Path"]
        image = Image.open(image_path)

        resized = self.resize(image)
        original_tensor = self.to_tensor(resized)

        # Additional features between 'Name' and 'Class'
        cols = list(self.features_dataframe.columns)
        name_idx = cols.index("Name")
        class_idx = cols.index("Class")
        feat_vals = row.iloc[name_idx + 1:class_idx].values.astype("float32")
        additional_features = torch.tensor(feat_vals, dtype=torch.float32)

        # Binary label: Class > 0 -> 1
        label_int = int(row["Class"])
        label_bin = 1 if label_int > 0 else 0
        label_tensor = torch.tensor(label_bin, dtype=torch.float32)

        # Model transform
        image_t = self.transform(image)

        return {
            "image": image_t,
            "additional_features": additional_features,
            "labels": label_tensor,
            "original": original_tensor,
        }


class XAIDatasetMulticlass(Dataset):
    """
    Dataset for 3-class XAI (Normal / Monolateral / Bilateral).

    Returns a dict with:
        - 'image': transformed image tensor (for model input)
        - 'additional_features': handcrafted features (float32 tensor)
        - 'labels': integer class label (0,1,2) as int64
        - 'original': resized image tensor (C,H,W) for visualization
    """

    def __init__(self, subjects, features_dataframe, transform=None):
        self.subjects = subjects
        self.features_dataframe = features_dataframe
        self.transform = transform if transform is not None else composer
        self.resize = transforms.Resize((250, 250))
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int):
        name = self.subjects[idx]
        row = self.features_dataframe.loc[self.features_dataframe["Name"] == name].iloc[0]

        # Image
        image_path = row["Path"]
        image = Image.open(image_path)

        resized = self.resize(image)
        original_tensor = self.to_tensor(resized)

        # Additional features between 'Name' and 'Class'
        cols = list(self.features_dataframe.columns)
        name_idx = cols.index("Name")
        class_idx = cols.index("Class")
        feat_vals = row.iloc[name_idx + 1:class_idx].values.astype("float32")
        additional_features = torch.tensor(feat_vals, dtype=torch.float32)

        # Label as integer (0,1,2)
        label = int(row["Class"])
        label_tensor = torch.tensor(label, dtype=torch.int64)

        image_t = self.transform(image)

        return {
            "image": image_t,
            "additional_features": additional_features,
            "labels": label_tensor,
            "original": original_tensor,
        }


def build_binary_model(n_additional: int,
                       cnn_out_dim: int = 10) -> CombinedModel:
    """
    Build the CombinedModel for 2-class classification:
    - ResNet18 -> cnn_out_dim
    - MLP with [8, 1] neurons (single logit output).
    """
    cnn = ModifiedResNet18(num_classes=cnn_out_dim)
    mlp_num_layers = 2
    mlp_num_neurons: Sequence[int] = [8, 1]

    mlp = MLPModule(
        input_size=cnn_out_dim + n_additional,
        num_layers=mlp_num_layers,
        num_neurons=mlp_num_neurons,
    )

    model = CombinedModel(cnn, mlp).to(device)
    return model


def build_multiclass_model(n_additional: int,
                           cnn_out_dim: int = 10,
                           n_classes: int = 3) -> CombinedModel:
    """
    Build the CombinedModel for 3-class classification:
    - ResNet18 -> cnn_out_dim
    - MLP with [8, n_classes] neurons.
    """
    cnn = ModifiedResNet18(num_classes=cnn_out_dim)
    mlp_num_layers = 2
    mlp_num_neurons: Sequence[int] = [8, n_classes]

    mlp = MLPModule(
        input_size=cnn_out_dim + n_additional,
        num_layers=mlp_num_layers,
        num_neurons=mlp_num_neurons,
    )

    model = CombinedModel(cnn, mlp).to(device)
    return model
