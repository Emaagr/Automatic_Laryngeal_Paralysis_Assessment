"""
Shared features useful for the two classifications (3-class and 2-class variants) inluding model, dataset, and data-loading.

This file contains ONLY code that is common between the two
pipelines and does not change their logic.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, Grayscale
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image


# -------------------------------------------------------
# Device + transforms
# -------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same transformation as in the original notebooks
composer = Compose([
    Resize((250, 250)),
    Grayscale(num_output_channels=3),  # Convert to RGB if images are grayscale
    ToTensor(),
    Normalize(mean=[0.456, 0.456, 0.456], std=[0.225, 0.225, 0.225]),
])


def set_global_seed(seed: int) -> None:
    """
    Set all global random seeds (torch, numpy, CUDA) to make behavior reproducible.
    """
    import random

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------
# Dataset
# -------------------------------------------------------

class CustomDataset(Dataset):
    """
    Dataset that returns:
        - transformed image
        - additional numeric features (from feature table)
        - label (optionally transformed, e.g. for binary classification)

    Note:
    - This preserves the logic of the two original notebooks:
      * In 3-class: labels are used as-is (0,1,2).
      * In 2-class: labels are binarized (Class > 0).
    """

    def __init__(self, subjects, features_dataframe, transform=None, label_transform=None):
        """
        Parameters
        ----------
        subjects : list of str
            List of feature table "Name" values in the desired order.
        features_dataframe : pd.DataFrame
            Full feature table.
        transform : callable or None
            Torchvision-like transform applied to the PIL image.
        label_transform : callable or None
            Function applied to the original label to obtain the final label.
            If None, labels are left unchanged.
        """
        self.subjects = subjects
        self.features_dataframe = features_dataframe
        self.transform = transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.features_dataframe)

    def __getitem__(self, idx):
        # Look up the row corresponding to this subject
        name = self.subjects[idx]
        row = self.features_dataframe.loc[self.features_dataframe['Name'] == name].iloc[0]

        # Load image (same as original code)
        image_path = row['Path']
        image = Image.open(image_path)

        # Load additional features: all columns between 'Name' and 'Class' (excluded)
        additional_features = torch.tensor(
            row.iloc[1:-2].values.astype('float32'),
            dtype=torch.float32
        )

        # Load label
        label = row['Class']
        if self.label_transform is not None:
            label = self.label_transform(label)
        label = torch.tensor(label, dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'additional_features': additional_features,
            'labels': label
        }


# -------------------------------------------------------
# Models
# -------------------------------------------------------

class ModifiedResNet18(nn.Module):
    """
    ResNet18 with a replaceable final fully-connected layer.

    This is the same architecture as in the original notebooks.
    """

    def __init__(self, num_classes=1000):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        # All layers except the last fully connected (fc)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # New fully connected layer to produce num_classes outputs
        self.new_fc_layer = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        # Flatten [batch_size, 512, 1, 1] -> [batch_size, 512]
        x = x.view(x.size(0), -1)
        x = self.new_fc_layer(x)
        return x


class MLPModule(nn.Module):
    """
    Simple MLP with configurable layer sizes, as in original code.
    """

    def __init__(self, input_size, num_layers, num_neurons):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_size, num_neurons[0]))
        layers.append(nn.ReLU())
        for i in range(1, num_layers):
            layers.append(nn.Linear(num_neurons[i - 1], num_neurons[i]))
            if i != num_layers - 1:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class CombinedModel(nn.Module):
    """
    Combined model that:
        - passes the image through the CNN
        - concatenates CNN features with additional numeric features
        - passes both through an MLP
    """

    def __init__(self, cnn: nn.Module, mlp: nn.Module):
        super().__init__()
        self.cnn = cnn
        self.mlp = mlp

    def forward(self, image_input, additional_input):
        cnn_features = self.cnn(image_input)
        combined_features = torch.cat((cnn_features, additional_input), dim=1)
        output = self.mlp(combined_features)
        return output


# -------------------------------------------------------
# DataLoader utility
# -------------------------------------------------------

class DeviceDataLoader:
    """
    Wrap a DataLoader to move its batches to a given device automatically.
    """

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield {k: v.to(self.device) for k, v in b.items()}

    def __len__(self):
        return len(self.dl)
