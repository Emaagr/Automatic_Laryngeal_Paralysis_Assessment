# src/training/utils.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, Grayscale
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import numpy as np

# -------------------------------------------------------
# Dataset
# -------------------------------------------------------

class CustomDataset(Dataset):
    """
    Dataset that returns:
        - transformed image
        - additional numeric features (from feature table)
        - label (optionally transformed, e.g. for binary classification)
    """
    def __init__(self, subjects, features_dataframe, transform=None, label_transform=None):
        self.subjects = subjects
        self.features_dataframe = features_dataframe
        self.transform = transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.features_dataframe)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.features_dataframe.loc[self.features_dataframe['Name'] == self.subjects[idx]]['Path'].values[0])

        # Load additional features
        additional_features = torch.tensor(
            self.features_dataframe.loc[self.features_dataframe['Name'] == self.subjects[idx]].iloc[:, 1:-2].values[0].astype('float32'),
            dtype=torch.float32
        )

        # Load label
        label = torch.tensor(self.features_dataframe.loc[self.features_dataframe['Name'] == self.subjects[idx]]['Class'].values[0], dtype=torch.int64)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return {'image': image, 'additional_features': additional_features, 'labels': label}


# -------------------------------------------------------
# CNN and MLP Models
# -------------------------------------------------------

class ModifiedResNet18(nn.Module):
    """
    ResNet18 with a replaceable final fully-connected layer.
    """
    def __init__(self, num_classes=1000):
        super(ModifiedResNet18, self).__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Copy all layers except the last fully connected layer (fc)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Define a new fully connected layer with the desired output size
        self.new_fc_layer = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)  # Extract features
        x = x.view(x.size(0), -1)  # Flatten the output (assuming the output shape is [batch_size, 512, 1, 1])
        x = self.new_fc_layer(x)  # Pass through the new fully connected layer
        return x


class MLPModule(nn.Module):
    """
    MLP with configurable layer sizes.
    """
    def __init__(self, input_size, num_layers, num_neurons):
        super(MLPModule, self).__init__()
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
# Device DataLoader
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
