# src/training/training_utils.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from PIL import Image
import os

# Set up the device for training (cuda if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class for loading images and additional features
class CustomDataset(Dataset):
    def __init__(self, subjects, features_dataframe, transform=None):
        self.subjects = subjects
        self.features_dataframe = features_dataframe
        self.transform = transform

    def __len__(self):
        return len(self.features_dataframe)

    def __getitem__(self, idx):
        # Load image
        image_path = self.features_dataframe.loc[self.features_dataframe['Name'] == self.subjects[idx]]['Path'].values[0]
        image = Image.open(image_path)

        # Load additional features (excluding 'Name' and 'Class' columns)
        additional_features = torch.tensor(
            self.features_dataframe.loc[self.features_dataframe['Name'] == self.subjects[idx]].iloc[:, 1:-2].values[0].astype('float32'),
            dtype=torch.float32
        )

        # Labels for classification
        labels = torch.tensor(self.features_dataframe.loc[self.features_dataframe['Name'] == self.subjects[idx]]['Class'].values[0], dtype=torch.int64)

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return {'image': image, 'additional_features': additional_features, 'labels': labels}

# Data loader for transferring data to the GPU
class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield {k: v.to(self.device) for k, v in b.items()}

    def __len__(self):
        return len(self.dl)

# Pre-trained ResNet18 model modification for feature extraction and custom final layer
class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModifiedResNet18, self).__init__()
        # Load pre-trained ResNet18
        resnet = resnet18(pretrained=True)
        
        # Use all layers except the last fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Custom final fully connected layer
        self.new_fc_layer = nn.Linear(512, num_classes)

    def forward(self, x):
        # Forward pass through feature extractor (ResNet layers)
        x = self.features(x)
        
        # Flatten the output from 4D to 2D (batch_size, 512)
        x = x.view(x.size(0), -1)
        
        # Pass through the new fully connected layer
        x = self.new_fc_layer(x)
        return x

# MLP Module for additional features
class MLPModule(nn.Module):
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

# Combined model that uses both CNN and MLP modules
class CombinedModel(nn.Module):
    def __init__(self, cnn, mlp):
        super(CombinedModel, self).__init__()
        self.cnn = cnn
        self.mlp = mlp

    def forward(self, image_input, additional_input):
        # Extract features from CNN
        cnn_features = self.cnn(image_input)
        
        # Concatenate CNN features and additional features
        combined_features = torch.cat((cnn_features, additional_input), dim=1)
        
        # Pass combined features through MLP
        output = self.mlp(combined_features)
        return output

# Define a transform for resizing and normalizing images
composer = transforms.Compose([
    transforms.Resize((250, 250)),
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.456, 0.456, 0.456], std=[0.225, 0.225, 0.225])
])

# Reinitialize weights of the model (He initialization)
def he_init(shape):
    return np.random.randn(*shape) * np.sqrt(2 / shape[0])

def reinit_weights(model, seed):
    # Set the random seed
    np.random.seed(seed)

    # Reinitialize weights of the last layer of CNN and MLP
    layers = list(model.cnn.children())
    last_n_layers = layers[-1:]

    for layer in last_n_layers:
        if isinstance(layer, torch.nn.Linear):
            layer.weight.data = torch.FloatTensor(he_init(layer.weight.size()))

    for param in model.mlp.parameters():
        param.data = torch.from_numpy(he_init(param.size()))

    model = model.float()

    # Save parameters
    torch.save(model.state_dict(), 'init_model.pth')
    return model

# Get predictions and calculate evaluation metrics
def get_trials_prediction(model, test_dl, classes, encoder):
    all_preds = []
    all_labels = []
    all_probs_t = []
    all_features = []

    for batch in test_dl:
        images = batch['image']
        additional_features = batch['additional_features']
        labels = batch['labels'].float().detach().cpu().numpy()
        outputs = model(images, additional_features)
        probs = torch.nn.functional.softmax(outputs, dim=1).detach().cpu()
        preds = np.array([tensor.argmax().item() for tensor in probs])
        all_preds.extend(preds)
        all_labels.extend(labels)
        all_probs_t.extend(np.array(probs))
        all_features.extend(additional_features.detach().cpu().numpy())

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs_t = np.array(all_probs_t)
    all_features = np.array(all_features)

    return all_labels, all_preds, all_probs_t, all_features

# Normalize confusion matrix and plot
def plot_confusion_matrix(cm, classes):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize by rows
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
