# src/training/training_utils.py

import os
from typing import List, Sequence, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

# Set up the device for training (cuda if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class CustomDataset(Dataset):
    """
    Dataset per immagini + feature aggiuntive.

    Parameters
    ----------
    subjects : Sequence[str]
        Lista dei nomi (valori della colonna 'Name') da usare.
    features_dataframe : pd.DataFrame
        DataFrame con almeno le colonne: 'Name', 'Class', 'Path' e le feature numeriche.
    transform : callable, optional
        Trasformazioni da applicare all'immagine (torchvision transforms).
    """
    def __init__(self, subjects, features_dataframe, transform=None):
        self.subjects: List[str] = list(subjects)
        # indicizziamo il DF per Name per lookup O(1)
        self.df = features_dataframe.set_index('Name')
        self.transform = transform

        # individua le colonne di feature (escludi Name, Class, Path)
        self.feature_cols = [
            c for c in self.df.columns
            if c not in ['Class', 'Path']
        ]

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int):
        name = self.subjects[idx]
        row = self.df.loc[name]

        # Load image
        image_path = row['Path']
        image = Image.open(image_path).convert('L')  # grayscale, poi Grayscale(3) nel transform

        # Additional features
        additional_features = torch.tensor(
            row[self.feature_cols].values.astype('float32'),
            dtype=torch.float32
        )

        # Label
        label = torch.tensor(int(row['Class']), dtype=torch.int64)

        # Transform image
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'additional_features': additional_features,
            'labels': label
        }


# ----------------------------------------------------------------------
# DataLoader su device
# ----------------------------------------------------------------------
class DeviceDataLoader:
    """Wrapper per spostare automaticamente i batch su device (cuda/cpu)."""
    def __init__(self, dl: DataLoader, device: torch.device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield {k: v.to(self.device) for k, v in b.items()}

    def __len__(self):
        return len(self.dl)


# ----------------------------------------------------------------------
# Modelli: CNN, MLP, Combined
# ----------------------------------------------------------------------
class ModifiedResNet18(nn.Module):
    """
    ResNet18 pre-addestrata (o meno) con ultimo layer fully connected sostituito.
    L'output di questa rete viene usato come feature dal modello combinato.
    """
    def __init__(self, num_classes: int = 1000, pretrained: bool = True):
        super(ModifiedResNet18, self).__init__()
        # Nota: per compatibilità con versioni più nuove di torchvision
        # si potrebbe usare il parametro 'weights', ma qui manteniamo
        # la firma classica.
        resnet = resnet18(pretrained=pretrained)

        # Usa tutte le layer eccetto l'ultimo FC
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Nuovo fully connected finale
        self.new_fc_layer = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)   # (batch_size, 512)
        x = self.new_fc_layer(x)    # (batch_size, num_classes) o feature_dim
        return x


class MLPModule(nn.Module):
    """
    MLP generico per le feature aggiuntive (o per feature combinate).
    """
    def __init__(self, input_size: int, num_layers: int, num_neurons: Sequence[int]):
        super(MLPModule, self).__init__()
        layers = []

        # Primo layer
        layers.append(nn.Linear(input_size, num_neurons[0]))
        layers.append(nn.ReLU())

        # Layer successivi
        for i in range(1, num_layers):
            layers.append(nn.Linear(num_neurons[i - 1], num_neurons[i]))
            if i != num_layers - 1:
                layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class CombinedModel(nn.Module):
    """
    Modello combinato: CNN (immagine) + MLP (immagine+feature).
    """
    def __init__(self, cnn: nn.Module, mlp: nn.Module):
        super(CombinedModel, self).__init__()
        self.cnn = cnn
        self.mlp = mlp

    def forward(self, image_input: torch.Tensor, additional_input: torch.Tensor) -> torch.Tensor:
        cnn_features = self.cnn(image_input)
        combined_features = torch.cat((cnn_features, additional_input), dim=1)
        output = self.mlp(combined_features)
        return output


# ----------------------------------------------------------------------
# Transforms per le immagini
# ----------------------------------------------------------------------
composer = transforms.Compose([
    transforms.Resize((250, 250)),
    transforms.Grayscale(num_output_channels=3),  # converte 1 canale -> 3 canali per ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.456, 0.456, 0.456],
                         std=[0.225, 0.225, 0.225])
])


# ----------------------------------------------------------------------
# Inizializzazione pesi
# ----------------------------------------------------------------------
def he_init_weight(tensor: torch.Tensor) -> None:
    """
    Inizializzazione He (Kaiming) per pesi di layer fully connected.
    Modifica il tensore in-place.
    """
    if tensor.ndimension() < 2:
        raise ValueError("he_init_weight expects a weight tensor with at least 2 dimensions")
    fan_in = tensor.size(1)
    std = np.sqrt(2.0 / fan_in)
    with torch.no_grad():
        tensor.normal_(0.0, std)


def reinit_weights(model: CombinedModel, seed: int) -> CombinedModel:
    """
    Reinizializza i pesi dell'ultimo layer della CNN e di tutti i layer lineari dell'MLP.

    Parameters
    ----------
    model : CombinedModel
        Modello combinato CNN + MLP.
    seed : int
        Seed per la randomizzazione.

    Returns
    -------
    CombinedModel
        Il modello con pesi reinizializzati (in float32 su CPU di default).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Reinit dell'ultimo layer lineare della CNN (es. new_fc_layer)
    layers = list(model.cnn.children())
    last_layer = layers[-1]
    if isinstance(last_layer, nn.Linear):
        he_init_weight(last_layer.weight)
        if last_layer.bias is not None:
            nn.init.zeros_(last_layer.bias)

    # Reinit dei layer lineari dell'MLP
    for m in model.mlp.modules():
        if isinstance(m, nn.Linear):
            he_init_weight(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model = model.float()

    # Salva lo stato iniziale (opzionale, ma utile per riproducibilità)
    torch.save(model.state_dict(), 'init_model.pth')
    return model


# ----------------------------------------------------------------------
# Predizioni su trials
# ----------------------------------------------------------------------
def get_trials_prediction(model: nn.Module,
                          test_dl: DeviceDataLoader,
                          classes,
                          encoder) -> tuple:
    """
    Esegue la predizione sul dataloader di test e restituisce etichette,
    predizioni, probabilità e feature aggiuntive.

    Parameters
    ----------
    model : nn.Module
        Modello addestrato.
    test_dl : DeviceDataLoader
        Dataloader dei batch di test (su device).
    classes, encoder :
        Parametri mantenuti per compatibilità con versioni precedenti (non usati).

    Returns
    -------
    all_labels : np.ndarray
    all_preds : np.ndarray
    all_probs_t : np.ndarray
    all_features : np.ndarray
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs_t = []
    all_features = []

    with torch.no_grad():
        for batch in test_dl:
            images = batch['image']
            additional_features = batch['additional_features']
            labels = batch['labels'].detach().cpu().numpy()

            outputs = model(images, additional_features)
            probs = torch.nn.functional.softmax(outputs, dim=1).detach().cpu()

            preds = probs.argmax(dim=1).numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs_t.extend(probs.numpy())
            all_features.extend(additional_features.detach().cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs_t = np.array(all_probs_t)
    all_features = np.array(all_features)

    return all_labels, all_preds, all_probs_t, all_features


# ----------------------------------------------------------------------
# Confusion matrix
# ----------------------------------------------------------------------
def plot_confusion_matrix(cm: np.ndarray, classes: Sequence[str]) -> None:
    """
    Normalizza la confusion matrix per riga e la plottizza con seaborn.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix (shape: [n_classes, n_classes]).
    classes : Sequence[str]
        Nomi delle classi (labels per assi).
    """
    cm = cm.astype('float')
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    # evita divisione per zero
    row_sums[row_sums == 0] = 1.0
    cm_normalized = cm / row_sums

    plt.figure()
    sns.heatmap(cm_normalized, annot=True, fmt=".2f",
                cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

