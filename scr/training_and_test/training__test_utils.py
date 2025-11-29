# src/training_and_test/training__test_utils.py

from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

# Device globale (se ti serve altrove)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# Dataset: immagini + feature tabellari
# ----------------------------------------------------------------------
class CustomDataset(Dataset):
    """
    Dataset per immagini + feature aggiuntive.

    Parameters
    ----------
    subjects : Sequence[str]
        Lista dei nomi (valori della colonna 'Name') da usare.
    features_dataframe : pd.DataFrame
        DataFrame con almeno le colonne: 'Name', 'Class', 'Path'
        e le feature numeriche.
    transform : callable, optional
        Trasformazioni da applicare all'immagine (torchvision transforms).
    """
    def __init__(self, subjects, features_dataframe, transform=None):
        self.subjects: List[str] = list(subjects)
        # indicizziamo il DF per Name per lookup O(1)
        self.df = features_dataframe.set_index('Name')
        self.transform = transform

        # individua le colonne di feature (escludi Class e Path)
        self.feature_cols = [
            c for c in self.df.columns
            if c not in ['Class', 'Path']
        ]

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int):
        name = self.subjects[idx]
        row = self.df.loc[name]

        # immagine
        image_path = row['Path']
        image = Image.open(image_path).convert('L')  # grayscale, poi Grayscale(3) nel transform

        # feature aggiuntive
        additional_features = torch.tensor(
            row[self.feature_cols].values.astype('float32'),
            dtype=torch.float32
        )

        # label
        label = torch.tensor(int(row['Class']), dtype=torch.int64)

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
    ResNet18 con ultimo layer fully connected sostituito.
    `num_classes` qui significa dimensione dell'output (feature o logit).
    """
    def __init__(self, num_classes: int = 1000, pretrained: bool = True):
        super(ModifiedResNet18, self).__init__()
        resnet = resnet18(pretrained=pretrained)

        # usa tutte le layer eccetto l'ultimo FC
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # nuovo fully connected finale
        self.new_fc_layer = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)   # (batch_size, 512)
        x = self.new_fc_layer(x)    # (batch_size, num_classes)
        return x


class MLPModule(nn.Module):
    """
    MLP generico per le feature aggiuntive (o per feature combinate).
    """
    def __init__(self, input_size: int, num_layers: int, num_neurons: Sequence[int]):
        super(MLPModule, self).__init__()
        layers = []

        # primo layer
        layers.append(nn.Linear(input_size, num_neurons[0]))
        layers.append(nn.ReLU())

        # layer successivi
        for i in range(1, num_layers):
            layers.append(nn.Linear(num_neurons[i - 1], num_neurons[i]))
            if i != num_layers - 1:
                layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class CombinedModel(nn.Module):
    """
    Modello combinato: CNN (immagine) + MLP (immagine + feature).
    - La CNN produce un vettore di feature per immagine.
    - Viene concatenato alle feature tabellari.
    - L'MLP produce logit (B, 1) per binario o (B, C) per multiclasse.
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
# Transforms per immagini
# ----------------------------------------------------------------------
composer = transforms.Compose([
    transforms.Resize((250, 250)),
    transforms.Grayscale(num_output_channels=3),  # converte 1 canale -> 3 canali per ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.456, 0.456, 0.456],
                         std=[0.225, 0.225, 0.225])
])


# ----------------------------------------------------------------------
# Inizializzazione pesi (He / Kaiming)
# ----------------------------------------------------------------------
def he_init_weight(tensor: torch.Tensor) -> None:
    """
    Inizializzazione He (Kaiming) per pesi di layer fully connected.
    Modifica il tensore in-place.
    """
    if tensor.ndimension() < 2:
        raise ValueError("he_init_weight expects a weight tensor with at least 2 dimensions")
    fan_in = tensor.size(1)  # per Linear: (out_features, in_features)
    std = np.sqrt(2.0 / fan_in)
    with torch.no_grad():
        tensor.normal_(0.0, std)


def reinit_weights(model: CombinedModel, seed: int) -> CombinedModel:
    """
    Reinizializza i pesi dell'ultimo layer della CNN e di tutti i layer lineari dell'MLP.

    - per binario: ultimo layer MLP deve avere 1 neurone (1 logit, BCEWithLogitsLoss)
    - per multiclasse: ultimo layer MLP deve avere C neuroni (C logit, CrossEntropyLoss)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ultimo layer della CNN (es. new_fc_layer)
    layers = list(model.cnn.children())
    last_layer = layers[-1]
    if isinstance(last_layer, nn.Linear):
        he_init_weight(last_layer.weight)
        if last_layer.bias is not None:
            nn.init.zeros_(last_layer.bias)

    # tutti i layer lineari dell'MLP
    for m in model.mlp.modules():
        if isinstance(m, nn.Linear):
            he_init_weight(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model = model.float()

    # opzionale ma utile per riproducibilità
    torch.save(model.state_dict(), 'init_model.pth')
    return model


# ----------------------------------------------------------------------
# Predizioni (compatibile BCE binaria e CE multiclasse)
# ----------------------------------------------------------------------
def get_trials_prediction(model: nn.Module,
                          test_dl: DeviceDataLoader,
                          classes,
                          encoder) -> tuple:
    """
    Esegue la predizione sul dataloader di test e restituisce:
    - all_labels : np.ndarray shape (N,)
    - all_preds  : np.ndarray shape (N,)
    - all_probs  : np.ndarray shape (N, C) dove:
        * C=2 per binario (probabilità [p(class0), p(class1)])
        * C>2 per multiclasse (softmax)
    - all_features : np.ndarray shape (N, n_features_tabellari)

    Logica:
    - se output del modello ha shape (B, 1)  -> assume BCE binaria
    - se output del modello ha shape (B, C) con C>1 -> CE multiclasse
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
            labels = batch['labels']

            outputs = model(images, additional_features)  # (B,1) o (B,C)

            # assicuriamoci che sia 2D
            if outputs.ndim == 1:
                outputs = outputs.unsqueeze(1)

            B, C = outputs.shape

            if C == 1:
                # ----- CASO BINARIO: BCEWithLogitsLoss -----
                logits = outputs.squeeze(1)              # (B,)
                probs_pos = torch.sigmoid(logits)        # (B,)

                # costruiamo [p0, p1] = [1-p, p]
                probs_two = torch.stack(
                    [1.0 - probs_pos, probs_pos],
                    dim=1
                )                                        # (B,2)

                preds = (probs_pos >= 0.5).long()        # (B,)
                probs_np = probs_two.cpu().numpy()
            else:
                # ----- CASO MULTICLASSE: CrossEntropyLoss -----
                probs_t = torch.nn.functional.softmax(outputs, dim=1)
                probs_np = probs_t.cpu().numpy()         # (B,C)
                preds = probs_t.argmax(dim=1)            # (B,)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs_t.extend(probs_np)
            all_features.extend(additional_features.detach().cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs_t = np.array(all_probs_t)
    all_features = np.array(all_features)

    return all_labels, all_preds, all_probs_t, all_features


# ----------------------------------------------------------------------
# Confusion matrix helper (opzionale)
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

