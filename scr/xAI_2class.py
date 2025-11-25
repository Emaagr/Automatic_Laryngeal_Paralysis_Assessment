"""
XAI script for 2-class classification (Normal / Paralyzed).

Main steps:
- Load Agati features table and split into train/val/test (stratified by Class)
- Build a CustomDataset that returns:
    - transformed image (for model input)
    - additional features
    - binary labels (0 / 1, where Class > 0 -> 1)
    - original resized image (for visualization)
- Rebuild the same CombinedModel (CNN + MLP) used for 2-class training
- Load pretrained weights (combined_model_weights_due_classi.pth)
- Evaluate on test set and compute confusion matrix (threshold logit > 0)
- Run Integrated Gradients (Captum) to obtain:
    - image attributions
    - feature attributions
- Save attributions to .npy and optionally visualize one example

NOTE:
- This script assumes you have `models_common.py` in the same project,
  defining:
    - device
    - composer (Resize+Gray+Normalize)
    - set_global_seed
    - ModifiedResNet18
    - MLPModule
    - CombinedModel
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from captum.attr import IntegratedGradients

from models_common import (
    device,
    composer,           # image transforms (Resize+Gray+Normalize)
    set_global_seed,
    ModifiedResNet18,
    MLPModule,
    CombinedModel,
)

# --------------------------------------------------
# Paths (ADJUST THESE FOR YOUR ENVIRONMENT)
# --------------------------------------------------

DATA_DIR = "/content/drive/MyDrive/AI in healthcare/Laryngeal paralysis/Spain Dataset/Trials"
FEATURES_XLSX = os.path.join(DATA_DIR, "Agati features.xlsx")
WEIGHTS_PATH = "/content/drive/MyDrive/AI in healthcare/Laryngeal paralysis/Weights/combined_model_weights_due_classi.pth"
ATTRIBUTIONS_OUT = "/content/drive/MyDrive/AI in healthcare/Laryngeal paralysis/Weights/attributions_bin.npy"


# --------------------------------------------------
# Dataset for XAI (binary)
# --------------------------------------------------

class XAIDatasetBinary(Dataset):
    """
    Dataset that:
    - loads image from 'Path' in the features DataFrame
    - returns:
        * 'image'  : transformed image tensor (for model)
        * 'additional_features': feature vector (float32)
        * 'labels': binary label (0/1) as float32 (Class > 0)
        * 'original': resized image tensor (C,H,W) for visualization
    """

    def __init__(self, subjects, features_dataframe, transform=None):
        self.subjects = subjects
        self.features_dataframe = features_dataframe
        self.transform = transform
        self.resize = transforms.Resize((250, 250))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.features_dataframe)

    def __getitem__(self, idx):
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
        if self.transform is not None:
            image_t = self.transform(image)
        else:
            image_t = original_tensor

        return {
            "image": image_t,
            "additional_features": additional_features,
            "labels": label_tensor,
            "original": original_tensor,
        }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    # ---------------- Seed ----------------
    seed = 42
    set_global_seed(seed)

    # ---------------- Load features ----------------
    featurestable = pd.read_excel(FEATURES_XLSX)

    # First split: train/test stratified by original Class (0,1,2)
    subjects_all = list(featurestable["Name"])
    classes_all = list(featurestable["Class"])

    subjects_train_all, subjects_test = train_test_split(
        subjects_all,
        test_size=0.2,
        random_state=42,
        stratify=classes_all,
    )

    # From train_all, split into train/val (again stratified by Class)
    features_train_all = featurestable[featurestable["Name"].isin(subjects_train_all)]
    names_train_all = list(features_train_all["Name"])
    classes_train_all = list(features_train_all["Class"])

    subjects_train, subjects_val = train_test_split(
        names_train_all,
        test_size=0.2,
        random_state=42,
        stratify=classes_train_all,
    )

    # Build final DataFrames
    features_train = featurestable[featurestable["Name"].isin(subjects_train)]
    features_val = featurestable[featurestable["Name"].isin(subjects_val)]
    features_test = featurestable[featurestable["Name"].isin(subjects_test)]

    # ---------------- Datasets & loaders ----------------
    train_loader = DataLoader(
        XAIDatasetBinary(subjects=subjects_train, features_dataframe=features_train, transform=composer),
        batch_size=8,
        shuffle=True,
    )
    val_loader = DataLoader(
        XAIDatasetBinary(subjects=subjects_val, features_dataframe=features_val, transform=composer),
        batch_size=8,
        shuffle=True,
    )
    test_loader = DataLoader(
        XAIDatasetBinary(subjects=subjects_test, features_dataframe=features_test, transform=composer),
        batch_size=8,
        shuffle=False,
    )

    # ---------------- Rebuild model (same architecture as 2-class model) ----------------
    # CNN output dim: 10 (as in notebook)
    cnn_out_dim = 10
    cnn = ModifiedResNet18(num_classes=cnn_out_dim)

    # Additional features count
    cols = list(features_train.columns)
    n_additional = cols.index("Class") - cols.index("Name") - 1

    mlp_num_layers = 2
    mlp_num_neurons = [8, 1]  # single logit output

    mlp = MLPModule(
        input_size=cnn_out_dim + n_additional,
        num_layers=mlp_num_layers,
        num_neurons=mlp_num_neurons,
    )

    combined_model = CombinedModel(cnn, mlp).to(device)

    # Load trained weights
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    combined_model.load_state_dict(state_dict)
    combined_model.eval()

    # ---------------- Evaluation on test set (confusion matrix, logit > 0) ----------------
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            additional_features = batch["additional_features"].to(device)
            labels = batch["labels"].to(device)  # 0/1 float

            outputs = combined_model(images, additional_features)  # shape (B,1)
            logits = outputs.view(-1)
            preds = (logits > 0).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds).astype(int)
    all_labels = np.array(all_labels).astype(int)

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion matrix (binary trials):")
    print(cm)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (2-class)")
    plt.tight_layout()
    plt.show()

    # ---------------- Integrated Gradients (Captum) ----------------
    # One batch as in notebook
    first_batch = next(iter(test_loader))
    images = first_batch["image"].to(device)
    additional_features = first_batch["additional_features"].to(device)
    labels = first_batch["labels"].to(device)  # 0/1
    originals = first_batch["original"]         # keep on CPU

    # Baselines (zeros)
    baseline_images = torch.zeros_like(images).to(device)
    baseline_features = torch.zeros_like(additional_features).to(device)

    ig = IntegratedGradients(combined_model)

    # Single logit output => no need to specify target
    attributions, delta = ig.attribute(
        inputs=(images, additional_features),
        baselines=(baseline_images, baseline_features),
        internal_batch_size=1,
        n_steps=100,
        method="gausslegendre",
        return_convergence_delta=True,
    )

    # Tuple: (image_attr, feature_attr)
    attr_images = attributions[0].detach().cpu().numpy()
    attr_features = attributions[1].detach().cpu().numpy()

    # Save as .npy (stack like [2, batch, ...])
    np.save(ATTRIBUTIONS_OUT, np.stack([attr_images, attr_features], axis=0))
    print(f"Saved binary attributions to: {ATTRIBUTIONS_OUT}")

    # ---------------- Example visualization ----------------
    idx = 0
    selected_original = originals[idx].numpy()
    selected_attr_img = attr_images[idx]

    # Sum channels to get a single heatmap
    heatmap = np.sum(selected_attr_img, axis=0)

    # Plot original (pre-transform)
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(selected_original.transpose(1, 2, 0), cmap="gray")
    plt.title(f"Original (class={int(labels[idx].item())})")
    plt.axis("off")

    # Plot IG heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap, cmap="viridis")
    plt.title("IG Attributions (image)")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

    # Feature attributions for same sample
    feat_attr = attr_features[idx]
    feature_names = cols[cols.index("Name") + 1:cols.index("Class")]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        feat_attr.reshape(-1, 1),
        annot=True,
        fmt=".3f",
        yticklabels=feature_names,
        xticklabels=["Attribution"],
        cmap="coolwarm",
    )
    plt.title("Integrated Gradients – Feature Attributions (binary)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
