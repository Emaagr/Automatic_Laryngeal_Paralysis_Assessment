# src/xai/xai_2class.py
"""
XAI pipeline for 2-class classification (Normal / Paralyzed).

Main steps:
- Load Agati features table and split into train/val/test (stratified by Class)
- Build XAIDatasetBinary
- Rebuild the same CombinedModel (CNN + MLP) used for 2-class training
- Load pretrained weights
- Evaluate on test set and compute confusion matrix (threshold logit > 0)
- Run Integrated Gradients (Captum) to obtain:
    - image attributions
    - feature attributions
- Save attributions to .npy and optionally visualize one example
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from .config import (
    DATA_DIR,
    FEATURES_XLSX,
    WEIGHTS_2CLASS,
    ATTRIBUTIONS_2CLASS,
    SEED,
)
from .xai_utils import (
    XAIDatasetBinary,
    build_binary_model,
)
from ..training.training_utils import (
    device,
    set_global_seed,
)


def run_xai_2class():
    """
    High-level entry point for the 2-class XAI pipeline.
    Can be called from code or used as CLI via `if __name__ == "__main__":`.
    """
    # ---------------- Seed ----------------
    set_global_seed(SEED)

    # ---------------- Load features ----------------
    featurestable = pd.read_excel(FEATURES_XLSX)

    # Split: stratify by original Class (0,1,2)
    subjects_all = list(featurestable["Name"])
    classes_all = list(featurestable["Class"])

    subjects_train_all, subjects_test = train_test_split(
        subjects_all,
        test_size=0.2,
        random_state=SEED,
        stratify=classes_all,
    )

    # From train_all, split into train/val (again stratified by Class)
    features_train_all = featurestable[featurestable["Name"].isin(subjects_train_all)]
    names_train_all = list(features_train_all["Name"])
    classes_train_all = list(features_train_all["Class"])

    subjects_train, subjects_val = train_test_split(
        names_train_all,
        test_size=0.2,
        random_state=SEED,
        stratify=classes_train_all,
    )

    # Build final DataFrames
    features_train = featurestable[featurestable["Name"].isin(subjects_train)]
    features_val = featurestable[featurestable["Name"].isin(subjects_val)]
    features_test = featurestable[featurestable["Name"].isin(subjects_test)]

    # ---------------- Datasets & loaders ----------------
    train_loader = DataLoader(
        XAIDatasetBinary(
            subjects=subjects_train,
            features_dataframe=features_train,
        ),
        batch_size=8,
        shuffle=True,
    )
    val_loader = DataLoader(
        XAIDatasetBinary(
            subjects=subjects_val,
            features_dataframe=features_val,
        ),
        batch_size=8,
        shuffle=True,
    )
    test_loader = DataLoader(
        XAIDatasetBinary(
            subjects=subjects_test,
            features_dataframe=features_test,
        ),
        batch_size=8,
        shuffle=False,
    )

    # ---------------- Rebuild model (same architecture as 2-class model) ----------------
    # CNN output dim: 10 (as in the original notebook)
    cols = list(features_train.columns)
    n_additional = cols.index("Class") - cols.index("Name") - 1

    model = build_binary_model(n_additional=n_additional, cnn_out_dim=10)

    # Load trained weights
    state_dict = torch.load(WEIGHTS_2CLASS, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # ---------------- Evaluation on test set (confusion matrix, logit > 0) ----------------
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            additional_features = batch["additional_features"].to(device)
            labels = batch["labels"].to(device)  # 0/1 float

            outputs = model(images, additional_features)  # shape (B,1)
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
    # One batch as example
    first_batch = next(iter(test_loader))
    images = first_batch["image"].to(device)
    additional_features = first_batch["additional_features"].to(device)
    labels = first_batch["labels"].to(device)  # 0/1
    originals = first_batch["original"]         # CPU

    # Baselines (zeros)
    baseline_images = torch.zeros_like(images).to(device)
    baseline_features = torch.zeros_like(additional_features).to(device)

    ig = IntegratedGradients(model)

    # Single logit output => no target needed
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

    # Save as .npy
    ATTRIBUTIONS_2CLASS.parent.mkdir(parents=True, exist_ok=True)
    np.save(ATTRIBUTIONS_2CLASS, np.stack([attr_images, attr_features], axis=0))
    print(f"Saved binary attributions to: {ATTRIBUTIONS_2CLASS}")

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
    run_xai_2class()
