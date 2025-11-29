# src/xai/xai_3class.py
"""
XAI pipeline for 3-class classification (Normal / Monolateral / Bilateral).

Main steps:
- Load Agati features table and split into train/val/test (stratified by Class)
- Build XAIDatasetMulticlass
- Rebuild the same CombinedModel (CNN + MLP) used for 3-class training
- Load pretrained weights
- Evaluate on test set and compute confusion matrix
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
    WEIGHTS_3CLASS,
    ATTRIBUTIONS_3CLASS,
    SEED,
)
from .xai_utils import (
    XAIDatasetMulticlass,
    build_multiclass_model,
)
from ..training.training_test_utils import (
    device,
    set_global_seed,
)


def run_xai_3class():
    """
    High-level entry point for the 3-class XAI pipeline.
    """

    # ---------------- Seed ----------------
    set_global_seed(SEED)

    # ---------------- Load features ----------------
    featurestable = pd.read_excel(FEATURES_XLSX)

    # Split as in the notebook: stratify by Class
    subjects_all = list(featurestable["Name"])
    classes_all = list(featurestable["Class"])

    subjects_train, subjects_test = train_test_split(
        subjects_all,
        test_size=0.2,
        random_state=SEED,
        stratify=classes_all,
    )

    # Rebuild DataFrames for splits
    features_train = featurestable[featurestable["Name"].isin(subjects_train)]
    features_test = featurestable[featurestable["Name"].isin(subjects_test)]

    # Second split: train -> train + val (again stratified)
    train_classes = list(
        features_train.sort_values("Name")["Class"]
    )
    train_names_sorted = list(features_train.sort_values("Name")["Name"])

    subjects_train_final, subjects_val = train_test_split(
        train_names_sorted,
        test_size=0.2,
        random_state=SEED,
        stratify=train_classes,
    )

    features_train_final = features_train[features_train["Name"].isin(subjects_train_final)]
    features_val = features_train[features_train["Name"].isin(subjects_val)]

    # ---------------- Datasets & loaders ----------------
    train_loader = DataLoader(
        XAIDatasetMulticlass(
            subjects=subjects_train_final,
            features_dataframe=features_train_final,
        ),
        batch_size=8,
        shuffle=True,
    )
    val_loader = DataLoader(
        XAIDatasetMulticlass(
            subjects=subjects_val,
            features_dataframe=features_val,
        ),
        batch_size=8,
        shuffle=True,
    )
    test_loader = DataLoader(
        XAIDatasetMulticlass(
            subjects=subjects_test,
            features_dataframe=features_test,
        ),
        batch_size=8,
        shuffle=False,
    )

    # ---------------- Rebuild model (same architecture as 3-class model) ----------------
    cols = list(features_train_final.columns)
    n_additional = cols.index("Class") - cols.index("Name") - 1

    model = build_multiclass_model(n_additional=n_additional,
                                   cnn_out_dim=10,
                                   n_classes=3)

    # Load trained weights
    state_dict = torch.load(WEIGHTS_3CLASS, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # ---------------- Evaluation on test set (confusion matrix) ----------------
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            additional_features = batch["additional_features"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(images, additional_features)
            probs = F.softmax(outputs, dim=1)

            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion matrix (trials, 3-class):")
    print(cm)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (3-class)")
    plt.tight_layout()
    plt.show()

    # ---------------- Integrated Gradients (Captum) ----------------
    # One batch as example
    first_batch = next(iter(test_loader))
    images = first_batch["image"].to(device)
    additional_features = first_batch["additional_features"].to(device)
    targets = first_batch["labels"].to(device)
    originals = first_batch["original"]  # CPU

    baseline_images = torch.zeros_like(images).to(device)
    baseline_features = torch.zeros_like(additional_features).to(device)

    ig = IntegratedGradients(model)

    attributions, delta = ig.attribute(
        (images, additional_features),
        baselines=(baseline_images, baseline_features),
        target=targets,  # per-sample target
        method="gausslegendre",
        return_convergence_delta=True,
    )

    # attributions is a tuple: (attr_images, attr_features)
    attr_images = attributions[0].detach().cpu().numpy()
    attr_features = attributions[1].detach().cpu().numpy()

    ATTRIBUTIONS_3CLASS.parent.mkdir(parents=True, exist_ok=True)
    np.save(ATTRIBUTIONS_3CLASS, np.stack([attr_images, attr_features], axis=0))
    print(f"Saved attributions to: {ATTRIBUTIONS_3CLASS}")

    # ---------------- Example visualization ----------------
    idx = 0
    selected_original = originals[idx].numpy()
    selected_attr_img = attr_images[idx]

    # Sum channels of attribution to get single heatmap
    heatmap = np.sum(selected_attr_img, axis=0)

    # Plot original image
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(selected_original.transpose(1, 2, 0), cmap="gray")
    plt.title(f"Original (class={int(targets[idx])})")
    plt.axis("off")

    # Plot heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap, cmap="viridis")
    plt.title("IG Attributions (image)")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

    # Plot feature attributions for the same sample
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
    plt.title("Integrated Gradients – Feature Attributions (3-class)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_xai_3class()

