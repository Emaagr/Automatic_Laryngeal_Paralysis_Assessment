"""
3-class classification (Normal / Monolateral / Bilateral) pipeline.

This script reproduces the original CNN+MLP_3_Classes.ipynb including:
- data paths
- subject-level split strategy
- model architectures
- hyperparameter search
- evaluation loops
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader
from sklearn.metrics import RocCurveDisplay

from models_common import (
    device,
    composer,
    set_global_seed,
    CustomDataset,
    ModifiedResNet18,
    MLPModule,
    CombinedModel,
    DeviceDataLoader,
)


# -------------------------------------------------------
# Paths & data loading (1 dataset section, as in final code)
# -------------------------------------------------------

data_dir = "/content/drive/MyDrive/AI in healthcare/Laryngeal paralysis/Spain Dataset/Trials"
datasetpath = os.path.join(data_dir, "Difference Images")

featurestable = pd.read_excel(
    "/content/drive/MyDrive/AI in healthcare/Laryngeal paralysis/Spain Dataset/Trials/Agati features.xlsx"
)
# Remove velocity features (as in notebook)
featurestable.drop(
    columns=[
        "Velocity Correlation",
        "Velocity Granger Causality",
        "Velocity Minimum Std",
        "Velocity Diff Std",
    ],
    inplace=True,
)

valid_names = featurestable["Name"].dropna()
spa_subj = list(set([name[:6] for name in valid_names]))

featurestable["Name"].fillna("", inplace=True)
seed = 42
set_global_seed(seed)

# Create subject-level labels (3-class)
target = []
for subj in spa_subj:
    mask = featurestable["Name"].str.contains(subj)
    target.append(featurestable.loc[mask, "Class"].iloc[0])

# Train/test on subjects
subjects_train, subjects_test = train_test_split(
    spa_subj, test_size=0.2, random_state=seed, stratify=target
)

# Recompute labels for train to split into train/val
target = []
for subj in subjects_train:
    mask = featurestable["Name"].str.contains(subj)
    target.append(featurestable.loc[mask, "Class"].iloc[0])

subjects_train, subjects_val = train_test_split(
    subjects_train, test_size=0.2, random_state=seed, stratify=target
)

# Expand subject IDs into specific trial Names for each split
pattern = "|".join(subjects_train)
matching_rows = featurestable[featurestable["Name"].str.contains(pattern)]
subjects_train = matching_rows["Name"].tolist()

pattern = "|".join(subjects_val)
matching_rows = featurestable[featurestable["Name"].str.contains(pattern)]
subjects_val = matching_rows["Name"].tolist()

pattern = "|".join(subjects_test)
matching_rows = featurestable[featurestable["Name"].str.contains(pattern)]
subjects_test = matching_rows["Name"].tolist()

features_train = featurestable[featurestable["Name"].isin(subjects_train)]
features_val = featurestable[featurestable["Name"].isin(subjects_val)]
features_test = featurestable[featurestable["Name"].isin(subjects_test)]

# -------------------------------------------------------
# Datasets & loaders
# -------------------------------------------------------

# 3-class: labels unchanged
label_transform_3class = None

train_dataset = CustomDataset(
    subjects=subjects_train,
    features_dataframe=features_train,
    transform=composer,
    label_transform=label_transform_3class,
)
val_dataset = CustomDataset(
    subjects=subjects_val,
    features_dataframe=features_val,
    transform=composer,
    label_transform=label_transform_3class,
)
test_dataset = CustomDataset(
    subjects=subjects_test,
    features_dataframe=features_test,
    transform=composer,
    label_transform=label_transform_3class,
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

train_dl = DeviceDataLoader(train_loader, device)
val_dl = DeviceDataLoader(val_loader, device)
test_dl = DeviceDataLoader(test_loader, device)

# -------------------------------------------------------
# Model definition (same as notebook)
# -------------------------------------------------------

cnn_out_dim = 8
cnn = ModifiedResNet18(num_classes=cnn_out_dim)

num_additional_features = len(
    list(featurestable.columns)[
        list(featurestable.columns).index("Name") + 1 : list(featurestable.columns).index(
            "Class"
        )
    ]
)
mlp_num_neurons = [8, 3]
mlp = MLPModule(
    cnn_out_dim + num_additional_features, len(mlp_num_neurons), mlp_num_neurons
)

combined_model = CombinedModel(cnn, mlp).to(device)


def reinit_weights(model, seed):
    """
    Reinitialize weights as in the original 3-class notebook:
    - last linear layer of CNN with uniform random
    - all parameters of MLP with uniform random
    """
    np.random.seed(seed)
    layers = list(model.cnn.children())
    last_n_layers = layers[-1:]
    for layer in last_n_layers:
        if isinstance(layer, torch.nn.Linear):
            layer.weight.data = torch.FloatTensor(
                np.random.rand(*layer.weight.size())
            )
    for param in model.mlp.parameters():
        param.data = torch.from_numpy(np.random.rand(*param.size()))
    model = model.float()
    torch.save(model.state_dict(), "init_model.pth")
    return model


model = reinit_weights(combined_model, seed)


# -------------------------------------------------------
# Training function (cross-entropy, same logic)
# -------------------------------------------------------

def my_train(model, optimizer, loss_fn, train_loader, val_loader, epochs=30, to_print=True):
    min_val_loss = float("inf")
    epochs_no_improve = 0
    history = []

    try:
        model.load_state_dict(torch.load("init_model.pth"))
    except Exception:
        model = reinit_weights(model, seed)
        model = model.to(device)

    for epoch in range(1, epochs + 1):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            images = batch["image"]
            additional_features = batch["additional_features"]
            labels = batch["labels"].long()
            outputs = model(images, additional_features)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.item() * images.size(0)
        training_loss /= len(subjects_train)

        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for batch in val_loader:
                images = batch["image"]
                additional_features = batch["additional_features"]
                labels = batch["labels"].long()
                outputs = model(images, additional_features)
                loss = loss_fn(outputs, labels)
                valid_loss += loss.item() * images.size(0)
                predicted = torch.argmax(F.softmax(outputs, dim=1), dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            valid_loss /= len(subjects_val)
            accy = correct / total

            if to_print:
                print(
                    f"Epoch: {epoch}, Training Loss: {training_loss:.4f}, "
                    f"Validation Loss: {valid_loss:.4f}, Accuracy: {accy:.4f}"
                )

            history.append(
                {"loss": training_loss, "val_loss": valid_loss, "val_acc": accy}
            )

            if valid_loss < min_val_loss:
                min_val_loss = valid_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), "best_model.pth")
            else:
                epochs_no_improve += 1

            if (epoch > 50) and (epochs_no_improve > 20):
                print(f"Early stopping after {epoch+1} epochs")
                break

    return history, min_val_loss


# -------------------------------------------------------
# Class weighting for CrossEntropy
# -------------------------------------------------------

num_positive_classes = [0, 0, 0]
total_examples_classes = [0, 0, 0]

for batch in train_loader:
    labels = batch["labels"]
    for i in range(len(num_positive_classes)):
        num_positive_classes[i] += (labels == i).sum().item()
        total_examples_classes[i] += labels.size(0)

pos_weights = [
    total_examples_classes[i] / num_positive_classes[i]
    for i in range(len(num_positive_classes))
]
loss_weights = torch.tensor(pos_weights).to(device)
loss_fn = nn.CrossEntropyLoss(weight=loss_weights)

optimizer = torch.optim.Adam(combined_model.parameters(), lr=0.01)
epochs = 100
# history, min_val_loss = my_train(combined_model, optimizer, loss_fn, train_dl, val_dl, epochs)


# -------------------------------------------------------
# Cross-validation for hyperparameter tuning (same logic)
# -------------------------------------------------------

inner_splits = 5
cm = np.zeros((3, 3))
best_val_loss = 1e10
torch.save(combined_model.state_dict(), "init_model.pth")

hyperparameters = {
    "batch_size": [4, 8, 16],
    "lr": [0.01, 0.001],
    "num_neurons": [[8, 3], [32, 3], [32, 8, 3], [8, 32, 3]],
    "cnn_outdim": [1, num_additional_features],
}
max_attempts = 50

all_hyperparams = list(__import__("itertools").product(*hyperparameters.values()))
random.shuffle(all_hyperparams)

# define y for CV (subject-level labels)
y = []
for subj in subjects_train:
    mask = featurestable["Name"].str.contains(subj[:6])
    y.append(featurestable.loc[mask, "Class"].iloc[0])

inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=42)

best_models = []
best_hyp = None

for i in range(0, min(len(all_hyperparams), max_attempts)):
    params = all_hyperparams[i]
    print(params)
    selected_hyperparams = dict(zip(hyperparameters.keys(), params))

    set_val_loss = 0
    models = []

    for inner_train_indices, inner_val_indices in inner_cv.split(subjects_train, y):
        subjects_train_inner = list(
            np.array(subjects_train)[np.array(inner_train_indices)]
        )
        subjects_val_inner = list(
            np.array(subjects_train)[np.array(inner_val_indices)]
        )

        pattern = "|".join(subjects_train_inner)
        matching_rows = featurestable[featurestable["Name"].str.contains(pattern)]
        subjects_train_inner = matching_rows["Name"].tolist()

        pattern = "|".join(subjects_val_inner)
        matching_rows = featurestable[featurestable["Name"].str.contains(pattern)]
        subjects_val_inner = matching_rows["Name"].tolist()

        features_train_inner = featurestable[
            featurestable["Name"].isin(subjects_train_inner)
        ]
        features_val_inner = featurestable[
            featurestable["Name"].isin(subjects_val_inner)
        ]

        train_loader_inner = DataLoader(
            CustomDataset(
                subjects=subjects_train_inner,
                features_dataframe=features_train_inner,
                transform=composer,
                label_transform=None,
            ),
            batch_size=selected_hyperparams["batch_size"],
            shuffle=True,
        )
        val_loader_inner = DataLoader(
            CustomDataset(
                subjects=subjects_val_inner,
                features_dataframe=features_val_inner,
                transform=composer,
                label_transform=None,
            ),
            batch_size=selected_hyperparams["batch_size"],
            shuffle=True,
        )

        train_dl_inner = DeviceDataLoader(train_loader_inner, device)
        val_dl_inner = DeviceDataLoader(val_loader_inner, device)

        # class weights for this inner split
        num_positive_classes = [0, 0, 0]
        total_examples_classes = [0, 0, 0]
        for batch in train_loader_inner:
            labels = batch["labels"]
            for k in range(len(num_positive_classes)):
                num_positive_classes[k] += (labels == k).sum().item()
                total_examples_classes[k] += labels.size(0)
        pos_weights = [
            total_examples_classes[k] / num_positive_classes[k]
            for k in range(len(num_positive_classes))
        ]
        loss_weights_inner = torch.tensor(pos_weights).to(device)
        loss_fn_inner = nn.CrossEntropyLoss(weight=loss_weights_inner)

        cnn_inner = ModifiedResNet18(num_classes=selected_hyperparams["cnn_outdim"])
        mlp_num_neurons_inner = selected_hyperparams["num_neurons"]
        mlp_inner = MLPModule(
            selected_hyperparams["cnn_outdim"] + num_additional_features,
            len(mlp_num_neurons_inner),
            mlp_num_neurons_inner,
        )
        combined_model_inner = CombinedModel(cnn_inner, mlp_inner).to(device)

        optimizer_inner = torch.optim.Adam(
            combined_model_inner.parameters(), lr=selected_hyperparams["lr"]
        )
        epochs = 1000

        history, min_val_loss = my_train(
            combined_model_inner, optimizer_inner, loss_fn_inner, train_dl_inner, val_dl_inner, epochs
        )
        set_val_loss += min_val_loss
        if set_val_loss > best_val_loss:
            break
        models.append(torch.load("best_model.pth"))

    if set_val_loss < best_val_loss:
        best_val_loss = set_val_loss
        best_hyp = selected_hyperparams
        best_models = models

    print(set_val_loss, selected_hyperparams)

print(best_val_loss, best_hyp)

for i in range(0, inner_splits):
    torch.save(best_models[i], f"best_cv_model_{i}.pth")


# -------------------------------------------------------
# Final evaluation (same as notebook)
# -------------------------------------------------------

cnn_final = ModifiedResNet18(num_classes=best_hyp["cnn_outdim"])
mlp_num_neurons_final = best_hyp["num_neurons"]
mlp_final = MLPModule(
    best_hyp["cnn_outdim"] + num_additional_features,
    len(mlp_num_neurons_final),
    mlp_num_neurons_final,
)
combined_model_final = CombinedModel(cnn_final, mlp_final).to(device)
combined_model_final.eval()

encoder = LabelBinarizer()
classes = ["Normal", "Monolateral", "Bilateral"]

for i in range(0, 5):
    print(f"\n\n\n FOLD_{i+1}\n")
    combined_model_final.load_state_dict(best_models[i])

    all_preds = []
    all_labels = []
    all_probs = []
    all_features = []

    for batch in test_dl:
        images = batch["image"]
        additional_features = batch["additional_features"]
        labels = batch["labels"].float().detach().cpu().numpy()
        outputs = combined_model_final(images, additional_features)
        probs = F.softmax(outputs, dim=1).detach().cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_preds.extend(preds)
        all_labels.extend(labels)
        all_probs.extend(probs)
        all_features.extend(additional_features.detach().cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_features = np.array(all_features)

    print("\nTrial Confusion Matrix\n")
    cm = confusion_matrix(all_labels, all_preds)
    tsc_confusion_matrix1 = np.transpose(
        np.transpose(cm) / np.sum(cm, axis=1)
    )
    seaborn.heatmap(tsc_confusion_matrix1, annot=True, vmin=0, vmax=1, cmap="Blues")
    plt.xticks(np.arange(len(classes)) + 0.5, classes)
    plt.yticks(np.arange(len(classes)) + 0.5, classes)
    plt.savefig(f"Trials Confusion Matrix {i+1}")
    plt.show()

    for j in range(0, 3):
        display = RocCurveDisplay.from_predictions(
            encoder.fit_transform(all_labels)[:, j],
            all_probs[:, j],
            name=f"{classes[j]} vs the rest",
            color="darkorange",
        )
        plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
        _ = display.ax_.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title="One-vs-Rest ROC curves: " + classes[j],
        )
        plt.savefig(f"Trial ROC-AUC {classes[j]} {i+1}")
        plt.show()

    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(
        encoder.fit_transform(all_labels), all_probs, average="macro", multi_class="ovr"
    )
    print("Results:")
    print(precision, recall, f1, balanced_accuracy, auc)

    # patient-level aggregation (unchanged logic)
    test_pat = []
    for j in range(len(all_features)):
        match = featurestable[
            np.isclose(featurestable["Maximum Angle"], all_features[j, 0], atol=1e-5)
        ]["Name"].values[0]
        test_pat.append(match)

    test_pat_unique = list(set([name[:6] for name in test_pat]))

    y_pred_pat = []
    y_true_pat = []
    y_prob_pat = []

    for j in test_pat_unique:
        idx = [k for k, name in enumerate(test_pat) if j in name]
        label = list(set(all_labels[idx]))
        out = all_probs[idx]
        out_softmax = np.array(
            [F.softmax(torch.tensor(logit), dim=0).numpy() for logit in out]
        )
        summed = np.sum(out_softmax, axis=0)
        y_pred_pat.append(np.argmax(summed))
        y_prob_pat.append(summed / np.sum(summed))
        y_true_pat.append(int(label[0]))

    print("\nPatients Confusion Matrix\n")
    cm = confusion_matrix(y_true_pat, y_pred_pat)
    tsc_confusion_matrix1 = np.transpose(
        np.transpose(cm) / np.sum(cm, axis=1)
    )
    seaborn.heatmap(tsc_confusion_matrix1, annot=True, vmin=0, vmax=1, cmap="Blues")
    plt.xticks(np.arange(len(classes)) + 0.5, classes)
    plt.yticks(np.arange(len(classes)) + 0.5, classes)
    plt.savefig(f"Patients Confusion Matrix {i+1}")
    plt.show()

    all_labels_pat = np.array(y_true_pat)
    all_preds_pat = np.array(y_pred_pat)
    all_probs_pat = np.array(y_prob_pat)

    precision = precision_score(all_labels_pat, all_preds_pat, average="macro")
    recall = recall_score(all_labels_pat, all_preds_pat, average="macro")
    f1 = f1_score(all_labels_pat, all_preds_pat, average="macro")
    balanced_accuracy = balanced_accuracy_score(all_labels_pat, all_preds_pat)
    auc = roc_auc_score(
        encoder.fit_transform(all_labels_pat),
        all_probs_pat,
        average="macro",
        multi_class="ovr",
    )
    print("Results:")
    print(precision, recall, f1, balanced_accuracy, auc)
