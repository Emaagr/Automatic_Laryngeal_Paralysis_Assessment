# ablation_3class.py
"""
Ablation study script for 3-class classification (Normal / Monolateral / Bilateral).

This reproduces the logic of Ablation_3_Classes.ipynb:
- 3-class classification
- cross-validated hyperparameter search over:
    * image-only CNN
    * image + subsets of additional features
    * additional features only
- coalitions of feature groups (players 1,2,3,4)

Common components (dataset, models, transforms, device, seeds) are imported
from models_common.py. Only ablation-specific functions are defined here.
"""

import os
import random
import itertools

import matplotlib.pyplot as plt  # (non strettamente necessario qui, ma lo lascio per coerenza)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader

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


# --------------------------------------------------
# Ablation-specific utils (3-class version)
# --------------------------------------------------

def he_init(shape):
    """He initialization, as in the original notebook."""
    return np.random.randn(*shape) * np.sqrt(2 / shape[0])


def reinit_weights(model, seed):
    """
    Reinitialize weights of the last CNN layer and MLP, with try/except
    to handle models that may not have cnn or mlp (as in the ablation notebook).
    """
    np.random.seed(seed)

    # Reinitialize last linear layer of CNN if present
    try:
        layers = list(model.cnn.children())
        last_n_layers = layers[-1:]
        for layer in last_n_layers:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data = torch.FloatTensor(he_init(layer.weight.size()))
    except Exception:
        pass

    # Reinitialize all MLP params if present
    try:
        for param in model.mlp.parameters():
            param.data = torch.from_numpy(he_init(param.size()))
    except Exception:
        pass

    model = model.float()
    torch.save(model.state_dict(), "init_model.pth")
    return model


def select_additional_indices(combs):
    """
    Map combinations of players (2,3,4) to column indices of the additional features.
    Kept exactly as in the notebook.

    Players:
      2 -> indices [0, 1]
      3 -> indices [2, 3, 5]
      4 -> indices [4, 6, 7]
    """
    additional_ok = []
    if 2 in combs:
        additional_ok.extend([0, 1])
    if 3 in combs:
        additional_ok.extend([2, 3, 5])
    if 4 in combs:
        additional_ok.extend([4, 6, 7])

    return np.array(additional_ok)


def my_train(
    model,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    ind_comb,
    ordered_combinations,
    epochs=30,
    to_print=False,
):
    """
    Training loop for ablation (3 classes), with:
    - different forward passes depending on ind_comb (image only / image+features / features only),
    - early stopping on validation loss,
    - reinit from 'init_model.pth' if available.
    Logic preserved from Ablation_3_Classes.ipynb.
    """
    min_val_loss = float("inf")
    epochs_no_improve = 0
    history = []

    try:
        model.load_state_dict(torch.load("init_model.pth"))
    except Exception:
        model = reinit_weights(model, seed)
        model = model.to(device)

    add_ind_ok = select_additional_indices(ordered_combinations[ind_comb])

    for epoch in range(1, epochs + 1):
        training_loss = 0.0
        valid_loss = 0.0
        total_train_samples = 0
        model.train()
        for batch in train_loader:
            images = batch["image"]
            additional_features = batch["additional_features"][:, add_ind_ok]
            labels = batch["labels"].long()

            if ind_comb == 0:
                # image only
                outputs = model(images)
            elif 0 < ind_comb < 7:
                # image + selected additional features
                outputs = model(images, additional_features)
            elif ind_comb >= 7:
                # additional features only
                outputs = model(additional_features)

            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * images.size(0)
            total_train_samples += images.size(0)
        training_loss /= total_train_samples

        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            total_val_samples = 0
            for batch in val_loader:
                images = batch["image"]
                additional_features = batch["additional_features"][:, add_ind_ok]
                labels = batch["labels"].long()

                if ind_comb == 0:
                    outputs = model(images)
                elif 0 < ind_comb < 7:
                    outputs = model(images, additional_features)
                elif ind_comb >= 7:
                    outputs = model(additional_features)

                loss = loss_fn(outputs, labels)
                valid_loss += loss.data.item() * images.size(0)
                total_val_samples += images.size(0)
                predicted = torch.argmax(F.softmax(outputs, dim=1), dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            valid_loss /= total_val_samples
            accy = correct / total

            if to_print:
                print(
                    "Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, accuracy = {:.4f}".format(
                        epoch, training_loss, valid_loss, accy
                    )
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
                print(f"Early stopping after {epoch} epochs")
                break

    return history, min_val_loss


# --------------------------------------------------
# Main script
# --------------------------------------------------

if __name__ == "__main__":
    # ---------------- Dataset & basic split (as in notebook) ----------------
    data_dir = (
        "/content/drive/MyDrive/AI in healthcare/Laryngeal paralysis/"
        "Spain Dataset/Trials"
    )
    datasetpath = os.path.join(data_dir, "Difference Images")

    featurestable = pd.read_excel(
        "/content/drive/MyDrive/AI in healthcare/Laryngeal paralysis/"
        "Spain Dataset/Trials/Agati features.xlsx"
    )
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

    target = []
    for subj in spa_subj:
        mask = featurestable["Name"].str.contains(subj)
        target.append(featurestable.loc[mask, "Class"].iloc[0])

    spa_subj = np.sort(spa_subj)
    subjects_train, subjects_test = train_test_split(
        spa_subj, test_size=0.2, random_state=42, stratify=target
    )

    y = []
    for subj in subjects_train:
        mask = featurestable["Name"].str.contains(subj)
        y.append(featurestable.loc[mask, "Class"].iloc[0])
    y = np.array(y)

    # ---------------- Coalitions of feature groups {1,2,3,4} ----------------
    values = [1, 2, 3, 4]
    combinations = []
    for r in range(1, 5):
        combinations.extend(list(itertools.combinations(values, r)))

    combinations_with_1 = [comb for comb in combinations if 1 in comb]
    combinations_without_1 = [comb for comb in combinations if 1 not in comb]

    # ordered_combinations è la stessa struttura del notebook
    ordered_combinations = combinations_with_1 + combinations_without_1
    del ordered_combinations[7]

    # ---------------- Inner CV & ablation training ----------------
    inner_splits = 5
    best_models_list = []
    best_hyp_list = []
    best_loss_list = []
    inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=42)

    for ind_comb in range(0, len(ordered_combinations)):
        num_additional_features = len(
            select_additional_indices(ordered_combinations[ind_comb])
        )

        best_val_loss = float("inf")

        if ind_comb == 0:
            # Image only: ResNet18 -> 3 logits
            hyperparameters = {
                "batch_size": [8, 16],
                "lr": [0.01, 0.001, 0.0001],
            }
            cnn = ModifiedResNet18(num_classes=3)
            combined_model = cnn.to(device)

        elif 0 < ind_comb < 7:
            # Image + additional features
            hyperparameters = {
                "batch_size": [8, 16],
                "lr": [0.01, 0.001],
                "num_neurons": [[8, 3], [32, 3], [32, 8, 3], [8, 32, 3]],
                "cnn_outdim": [1, num_additional_features],
            }

        elif ind_comb >= 7:
            # Additional features only
            hyperparameters = {
                "batch_size": [8, 16],
                "lr": [0.1, 0.01, 0.001],
                "num_neurons": [[16, 64, 3], [64, 16, 3], [32, 3], [64, 32, 16, 3]],
            }

        all_hyperparams = list(itertools.product(*hyperparameters.values()))
        random.shuffle(all_hyperparams)

        for i_hyp in range(0, len(all_hyperparams)):
            params = all_hyperparams[i_hyp]
            print(params)
            selected_hyperparams = dict(zip(hyperparameters.keys(), params))

            set_val_loss = 0.0
            models = []

            # Iterate over the shuffled randomized combinations of hyperparameters
            for inner_train_indices, inner_val_indices in inner_cv.split(
                subjects_train, y
            ):
                # Subjects for this inner split
                subjects_train_inner = list(
                    np.array(subjects_train)[np.array(inner_train_indices)]
                )
                subjects_val_inner = list(
                    np.array(subjects_train)[np.array(inner_val_indices)]
                )

                pattern = "|".join(subjects_train_inner)
                matching_rows = featurestable[
                    featurestable["Name"].str.contains(pattern)
                ]
                subjects_train_inner = matching_rows["Name"].tolist()

                pattern = "|".join(subjects_val_inner)
                matching_rows = featurestable[
                    featurestable["Name"].str.contains(pattern)
                ]
                subjects_val_inner = matching_rows["Name"].tolist()

                features_train = featurestable[
                    featurestable["Name"].isin(subjects_train_inner)
                ]
                features_val = featurestable[
                    featurestable["Name"].isin(subjects_val_inner)
                ]

                # 3-class dataset: labels are 0/1/2, no binarization
                train_loader = DataLoader(
                    CustomDataset(
                        subjects=subjects_train_inner,
                        features_dataframe=features_train,
                        transform=composer,
                    ),
                    batch_size=selected_hyperparams["batch_size"],
                    shuffle=True,
                )
                val_loader = DataLoader(
                    CustomDataset(
                        subjects=subjects_val_inner,
                        features_dataframe=features_val,
                        transform=composer,
                    ),
                    batch_size=selected_hyperparams["batch_size"],
                    shuffle=True,
                )

                train_dl = DeviceDataLoader(train_loader, device)
                val_dl = DeviceDataLoader(val_loader, device)

                # Class weights for CrossEntropyLoss (like in notebook)
                num_positive_classes = [0, 0, 0]
                total_examples_classes = [0, 0, 0]

                for batch in train_loader:
                    labels = batch["labels"]
                    for ci in range(len(num_positive_classes)):
                        num_positive_classes[ci] += (labels == ci).sum().item()
                        total_examples_classes[ci] += labels.size(0)

                pos_weights = torch.tensor(
                    [
                        total_examples_classes[ci] / max(num_positive_classes[ci], 1)
                        for ci in range(len(num_positive_classes))
                    ]
                )
                loss_fn = nn.CrossEntropyLoss(weight=pos_weights).to(device)

                # Build appropriate model for this coalition & hyperparams
                if 0 < ind_comb < 7:
                    cnn = ModifiedResNet18(
                        num_classes=selected_hyperparams["cnn_outdim"]
                    )
                    mlp_num_neurons = selected_hyperparams["num_neurons"]
                    mlp = MLPModule(
                        selected_hyperparams["cnn_outdim"] + num_additional_features,
                        len(mlp_num_neurons),
                        mlp_num_neurons,
                    )
                    combined_model = CombinedModel(cnn, mlp).to(device)
                elif ind_comb >= 7:
                    mlp_num_neurons = selected_hyperparams["num_neurons"]
                    mlp = MLPModule(
                        num_additional_features,
                        len(mlp_num_neurons),
                        mlp_num_neurons,
                    )
                    combined_model = mlp.to(device)

                optimizer = torch.optim.Adam(
                    combined_model.parameters(),
                    lr=selected_hyperparams["lr"],
                )
                epochs = 1000

                history, min_val_loss = my_train(
                    combined_model,
                    optimizer,
                    loss_fn,
                    train_dl,
                    val_dl,
                    ind_comb,
                    ordered_combinations,
                    epochs,
                )
                set_val_loss += min_val_loss
                if set_val_loss > best_val_loss:
                    break
                models.append(torch.load("best_model.pth"))

            if set_val_loss < best_val_loss:
                best_val_loss = set_val_loss
                best_hyp = selected_hyperparams
                best_models = models

        best_models_list.append(best_hyp)
        best_hyp_list.append(best_models)
        best_loss_list.append(best_val_loss)

    # A questo punto hai:
    # - best_models_list: iperparametri migliori per ogni combinazione
    # - best_hyp_list: lista dei modelli migliori per ogni split
    # - best_loss_list: best_val_loss per ciascuna combinazione
    # Puoi salvarli su disco come preferisci, o caricarli in un altro script/notebook.
