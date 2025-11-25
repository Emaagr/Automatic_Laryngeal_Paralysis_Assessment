"""
Ablation study script for binary (2-class) classification.

This reproduces the logic of Ablation.ipynb:
- binary classification (Normal vs Paralyzed)
- cross-validated hyperparameter search over:
    * image-only CNN
    * image + subsets of additional features
    * additional features only
- coalitions of feature groups (players 1,2,3,4)
- Shapley values from precomputed CSV results (binary + multiclass)

Common components (dataset, models, transforms, device, seeds) are imported
from models_common.py. Only ablation-specific functions are defined here.
"""

import os
import math
import random
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, KFold

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
# Ablation-specific utils
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
    Training loop for ablation, with:
    - different forward passes depending on ind_comb (image only / image+features / features only),
    - early stopping on validation loss,
    - reinit from 'init_model.pth' if available.
    Logic preserved from Ablation.ipynb.
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
            labels = batch["labels"].float()

            if ind_comb == 0:
                # image only
                outputs = model(images)
            elif 0 < ind_comb < 7:
                # image + selected additional features
                outputs = model(images, additional_features)
            elif ind_comb >= 7:
                # additional features only
                outputs = model(additional_features)

            outputs = outputs.view(-1)
            loss = loss_fn(
                torch.reshape(outputs, (-1, 1)),
                torch.reshape(labels, (-1, 1)),
            )
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
                labels = batch["labels"].float()

                if ind_comb == 0:
                    outputs = model(images)
                elif 0 < ind_comb < 7:
                    outputs = model(images, additional_features)
                elif ind_comb >= 7:
                    outputs = model(additional_features)

                outputs = outputs.view(-1)
                loss = loss_fn(
                    torch.reshape(outputs, (-1, 1)),
                    torch.reshape(labels, (-1, 1)),
                )
                valid_loss += loss.data.item() * images.size(0)
                total_val_samples += images.size(0)
                predicted = np.resize((outputs.cpu().numpy() > 0) * 1, labels.size(0))
                total += labels.size(0)
                correct += (predicted == labels.cpu().numpy()).sum().item()
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
                print(f"Early stopping after {epoch+1} epochs")
                break

    return history, min_val_loss


def shapley_value(coalitions, values):
    """
    Shapley value computation, using global ordered_combinations
    (kept like the notebook, i.e. relying on the outer variable).
    """
    players = list(set().union(*coalitions))
    n = len(players)
    shapley_values = {player: 0 for player in players}

    for player in players:
        for coalition in coalitions:
            if player not in coalition:
                ind_without_player = coalitions.index(coalition)
                ind_with_player = ordered_combinations.index(
                    tuple(set().union(coalition, tuple([player])))
                )

                coalition_value_without_player = values[ind_without_player]
                coalition_value_with_player = values[ind_with_player]

                marginal_contribution = (
                    coalition_value_with_player - coalition_value_without_player
                )
                weight = (
                    math.factorial(len(coalition))
                    * math.factorial(n - len(coalition) - 1)
                    / math.factorial(n)
                )
                shapley_values[player] += weight * marginal_contribution

    return shapley_values


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
    target = (np.array(target) > 0) * 1

    spa_subj = np.sort(spa_subj)
    subjects_train, subjects_test = train_test_split(
        spa_subj, test_size=0.2, random_state=42, stratify=target
    )

    y = []
    for subj in subjects_train:
        mask = featurestable["Name"].str.contains(subj)
        y.append(featurestable.loc[mask, "Class"].iloc[0])
    y = (np.array(y) > 0) * 1

    # ---------------- Coalitions of feature groups {1,2,3,4} ----------------
    values = [1, 2, 3, 4]
    combinations = []
    for r in range(1, 5):
        combinations.extend(list(itertools.combinations(values, r)))

    combinations_with_1 = [comb for comb in combinations if 1 in comb]
    combinations_without_1 = [comb for comb in combinations if 1 not in comb]

    # ordered_combinations is global and used inside shapley_value
    ordered_combinations = combinations_with_1 + combinations_without_1
    # As in notebook: delete index 7
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

        # Define hyperparameters and base model depending on coalition index
        if ind_comb == 0:
            # Image only
            hyperparameters = {
                "batch_size": [8, 16],
                "lr": [0.01, 0.001, 0.0001],
            }
            cnn = ModifiedResNet18(num_classes=1)
            combined_model = cnn.to(device)

        elif 0 < ind_comb < 7:
            # Image + additional features
            hyperparameters = {
                "batch_size": [8, 16],
                "lr": [0.01, 0.001],
                "num_neurons": [[8, 1], [32, 1], [32, 8, 1], [8, 32, 1]],
                "cnn_outdim": [1, num_additional_features],
            }

        elif ind_comb >= 7:
            # Additional features only
            hyperparameters = {
                "batch_size": [8, 16],
                "lr": [0.1, 0.01, 0.001],
                "num_neurons": [[16, 64, 1], [64, 16, 1], [32, 1], [64, 32, 16, 1]],
            }

        all_hyperparams = list(itertools.product(*hyperparameters.values()))
        random.shuffle(all_hyperparams)

        for i in range(0, len(all_hyperparams)):
            params = all_hyperparams[i]
            print(params)
            selected_hyperparams = dict(zip(hyperparameters.keys(), params))

            set_val_loss = 0
            models = []

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

                train_loader = DataLoader(
                    CustomDataset(
                        subjects=subjects_train_inner,
                        features_dataframe=features_train,
                        transform=composer,
                        label_transform=lambda y: int(np.array(y) > 0),
                    ),
                    batch_size=selected_hyperparams["batch_size"],
                    shuffle=True,
                )
                val_loader = DataLoader(
                    CustomDataset(
                        subjects=subjects_val_inner,
                        features_dataframe=features_val,
                        transform=composer,
                        label_transform=lambda y: int(np.array(y) > 0),
                    ),
                    batch_size=selected_hyperparams["batch_size"],
                    shuffle=True,
                )

                train_dl = DeviceDataLoader(train_loader, device)
                val_dl = DeviceDataLoader(val_loader, device)

                # Class weights for BCEWithLogitsLoss
                num_positive, num_negative = 0, 0
                for batch in train_loader:
                    labels = batch["labels"].float()
                    num_positive += (labels == 1).sum().item()
                    num_negative += (labels == 0).sum().item()

                pos_weight = torch.tensor([num_negative / num_positive])
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)

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

    # --------------------------------------------------
    # Players' contributions (Shapley values)
    # --------------------------------------------------
    bin_df = pd.read_csv(
        "/content/drive/MyDrive/AI in healthcare/Laryngeal paralysis/"
        "Spain Dataset/Risultati/Binary_oct_09_24/ablation_oct_13_24/results/out.csv"
    )
    mc_df = pd.read_csv(
        "/content/drive/MyDrive/AI in healthcare/Laryngeal paralysis/"
        "Spain Dataset/Risultati/Multiclass_20 07 2024/Ablation/excel con i risultati/"
        "model_results.csv"
    )

    ordered_combinations.append((1, 2, 3, 4))
    ordered_combinations.insert(0, ())

    # ---- Binary ----
    bin_val = [0.5]
    for i in range(0, int(len(bin_df) / 5)):
        bin_val.append(
            np.median(
                bin_df["balanced_accuracy_trials"][i * 5 : i * 5 + 5].values
            )
        )
    bin_val.append(0.7655)

    bin_shapley_vals = shapley_value(ordered_combinations, bin_val)

    players = list(bin_shapley_vals.keys())
    contributions = list(bin_shapley_vals.values())

    norm = plt.Normalize(0, max(contributions))
    colors = plt.cm.Blues(norm(contributions))

    seaborn.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    bars = plt.bar(players, contributions, color=colors, edgecolor="black")
    plt.xlabel("Player", fontsize=20)
    plt.ylabel("Shapley Value", fontsize=20)
    plt.title("Binary classification", fontsize=20)
    plt.xticks(players, fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(False)
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.002,
            f"{yval:.3f}",
            ha="center",
            va="bottom",
            fontsize=16,
        )
    plt.ylim(0, max(contributions) + 0.01)
    seaborn.despine()
    plt.tight_layout()
    plt.savefig("shapley_values_bin.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ---- Multiclass ----
    mc_val = [0.33]
    for i in range(0, int(len(bin_df) / 5)):
        mc_val.append(
            np.median(
                mc_df["balanced_accuracy_trials"][i * 5 : i * 5 + 5].values
            )
        )
    mc_val.append(0.76)

    mc_shapley_vals = shapley_value(ordered_combinations, mc_val)

    players = list(mc_shapley_vals.keys())
    contributions = list(mc_shapley_vals.values())

    norm = plt.Normalize(0, max(contributions))
    colors = plt.cm.Blues(norm(contributions))

    seaborn.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    bars = plt.bar(players, contributions, color=colors, edgecolor="black")
    plt.xlabel("Player", fontsize=20)
    plt.ylabel("Shapley Value", fontsize=20)
    plt.title("Multiclass Classification", fontsize=20)
    plt.xticks(players, fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(False)
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.002,
            f"{yval:.3f}",
            ha="center",
            va="bottom",
            fontsize=16,
        )
    plt.ylim(0, max(contributions) + 0.01)
    seaborn.despine()
    plt.tight_layout()
    plt.savefig("shapley_values_mc.png", dpi=300, bbox_inches="tight")
    plt.show()
