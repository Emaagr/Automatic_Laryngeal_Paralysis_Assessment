# src/ablation/ablation_utils.py

import torch
import numpy as np
import math
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import torch.nn as nn
import itertools
from sklearn.metrics import confusion_matrix
import seaborn
import matplotlib.pyplot as plt

# ---------------- Ablation-specific utils ----------------

def he_init(shape):
    """He initialization."""
    return nn.init.kaiming_normal_(torch.empty(*shape), mode='fan_in', nonlinearity='relu')

def reinit_weights(model, seed, device):
    """Reinitialize weights of the last CNN layer and MLP."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Reinizializza l'ultimo layer CNN
    try:
        layers = list(model.cnn.children())
        last_layer = layers[-1]
        if isinstance(last_layer, torch.nn.Linear):
            nn.init.kaiming_normal_(last_layer.weight, mode='fan_in', nonlinearity='relu')
            if last_layer.bias is not None:
                nn.init.zeros_(last_layer.bias)
    except Exception as e:
        print(f"Error reinitializing CNN layer: {e}")

    for param in model.mlp.parameters():
        nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')

    model = model.float().to(device)
    torch.save(model.state_dict(), "init_model.pth")
    return model

def select_additional_indices(combs):
    """Map combinations of players (2,3,4) to feature indices."""
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
    device=None,
):
    """Training loop for ablation, handling different forward passes."""
    min_val_loss = float("inf")
    epochs_no_improve = 0
    history = []

    try:
        model.load_state_dict(torch.load("init_model.pth"))
    except:
        model = reinit_weights(model, seed, device)
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
                outputs = model(images)  
            elif 0 < ind_comb < 7:
                outputs = model(images, additional_features)  
            elif ind_comb >= 7:
                outputs = model(additional_features) 

            outputs = outputs.view(-1)
            loss = loss_fn(torch.reshape(outputs, (-1, 1)), torch.reshape(labels, (-1, 1)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * images.size(0)
            total_train_samples += images.size(0)

        training_loss /= max(total_train_samples, 1)

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
                loss = loss_fn(torch.reshape(outputs, (-1, 1)), torch.reshape(labels, (-1, 1)))
                valid_loss += loss.data.item() * images.size(0)
                total_val_samples += images.size(0)

                # Predizioni (caso binario o multiclass)
                if outputs.size(1) == 1:
                    preds = (torch.sigmoid(outputs) > 0.5).long()  # binario
                else:
                    preds = outputs.argmax(dim=1)  # multiclass

                correct += (preds == labels).sum().item()
                total += labels.size(0)

            valid_loss /= max(total_val_samples, 1)
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
    """Compute Shapley values for feature groups."""
    players = list(set().union(*coalitions))
    n = len(players)
    shapley_values = {player: 0 for player in players}

    for player in players:
        for coalition in coalitions:
            if player not in coalition:
                ind_without_player = coalitions.index(coalition)
                ind_with_player = ordered_combinations.index(
                    tuple(set().union(coalition, tuple([player]))))
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


