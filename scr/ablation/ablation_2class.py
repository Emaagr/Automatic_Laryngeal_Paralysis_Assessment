# src/ablation/ablation_2class.py

import os
import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from training_utils import CustomDataset, DeviceDataLoader, ModifiedResNet18, MLPModule, CombinedModel
from ablation_utils import my_train, shapley_value, reinit_weights, select_additional_indices, he_init

# Ablation-specific configurations
# (e.g. dataset paths, hyperparameters, etc.)

def ablation_2class():
    # Step 1: Load dataset and split (e.g. Normal vs Paralyzed)
    data_dir = "/path/to/data"
    featurestable = pd.read_excel(os.path.join(data_dir, 'featurestable.xlsx'))

    # Prepare dataset
    # (Assuming similar processing as in the ablation script)

    # Define hyperparameters
    hyperparameters = {
        "batch_size": [8, 16],
        "lr": [0.01, 0.001],
        "num_neurons": [[8, 1], [32, 1]],
    }

    # Step 2: Train models and perform ablation for binary classification
    for ind_comb in range(0, len(ordered_combinations)):
        num_additional_features = len(select_additional_indices(ordered_combinations[ind_comb]))

        # Initialize model, loss, optimizer, etc.
        cnn = ModifiedResNet18(num_classes=1)
        mlp = MLPModule(cnn_outdim=1, num_neurons=[8, 1])
        combined_model = CombinedModel(cnn, mlp).to(device)

        optimizer = torch.optim.Adam(combined_model.parameters(), lr=0.001)
        loss_fn = nn.BCEWithLogitsLoss()

        # Step 3: Perform training
        history, min_val_loss = my_train(combined_model, optimizer, loss_fn, train_dl, val_dl, ind_comb, ordered_combinations)
        
        # Step 4: Shapley value calculation (optional)
        shapley_vals = shapley_value(coalitions, values)

        # Save results
        torch.save(combined_model.state_dict(), 'best_model.pth')

# Execute ablation study
ablation_2class()
