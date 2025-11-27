# src/ablation/ablation_3class.py

import os
import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from training_utils import CustomDataset, DeviceDataLoader, ModifiedResNet18, MLPModule, CombinedModel
from ablation_utils import my_train, shapley_value, reinit_weights, select_additional_indices

# Ablation-specific configurations
# (e.g. dataset paths, hyperparameters, etc.)

def ablation_3class():
    # Load 3-class dataset and split
    data_dir = "/path/to/data"
    featurestable = pd.read_excel(os.path.join(data_dir, 'featurestable.xlsx'))

    # Define hyperparameters and settings for 3-class task
    hyperparameters = {
        "batch_size": [8, 16],
        "lr": [0.01, 0.001],
        "num_neurons": [[8, 3], [32, 3]],
    }

    # Step 2: Train models and perform ablation for multiclass classification
    for ind_comb in range(0, len(ordered_combinations)):
        num_additional_features = len(select_additional_indices(ordered_combinations[ind_comb]))

        cnn = ModifiedResNet18(num_classes=3)  # 3 classes for multiclass
        mlp = MLPModule(cnn_outdim=3, num_neurons=[8, 3])
        combined_model = CombinedModel(cnn, mlp).to(device)

        optimizer = torch.optim.Adam(combined_model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        # Step 3: Perform training
        history, min_val_loss = my_train(combined_model, optimizer, loss_fn, train_dl, val_dl, ind_comb, ordered_combinations)

        # Step 4: Shapley value calculation (optional)
        shapley_vals = shapley_value(coalitions, values)

        # Save results
        torch.save(combined_model.state_dict(), 'best_model.pth')

# Execute ablation study
ablation_3class()
