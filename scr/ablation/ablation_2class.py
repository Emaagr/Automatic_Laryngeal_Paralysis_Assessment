# src/ablation/ablation_2class.py

import os
import torch
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from training_utils import CustomDataset, DeviceDataLoader, ModifiedResNet18, MLPModule, CombinedModel
from ablation_utils import my_train, shapley_value, reinit_weights, select_additional_indices, he_init

# Ablation-specific configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ablation_2class():
    # Step 1: Load dataset and split (e.g. Normal vs Paralyzed)
    data_dir = "/path/to/data"
    featurestable = pd.read_excel(os.path.join(data_dir, 'featurestable.xlsx'))

    # Example: Prepare dataset (assuming 'Name', 'Class', 'Path' and other features are in the table)
    # Train-test split based on features (you may have a different strategy)
    subjects_train, subjects_test = train_test_split(featurestable['Name'], test_size=0.2, random_state=SEED)
    features_train = featurestable[featurestable['Name'].isin(subjects_train)]
    features_test = featurestable[featurestable['Name'].isin(subjects_test)]

    # Define hyperparameters
    hyperparameters = {
        "batch_size": [8, 16],
        "lr": [0.01, 0.001],
        "num_neurons": [[8, 1], [32, 1]],
    }

    # Step 2: Define ordered feature combinations (you need to define this list)
    ordered_combinations = [
        [2], [3], [4], [2, 3], [2, 4], [3, 4],
        # Add more combinations as necessary
    ]

    # Step 3: Train models and perform ablation for binary classification
    for ind_comb in range(len(ordered_combinations)):
        # Get the indices of additional features to use based on the combination
        additional_feature_indices = select_additional_indices(ordered_combinations[ind_comb])

        # Prepare the dataset with the selected feature combination
        train_ds = CustomDataset(
            subjects=subjects_train,
            features_dataframe=features_train,
            transform=composer
        )
        test_ds = CustomDataset(
            subjects=subjects_test,
            features_dataframe=features_test,
            transform=composer
        )

        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

        train_loader = DeviceDataLoader(train_dl, device)
        test_loader = DeviceDataLoader(test_dl, device)

        # Initialize model
        cnn = ModifiedResNet18(num_classes=1)  # Binary classification: 1 output logit
        mlp = MLPModule(input_size=1 + len(additional_feature_indices), num_layers=2, num_neurons=[8, 1])
        model = CombinedModel(cnn, mlp).to(device)

        # Optimizer and Loss function for Binary Classification
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.BCEWithLogitsLoss()

        # Step 4: Perform training
        history, min_val_loss = my_train(
            model,
            optimizer,
            loss_fn,
            train_loader,
            test_loader,
            ind_comb,
            ordered_combinations,
            epochs=EPOCHS,
            to_print=True,
            device=device
        )

        # Step 5: Shapley value calculation (optional)
        coalitions = [...]  # You need to define how you generate coalitions for Shapley
        values = [...]  # Corresponding values for these coalitions
        shapley_vals = shapley_value(coalitions, values)

        # Save results
        torch.save(model.state_dict(), f'best_model_comb_{ind_comb}.pth')

    print("Ablation Study Completed.")

# Execute ablation study
ablation_2class()

