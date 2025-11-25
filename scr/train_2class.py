"""
2-class classification (Normal vs Paralyzed) pipeline.

This script reproduces the original CNN+MLP.ipynb binary classification including:
- data paths
- subject-level splitting strategy
- CNN + MLP architecture
- BCEWithLogitsLoss with pos_weight
- hyperparameter search
- evaluation at trial and patient level

"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as seaborn
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    average_precision_score,
    roc_auc_score,
    RocCurveDisplay,
)
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

# -------------------------------------------------------
# Dataset (1 Dataset section, as used in your notebook)
# -------------------------------------------------------

data_dir = "/content/drive/MyDrive/AI in healthcare/Laryngeal paralysis/Spain Dataset/Trials"
datasetpath = os.path.join(data_dir, "Difference Images")

featurestable = pd.read_excel(
    "/content/drive/MyDrive/AI in healthcare/Laryngeal paralysis/Spain Dataset/Trials/Agati features.xlsx"
)

# Drop velocity-related columns (as in the notebook)
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

# Subject-level target (multiclass at first, then binarized)
target = []
for subj in spa_subj:
    mask = featurestable["Name"].str.contains(subj)
    target.append(featurestable.loc[mask, "Class"].iloc[0])

# Binary: 0 vs >0
target = (np.array(target) > 0) * 1

subjects_train, subjects_test = train_test_split(
    spa_subj, test_size=0.2, random_state=seed, stratify=target
)

target = []
for subj in subjects_train:
    mask = featurestable["Name"].str.contains(subj)
    target.append(featurestable.loc[mask, "Class"].iloc[0])
target = (np.array(target) > 0) * 1

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

# Binary labels: Class > 0 -> 1, else 0
def binarize_label(y):
    return int(np.array(y) > 0)


train_dataset = CustomDataset(
    subjects=subjects_train,
    features_dataframe=features_train,
    transform=composer,
    label_transform=binarize_label,
)
val_dataset = CustomDataset(
    subjects=subjects_val,
    features_dataframe=features_val,
    transform=composer,
    label_transform=binarize_label,
)
test_dataset = CustomDataset(
    subjects=subjects_test,
    features_dataframe=features_test,
    transform=composer,
    label_transform=binarize_label,
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuf_
