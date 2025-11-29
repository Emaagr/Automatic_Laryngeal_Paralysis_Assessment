# src/training_and_test/train_test_3class.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from .training_test_utils.py import (
    CustomDataset,
    DeviceDataLoader,
    CombinedModel,
    MLPModule,
    ModifiedResNet18,
    composer,
    reinit_weights,
    get_trials_prediction,
)
from .config import BATCH_SIZE, LEARNING_RATE, EPOCHS, SEED


def my_train(model, optimizer, loss_fn, train_loader, val_loader,
             epochs: int = 30, to_print: bool = True, device=None):
    """
    Training loop per modello multiclasse (3 classi) con CrossEntropyLoss.

    - output del modello: (B, 3)
    - labels: LongTensor (B,) con valori {0,1,2}
    """
    min_val_loss = float('inf')
    epochs_no_improve = 0
    history = []

    # Provo a caricare uno stato iniziale; se non esiste, reinizializzo i pesi
    try:
        state_dict = torch.load('init_model.pth', map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
    except Exception:
        model = reinit_weights(model, SEED).to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        training_loss = 0.0
        total_train_samples = 0

        # -------------------- TRAIN --------------------
        for batch in train_loader:
            images = batch['image']                  # già su device (DeviceDataLoader)
            additional_features = batch['additional_features']
            labels = batch['labels']                 # LongTensor (B,)

            optimizer.zero_grad()
            outputs = model(images, additional_features)   # (B,3)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item() * images.size(0)
            total_train_samples += images.size(0)

        training_loss /= max(total_train_samples, 1)

        # -------------------- VALIDATION --------------------
        model.eval()
        valid_loss = 0.0
        total_val_samples = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image']
                additional_features = batch['additional_features']
                labels = batch['labels']

                outputs = model(images, additional_features)   # (B,3)
                loss = loss_fn(outputs, labels)

                valid_loss += loss.item() * images.size(0)
                total_val_samples += images.size(0)

                preds = outputs.argmax(dim=1)                 # (B,)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        valid_loss /= max(total_val_samples, 1)
        val_acc = correct / max(total, 1)

        if to_print:
            print(
                f"Epoch: {epoch}, "
                f"Training Loss: {training_loss:.4f}, "
                f"Validation Loss: {valid_loss:.4f}, "
                f"Validation Accuracy: {val_acc:.4f}"
            )

        history.append({
            "loss": training_loss,
            "val_loss": valid_loss,
            "val_acc": val_acc
        })

        # Early stopping
        if valid_loss < min_val_loss:
            min_val_loss = valid_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1

        if (epoch > 50) and (epochs_no_improve > 20):
            print(f"Early stopping after {epoch} epochs")
            break

    return history, min_val_loss


def train_and_evaluate(subjects_train, features_train,
                       subjects_test, features_test, device):
    """
    Training + valutazione per modello a 3 classi (CrossEntropy).
    """
    # -------------------- DATASET & DATALOADERS --------------------
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

    num_additional_features = len(train_ds.feature_cols)

    # -------------------- MODELLO --------------------
    # Dimensione delle feature estratte dalla CNN
    cnn_feature_dim = 32  # puoi modificarlo (16/32/64) purché coerente con l'MLP

    # CNN come estrattore di feature
    cnn = ModifiedResNet18(num_classes=cnn_feature_dim, pretrained=True)

    # MLP: input = feature_CNN + feature_tabellari, output = 3 logit
    mlp_input_size = cnn_feature_dim + num_additional_features
    mlp = MLPModule(input_size=mlp_input_size,
                    num_layers=2,
                    num_neurons=[64, 3])

    model = CombinedModel(cnn, mlp).to(device)

    # -------------------- LOSS & OPTIMIZER --------------------
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # -------------------- TRAINING --------------------
    history, min_val_loss = my_train(
        model,
        optimizer,
        loss_fn,
        train_loader,
        test_loader,   # come validation
        epochs=EPOCHS,
        to_print=True,
        device=device
    )

    # Salva il modello finale
    torch.save(model.state_dict(), 'best_3class_model.pth')

    # Se vuoi usare il best (early stopping):
    # model.load_state_dict(torch.load('best_model.pth', map_location=device))
    # model.to(device)

    # -------------------- VALUTAZIONE --------------------
    class_names = ['Class_0', 'Class_1', 'Class_2']  # oppure ['Healthy','Unilateral','Bilateral']

    all_labels, all_preds, all_probs, all_features = get_trials_prediction(
        model,
        test_loader,
        classes=class_names,
        encoder=None
    )

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix for 3-Class Classification')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

    # Classification Report
    report = classification_report(all_labels, all_preds,
                                   target_names=class_names)
    print(report)

    return model, history, cm, report


