## src/training/train_3class.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .training_utils import CustomDataset, DeviceDataLoader
from .training_utils import CombinedModel, MLPModule, ModifiedResNet18
from .config import BATCH_SIZE, LEARNING_RATE, EPOCHS, SEED, LOG_LEVEL
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def my_train(model, optimizer, loss_fn, train_loader, val_loader, epochs=30, to_print=True):
    min_val_loss = float('inf')
    epochs_no_improve = 0
    history = []

    try:
        model.load_state_dict(torch.load('init_model.pth'))
    except:
        model = reinit_weights(model, SEED)
        model = model.to(device)

    for epoch in range(1, epochs + 1):
        training_loss = 0.0
        valid_loss = 0.0
        total_train_samples = 0
        model.train()

        for batch in train_loader:
            images = batch['image']
            additional_features = batch['additional_features']
            labels = batch['labels'].float()
            outputs = model(images, additional_features)
            outputs = outputs.view(-1)
            loss = loss_fn(torch.reshape(outputs, (-1, 1)), torch.reshape(labels, (-1, 1)))
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
                images = batch['image']
                additional_features = batch['additional_features']
                labels = batch['labels'].float()
                outputs = model(images, additional_features)
                outputs = outputs.view(-1)
                loss = loss_fn(torch.reshape(outputs, (-1, 1)), torch.reshape(labels, (-1, 1)))
                valid_loss += loss.data.item() * images.size(0)
                total_val_samples += images.size(0)
                predicted = np.resize((outputs.cpu().numpy() > 0) * 1, labels.size(0))
                total += labels.size(0)
                correct += (predicted == labels.cpu().numpy()).sum().item()
            valid_loss /= total_val_samples
            accy = correct / total

            if to_print:
                print(f'Epoch: {epoch}, Training Loss: {training_loss:.4f}, Validation Loss: {valid_loss:.4f}, accuracy = {accy:.4f}')
            history.append({
                "loss": training_loss,
                "val_loss": valid_loss,
                "val_acc": accy
            })

            if valid_loss < min_val_loss:
                min_val_loss = valid_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                epochs_no_improve += 1

            if (epoch > 50) and (epochs_no_improve > 20):
                print(f'Early stopping after {epoch + 1} epochs')
                break
    return history, min_val_loss

def train_and_evaluate(subjects_train, features_train, subjects_test, features_test, device):
    # Data loader initialization
    train_loader = DataLoader(CustomDataset(subjects=subjects_train, features_dataframe=features_train, transform=composer), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(CustomDataset(subjects=subjects_test, features_dataframe=features_test, transform=composer), batch_size=BATCH_SIZE, shuffle=True)

    # Model initialization
    cnn_outdim = 3  # 3 classes for classification
    cnn = ModifiedResNet18(num_classes=cnn_outdim)
    mlp = MLPModule(input_size=cnn_outdim, num_layers=2, num_neurons=[8, 3])
    model = CombinedModel(cnn, mlp).to(device)

    # Optimizer and Loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss().to(device)

    # Train the model
    history, min_val_loss = my_train(model, optimizer, loss_fn, train_loader, test_loader, epochs=EPOCHS)

    # Save the final model
    torch.save(model.state_dict(), 'best_3class_model.pth')

    # Evaluation and results reporting
    all_labels, all_preds, all_probs, all_features = get_trials_prediction(model, test_loader, ['Class_1', 'Class_2', 'Class_3'], encoder=None)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title('Confusion Matrix for 3-Class Classification')
    plt.show()

    # Classification Report
    report = classification_report(all_labels, all_preds)
    print(report)

