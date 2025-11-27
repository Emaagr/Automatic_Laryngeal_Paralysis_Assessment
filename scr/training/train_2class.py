# src/training/train_2class.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .models_common import CustomDataset, DeviceDataLoader
from .models_common import CombinedModel, MLPModule, ModifiedResNet18
from .config import BATCH_SIZE, LEARNING_RATE, EPOCHS, SEED, LOG_LEVEL

def my_train(model, optimizer, loss_fn, train_loader, val_loader, epochs=30, to_print=True):
    min_val_loss = float('inf')
    epochs_no_improve = 0
    history = []

    try:
        model.load_state_dict(torch.load('init_model.pth'))
    except:
        model = reinit_weights(model, SEED)
        model = model.to(device)

    for epoch in range(1, epochs+1):
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
                print(f'Early stopping after {epoch+1} epochs')
                break
    return history, min_val_loss
