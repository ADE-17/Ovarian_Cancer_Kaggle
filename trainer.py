import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score

from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

def train_classification_model(model, train_loader, valid_loader, num_epochs=20, lr=0.01, patience=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1)  # LR scheduler

    best_valid_loss = np.inf
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, labels, features in train_loader:
            inputs, labels, features = inputs.to(device), labels.to(device), features.to(device)
            optimizer.zero_grad()
            # outputs = model(inputs, features)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        predicted_labels = []
        true_labels = []
        with torch.no_grad():
            for inputs, labels, features in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device), features.to(device)
                outputs = model(inputs)
                # outputs = model(inputs, features)

                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)
        accuracy = 100 * correct / total

        balanced_acc = balanced_accuracy_score(true_labels, predicted_labels)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, "
              f"Valid Loss: {valid_loss:.4f}, "
              f"Accuracy: {accuracy:.2f}%, "
              f"Balanced Accuracy: {balanced_acc:.4f}")

        # LR scheduler step
        scheduler.step(valid_loss)

        # Early stopping based on validation loss
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print("Early stopping after {} epochs of no improvement.".format(patience))
            break

    print("Training completed!")

# Usage example:
# train_classification_model(your_model, your_train_data_loader, your_valid_data_loader, num_epochs=10, lr=0.001, patience=3)
