import os
import random
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from PIL import Image

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, pos_weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, labels):
        num_classes = logits.size(1)
        
        # Convert labels to one-hot encoding
        labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
        
        # Calculate probabilities with softmax
        probs = F.softmax(logits, dim=1)
        
        # Calculate focal loss
        pos_loss = -labels_one_hot * ((1 - probs) ** self.gamma) * torch.log(probs + 1e-10)
        if self.pos_weight is not None:
            pos_loss *= self.pos_weight
        
        neg_loss = -(1 - labels_one_hot) * (probs ** self.gamma) * torch.log(1 - probs + 1e-10)
        
        loss = (pos_loss + neg_loss).sum(dim=1).mean()
        
        return loss
    
class CustomCancerDataset(Dataset):
    def __init__(self, metadata_df, image_folder, transform=None):
        self.metadata_df = metadata_df
        self.image_folder = image_folder
        self.transform = transform  # Use the provided transform

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        image_ids = self.metadata_df.image_id[idx]  
        image_name = os.path.join(self.image_folder, "{}_thumbnail.png".format(image_ids))
        image = Image.open(image_name).convert('RGB')
        label = int(self.metadata_df.label[idx])
        if self.transform:
            image = self.transform(image)

        return image, label

class ComplexModel(nn.Module):
    def __init__(self, num_classes=5):
        super(ComplexModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 128 * 28 * 28)  # Adjust the flattened size based on your data
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        probabilities = self.softmax(x)
        return probabilities

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_samples = 0
    all_labels = []
    all_predictions = []

    for inputs, labels in tqdm(train_loader, leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(torch.argmax(outputs, 1).cpu().numpy())

        total_samples += labels.size(0)

    average_loss = running_loss / len(train_loader)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_predictions)

    return average_loss, balanced_accuracy

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(torch.argmax(outputs, 1).cpu().numpy())

            total_samples += labels.size(0)

    average_loss = running_loss / len(val_loader)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_predictions)

    return average_loss, balanced_accuracy

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=5):
    for epoch in range(num_epochs):
        train_loss, train_balanced_accuracy = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_balanced_accuracy = validate(model, val_loader, criterion, device)

        print(f'Epoch [{epoch+1}/{num_epochs}] | '
              f'Training Loss: {train_loss:.4f} | Training Balanced Accuracy: {train_balanced_accuracy:.4f} | '
              f'Validation Loss: {val_loss:.4f} | Validation Balanced Accuracy: {val_balanced_accuracy:.4f}')

    print('Training finished.')

if __name__ == "__main__":
    
    root = r'/home/woody/iwso/iwso092h/mad_ucb_kaggle'
    train_images = r'train_thumbnails'
    image_folder = r'/home/woody/iwso/iwso092h/mad_ucb_kaggle/train_thumbnails'
    train_df = pd.read_csv(os.path.join(root,'data','train.csv'))
    train_wo_tma = train_df[train_df.is_tma==False]
    label_mapping = {'HGSC': 0, 'LGSC': 1, 'EC': 2, 'CC': 3, 'MC': 4}
    train_wo_tma['label'] = train_wo_tma['label'].map(label_mapping)
    train_split, val_split = train_test_split(train_wo_tma, test_size=0.2, random_state=17, stratify=train_wo_tma.label)
    train_split = train_split.reset_index(drop=True)
    val_split = val_split.reset_index(drop=True)

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),# Color jitter
        transforms.ToTensor(),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_ucb_dataset = CustomCancerDataset(train_split, image_folder='/home/woody/iwso/iwso092h/mad_ucb_kaggle/train_thumbnails',
                                    transform=train_transforms)
    val_ucb_dataset = CustomCancerDataset(val_split, image_folder='/home/woody/iwso/iwso092h/mad_ucb_kaggle/train_thumbnails',
                                    transform=val_transforms)

    batch_size = 16
    train_loader = DataLoader(train_ucb_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ucb_dataset, batch_size=batch_size)
    
    model = ComplexModel()

    learning_rate = 0.001
    step_size = 10
    gamma = 0.1  
    num_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = FocalLoss(gamma=2.0)
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)
    
