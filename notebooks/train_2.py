import sys
sys.path.append('/home/woody/iwso/iwso092h/mad_ucb_kaggle')
import pandas as pd
import os
import torchvision.models as models
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
# from datasets import CustomCancerDataset
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
mlb = MultiLabelBinarizer()
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from trainer import train_classification_model
import torch
import pandas as pd
import os
from torch.utils.data import Dataset
from PIL import Image  
from torchvision import transforms

root = r'/home/woody/iwso/iwso092h/mad_ucb_kaggle'
train_images = r'train_thumbnails'
image_folder = r'/home/woody/iwso/iwso092h/mad_ucb_kaggle/train_thumbnails'

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):  # Adjust num_classes based on your task
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 64 * 64, 256)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Adjust dropout rate as needed
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

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
        # label_CC  = self.metadata_df.label_CC[idx].astype(int)  
        # label_EC  = self.metadata_df.label_EC[idx].astype(int)    
        # label_HGSC  = self.metadata_df.label_HGSC[idx].astype(int)    
        # label_LGSC  = self.metadata_df.label_LGSC[idx].astype(int)    
        # label_MC  = self.metadata_df.label_MC[idx].astype(int)    

        if self.transform:
            image = self.transform(image)

        return image, label
        # return image, torch.tensor([label_CC, label_EC, label_HGSC, label_LGSC, label_MC], dtype=torch.float)
        
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        pt = torch.exp(-nn.CrossEntropyLoss(reduction='none')(inputs, targets))
        loss = (1 - pt) ** self.gamma * nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        if self.alpha is not None:
            loss = loss * self.alpha
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

train_df = pd.read_csv(os.path.join(root,'data','train.csv'))
train_wo_tma = train_df[train_df.is_tma==False]
label_mapping = {'HGSC': 0, 'LGSC': 1, 'EC': 2, 'CC': 3, 'MC': 4}
train_wo_tma['label'] = train_wo_tma['label'].map(label_mapping)
train_split, val_split = train_test_split(train_wo_tma, test_size=0.2, random_state=17, stratify=train_wo_tma.label)
train_split = train_split.reset_index(drop=True)
val_split = val_split.reset_index(drop=True)

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),# Color jitter
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48828688, 0.42932517, 0.49162089], std=[0.41380908, 0.37492874, 0.41795654])
])

val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_ucb_dataset = CustomCancerDataset(train_split, image_folder='/home/woody/iwso/iwso092h/mad_ucb_kaggle/train_thumbnails',
                                  transform=train_transforms)
val_ucb_dataset = CustomCancerDataset(val_split, image_folder='/home/woody/iwso/iwso092h/mad_ucb_kaggle/train_thumbnails',
                                  transform=val_transforms)

# class_weights = {0: 0.46067416, 1: 2.48484848, 2: 0.82828283, 3: 1.15492958, 4: 2.82758621}
# sample_weights = [class_weights[label] for _, label in train_ucb_dataset]

# sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

batch_size = 16
train_loader = DataLoader(train_ucb_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ucb_dataset, batch_size=batch_size)

def compute_balanced_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    return balanced_accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())

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

########################################
num_classes = 5  
learning_rate = 0.01
step_size = 10
gamma = 0.1  
num_epochs = 50
########################################
#simple CNN
model = SimpleCNN(num_classes=5)  

#######################################
# ResNet
resnet_model = models.resnet34(pretrained=True)
in_features = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(in_features, num_classes)
model = resnet_model
#########################################3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)

print('Training END')