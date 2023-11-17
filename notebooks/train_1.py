import sys
sys.path.append('/home/woody/iwso/iwso092h/mad_ucb_kaggle')
import pandas as pd
import os
from sklearn.model_selection import train_test_split
# from datasets import CustomCancerDataset
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
mlb = MultiLabelBinarizer()
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torchvision.models as models

import pandas as pd
import os
from PIL import Image  
from torchvision import transforms

root = r'/home/woody/iwso/iwso092h/mad_ucb_kaggle'
train_images = r'train_thumbnails'
image_folder = r'/home/woody/iwso/iwso092h/mad_ucb_kaggle/train_thumbnails'

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
        image = Image.open(image_name)
        label_CC  = self.metadata_df.label_CC[idx].astype(int)  
        label_EC  = self.metadata_df.label_EC[idx].astype(int)    
        label_HGSC  = self.metadata_df.label_HGSC[idx].astype(int)    
        label_LGSC  = self.metadata_df.label_LGSC[idx].astype(int)    
        label_MC  = self.metadata_df.label_MC[idx].astype(int)    

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor([label_CC, label_EC, label_HGSC, label_LGSC, label_MC], dtype=torch.float)

train_df = pd.read_csv(os.path.join(root,'data','train.csv'))
train_wo_tma = train_df[train_df.is_tma==False]
train_enc = pd.get_dummies(train_wo_tma, columns=['label'])
train_enc[['label_CC',
       'label_EC', 'label_HGSC', 'label_LGSC', 'label_MC']] = train_enc[['label_CC',
       'label_EC', 'label_HGSC', 'label_LGSC', 'label_MC']].astype(int)
train_split, val_split = train_test_split(train_enc, test_size=0.2, random_state=17)
train_split = train_split.reset_index(drop=True)
val_split = val_split.reset_index(drop=True)
batch_size = 32

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),  # Rotate by a random angle between -15 and 15 degrees
    transforms.Resize((224, 224)),# Color jitter
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48828688, 0.42932517, 0.49162089], std=[0.41380908, 0.37492874, 0.41795654])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48828688, 0.42932517, 0.49162089], std=[0.41380908, 0.37492874, 0.41795654])
])

train_ucb_dataset = CustomCancerDataset(train_split, image_folder='/home/woody/iwso/iwso092h/mad_ucb_kaggle/train_thumbnails',
                                  transform=train_transforms)
val_ucb_dataset = CustomCancerDataset(val_split, image_folder='/home/woody/iwso/iwso092h/mad_ucb_kaggle/train_thumbnails',
                                  transform=val_transforms)

batch_size = 32
train_loader = DataLoader(train_ucb_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ucb_dataset, batch_size=batch_size)

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

def train_and_evaluate(model, train_loader, val_loader, num_epochs, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = FocalLoss()  # Use the Focal Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Define the learning rate scheduler
    step_size = 3  # Adjust the step size as needed
    gamma = 0.1  # Adjust the gamma factor as needed
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        predictions = []
        true_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                true_labels = [np.argmax(label) for label in true_labels]

        balanced_acc = balanced_accuracy_score(true_labels, predictions)
        # print(predictions)
        # print(true_labels)
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}] | "
              f"Training Loss: {avg_train_loss:.4f} | "
              f"Validation Loss: {avg_val_loss:.4f} | "
              f"Balanced Accuracy: {balanced_acc:.4f}"
                                                        )


resnet_model = models.resnet34(pretrained=True)

num_classes = 10
in_features = resnet_model.fc.in_features
resnet_model.fc = nn.Sequential(
    nn.Linear(in_features, num_classes),
    nn.Softmax(dim=1) 
)

num_epochs = 5
learning_rate = 0.01
train_and_evaluate(resnet_model, train_loader, val_loader, num_epochs, learning_rate)











































