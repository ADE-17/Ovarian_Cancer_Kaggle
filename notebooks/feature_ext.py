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

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import xgboost as xgb

def extract_features(model, dataloader):
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Extracting:', leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.extend(outputs.cpu().numpy())
            labels.extend(targets.numpy())
    return np.array(features), np.array(labels)

def save_features(filename, features, labels):
    np.savez(filename, features=features, labels=labels)

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
    
if __name__ == "__main__":
    
    root = r'/home/woody/iwso/iwso092h/mad_ucb_kaggle'
    train_images = r'train_thumbnails'
    image_folder = r'/home/woody/iwso/iwso092h/mad_ucb_kaggle/train_thumbnails'
    train_df = pd.read_csv(os.path.join(root,'data','train.csv'))
    train_wo_tma = train_df[train_df.is_tma==False]
    label_mapping = {'HGSC': 0, 'LGSC': 1, 'EC': 2, 'CC': 3, 'MC': 4}
    train_wo_tma['label'] = train_wo_tma['label'].map(label_mapping)
    train_split = train_wo_tma.reset_index(drop=True)

    # train_split, val_split = train_test_split(train_enc, test_size=0.2, random_state=17, stratify=train_wo_tma.label)
    # train_split = train_split.reset_index(drop=True)
    # val_split = val_split.reset_index(drop=True)
    
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),# Color jitter
        transforms.ToTensor(),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_ucb_dataset = CustomCancerDataset(train_split, image_folder='/home/woody/iwso/iwso092h/mad_ucb_kaggle/train_thumbnails',
                                    transform=train_transforms)

    batch_size = 16
    train_loader = DataLoader(train_ucb_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_ucb_dataset, batch_size=batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = resnet18(pretrained=True)
    resnet = resnet.to(device)
    resnet = torch.nn.Sequential(*(list(resnet.children())[:-1])) 
    
    features, labels = extract_features(resnet, train_loader)

    features_file = '/home/woody/iwso/iwso092h/mad_ucb_kaggle/data/extracted_features.npz'

    save_features(features_file, features, labels)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    print('XGBOOSSSYYYYYYYYYYYYYYYY')
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train, y_train)
        
    accuracy = xgb_model.score(X_test, y_test)
    print(f"Accuracy: {accuracy}")
