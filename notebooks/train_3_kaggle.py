import sys
sys.path.append('/home/woody/iwso/iwso092h/mad_ucb_kaggle')
import pandas as pd
import timm
import os
import torchvision.models as models
from tqdm import tqdm
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
# from datasets import CustomCancerDataset
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
mlb = MultiLabelBinarizer()
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
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
import cv2

CONFIG = {
    "seed": 42,
    "img_size": 512,
    "model_name": "tf_efficientnetv2_s_in21ft1k",
    "num_classes": 5,
    "valid_batch_size": 32,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

class UBCDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.file_names = df['file_path'].values
        self.labels = df['label'].values
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[index]
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return {
            'image': img,
            'label': torch.tensor(label, dtype=torch.long)
        }
        
data_transforms = {
    "valid": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.)
}

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'
                
class UBCModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=False, checkpoint_path=None):
        super(UBCModel, self).__init__()
        self.model = timm.create_model('tf_efficientnetv2_s_in21ft1k', pretrained=pretrained)

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.linear = nn.Linear(in_features, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        output = self.linear(pooled_features)
        return output
    
train_dataset = UBCDataset(df, transforms=data_transforms["valid"])
test_loader = DataLoader(test_dataset, batch_size=CONFIG['valid_batch_size'], 
                          num_workers=2, shuffle=False, pin_memory=True)
    
    
    
    
    
    
    
    
    
    
    
    