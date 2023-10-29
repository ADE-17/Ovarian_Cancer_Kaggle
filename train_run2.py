# import timm
import torch.nn as nn
import torchvision
import torch
import pytorch_lightning as pl
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet34
import pandas as pd
import os
from torch.utils.data import Dataset
from PIL import Image  
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import OneCycleLR

import torchmetrics
from torchmetrics.classification import MulticlassF1Score

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning  import Trainer, seed_everything


seed_everything(42, workers=True)

train_data = pd.read_csv("/home/woody/iwso/iwso092h/ucb_kaggle/data/train.csv")
image_folder = r'/home/woody/iwso/iwso092h/ucb_kaggle/train_thumbnails'

class UBCModel(pl.LightningModule):

    def __init__(self, steps_per_epoch):
        super(UBCModel, self).__init__()
        self.num_classes = 5
        self.steps_per_epoch = steps_per_epoch

        self.model = resnet34(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        
        self.criterion = nn.CrossEntropyLoss()
        self.f1 = MulticlassF1Score(num_classes=self.num_classes, average='macro')
        self.accuracy = torchmetrics.Accuracy(num_classes=self.num_classes, task='multiclass')
        self.precision = torchmetrics.Precision(average='macro', num_classes=self.num_classes, task='multiclass')
        self.recall = torchmetrics.Recall(average='macro', num_classes=self.num_classes, task='multiclass')
        
    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('train_loss', loss)
        self.log('train_f1', self.f1(y_pred, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('val_loss', loss)
        self.log('val_f1', self.f1(y_pred, y))
        self.log('val_acc', self.accuracy(y_pred, y))
        self.log('val_precision', self.precision(y_pred, y))
        self.log('val_recall', self.recall(y_pred, y))
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'monitor': 'val_f1',
                'frequency': 1,
                'strict': True,
            }
        }

def preprocess_dataframe(dataframe, image_folder):
    # Get a list of image files in the folder
    image_files = os.listdir(image_folder)

    # Create a set of image IDs from the image files
    image_ids_in_folder = {int(filename.split('_')[0]) for filename in image_files}

    # Filter the dataframe to keep only rows with image IDs present in the folder
    dataframe_filtered = dataframe[dataframe['image_id'].isin(image_ids_in_folder)]
    
    # train_df, val_df = train_test_split(dataframe_filtered, test_size=0.2, stratify=dataframe_filtered.label)
    
    # dataframe = pd.get_dummies(dataframe_filtered, columns=['label'])
    # val_op = pd.get_dummies(val_df, columns=['label'])
    label_mapping = {'HGSC': 0, 'LGSC': 1, 'EC': 2, 'CC': 3, 'MC': 4}
    
    dataframe_filtered['label'] = dataframe_filtered['label'].map(label_mapping)
    
    return dataframe_filtered.reset_index(drop=True)

preprocess_data = preprocess_dataframe(train_data, image_folder)

class ImageClassificationDataModule(pl.LightningDataModule):
    def __init__(self, custom_dataset, batch_size=32):
        super().__init__()
        self.custom_dataset = custom_dataset
        self.batch_size = batch_size

    def setup(self, stage=None):
        num_data = len(self.custom_dataset)
        train_size = int(0.8 * num_data)
        val_size = num_data - train_size
        self.train_data, self.val_data = random_split(self.custom_dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=47)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=12)

class CustomCancerDataset(Dataset):
    def __init__(self, metadata_df, image_folder, transform=None):
        self.metadata_df = metadata_df
        self.image_folder = image_folder
        self.transform = transforms.Compose(
                [transforms.Resize((224, 224)),
                 transforms.ToTensor(), 
                 transforms.Normalize(mean=[0.48828688, 0.42932517, 0.49162089], std=[0.41380908, 0.37492874, 0.41795654])]
            )

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        image_ids = self.metadata_df.image_id[idx]  
        image_name = os.path.join(self.image_folder, "{}_thumbnail.png".format(image_ids))
        # print(image_name)
        image = Image.open(image_name)
        # label_CC  = self.metadata_df.label_CC[idx].astype(int)  
        # label_EC  = self.metadata_df.label_EC[idx].astype(int)    
        # label_HGSC  = self.metadata_df.label_HGSC[idx].astype(int)    
        # label_LGSC  = self.metadata_df.label_LGSC[idx].astype(int)    
        # label_MC  = self.metadata_df.label_MC[idx].astype(int)    
        label = self.metadata_df.label[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

model = UBCModel(50)

# from datasets import CustomCancerDataset
image_folder = r'/home/woody/iwso/iwso092h/ucb_kaggle/train_thumbnails'
custom_dataset = CustomCancerDataset(metadata_df=preprocess_data, image_folder=image_folder)

data_module = ImageClassificationDataModule(custom_dataset, batch_size=32)

trainer = pl.Trainer(
    max_epochs=10, 
    accelerator = 'cpu',
    log_every_n_steps=1)

trainer.fit(model, data_module)















