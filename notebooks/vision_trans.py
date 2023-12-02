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
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
import timm
from pytorch_lightning.callbacks import ModelCheckpoint


def get_tiles(img, tile_size=256, n_tiles=30, mode=0):
    h, w, c = img.shape
    pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
    pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)

    img = np.pad(
        img,
        [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]],
        constant_values=0,
    )
    img = img.reshape(
        img.shape[0] // tile_size, tile_size, img.shape[1] // tile_size, tile_size, 3
    )
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, 3)
    
    idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))
    if len(img) < n_tiles:
        img = np.pad(
            img, [[0, n_tiles - len(img)], [0,0], [0,0], [0,0]], constant_values=255
        )
    # idxs = np.argsort(-img.reshape(img.shape[0], -1).sum(-1))[:n_tiles]
    # print(type(idxs))
    if idxs.shape[0]>n_tiles:
        idxs = idxs[-n_tiles:]
    img = img[idxs]
    
    return img

def concat_tiles(tiles, n_tiles, image_size):
    idxes = list(range(n_tiles))
    
    n_row_tiles = int(np.sqrt(n_tiles))
    img = np.zeros(
        (image_size*n_row_tiles, image_size*n_row_tiles, 3), dtype="uint8"
    )
    
    for h in range(n_row_tiles):
        for w in range(n_row_tiles):
            i = h * n_row_tiles + w
            if len(tiles) > idxes[i]:
                this_img = tiles[idxes[i]]
            else:
                this_img = np.ones((image_size, image_size, 3), dtype="uint8") * 255
                
            h1 = h * image_size
            w1 = w * image_size
            img[h1 : h1 + image_size, w1 : w1 + image_size] = this_img
    return img

def sort_tiles_by_intensity(tiles):
    intensities = np.mean(tiles, axis=(1, 2, 3))  # Calculate mean intensity for each tile
    
    sorted_indices = np.argsort(-intensities)
    
    sorted_tiles = tiles[sorted_indices]
    sorted_intensities = intensities[sorted_indices]
    
    return sorted_tiles, sorted_intensities, sorted_indices

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
        img = get_tiles(
            np.array(image),
            mode=0, n_tiles= 64
        )
        
        sorted_img = sort_tiles_by_intensity(img)
        img = concat_tiles(
            img, 64, 256
        )

        # img = to_tensor(img)
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        label_CC  = self.metadata_df.label_CC[idx].astype(int)  
        label_EC  = self.metadata_df.label_EC[idx].astype(int)    
        label_HGSC  = self.metadata_df.label_HGSC[idx].astype(int)    
        label_LGSC  = self.metadata_df.label_LGSC[idx].astype(int)    
        label_MC  = self.metadata_df.label_MC[idx].astype(int)

        return img, torch.tensor([label_CC, label_EC, label_HGSC, label_LGSC, label_MC], dtype=torch.float)

class ViT(pl.LightningModule):
    def __init__(self, num_classes, image_size=224, backbone='vit_base_patch16_224', finetune_layer=True):
        super().__init__()
        self.model = timm.create_model(backbone, pretrained=True)
        
        if finetune_layer:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.head = nn.Identity()
            self.finetune_layer = nn.Sequential(
                nn.Linear(self.model(torch.randn(1, 3, image_size, image_size)).shape[-1], 512),
                nn.ReLU(),
                nn.Linear(512, num_classes)
            )
        else:
            self.model.head = nn.Linear(self.model.head.in_features, num_classes)
            self.finetune_layer = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        if self.finetune_layer is not None:
            logits = self.finetune_layer(logits)

        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        if self.finetune_layer is not None:
            logits = self.finetune_layer(logits)

        loss = F.cross_entropy(logits, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        if self.finetune_layer is not None:
            parameters = list(self.model.parameters()) + list(self.finetune_layer.parameters())
        else:
            parameters = self.parameters()

        optimizer = torch.optim.Adam(parameters, lr=1e-4)
        return optimizer

if __name__ == "__main__":
    
    root = r'/home/woody/iwso/iwso092h/mad_ucb_kaggle'
    train_images = r'train_thumbnails'
    image_folder = r'/home/woody/iwso/iwso092h/mad_ucb_kaggle/train_thumbnails'
    train_df = pd.read_csv(os.path.join(root,'data','train.csv'))
    train_wo_tma = train_df[train_df.is_tma==False]
    train_enc = pd.get_dummies(train_wo_tma, columns=['label'])
    train_enc[['label_CC',
        'label_EC', 'label_HGSC', 'label_LGSC', 'label_MC']] = train_enc[['label_CC',
        'label_EC', 'label_HGSC', 'label_LGSC', 'label_MC']].astype(int)
    train_split, val_split = train_test_split(train_enc, test_size=0.2, random_state=17)
    train_split = train_split.reset_index(drop=True)
    val_split = val_split.reset_index(drop=True)
    
    data_transforms = transforms.Compose([
        
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    train_ucb_dataset = CustomCancerDataset(train_split, image_folder='/home/woody/iwso/iwso092h/mad_ucb_kaggle/train_thumbnails',
                                            transform=data_transforms)
    val_ucb_dataset = CustomCancerDataset(val_split, image_folder='/home/woody/iwso/iwso092h/mad_ucb_kaggle/train_thumbnails',
                                      transform=data_transforms)
    
    batch_size = 16
    train_loader = DataLoader(train_ucb_dataset, batch_size=batch_size, shuffle=True, num_workers=7)
    val_loader = DataLoader(val_ucb_dataset, batch_size=batch_size, num_workers=7)
    
    model = ViT(num_classes=5, finetune_layer=True)

    checkpoint_callback = ModelCheckpoint(
    dirpath='/home/woody/iwso/iwso092h/mad_ucb_kaggle/saved_models',  # Directory to save the checkpoints
    filename='vit-{epoch:02d}-{val_loss:.4f}',
    save_top_k=1,  # Save the best model based on the lowest validation loss
    mode='min'
    )
    trainer = pl.Trainer(max_epochs=15, log_every_n_steps=15,
                         callbacks=[checkpoint_callback])  
    trainer.fit(model, train_loader, val_loader)