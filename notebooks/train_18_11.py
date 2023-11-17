import os
import random
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

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

def to_tensor(x):
    x = x.astype("float32") / 255

    return torch.from_numpy(x).permute(2, 0, 1)

class UCBDataset(Dataset):
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
            mode=0,
        )
        img = concat_tiles(
            img, 16, 256
        )

        img = to_tensor(img)
        
        label_CC  = self.metadata_df.label_CC[idx].astype(int)  
        label_EC  = self.metadata_df.label_EC[idx].astype(int)    
        label_HGSC  = self.metadata_df.label_HGSC[idx].astype(int)    
        label_LGSC  = self.metadata_df.label_LGSC[idx].astype(int)    
        label_MC  = self.metadata_df.label_MC[idx].astype(int)

        return img, torch.tensor([label_CC, label_EC, label_HGSC, label_LGSC, label_MC], dtype=torch.float)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(focal_loss)
    
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=10, save_path='model'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc='training:', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())  # Convert labels to float
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            all_preds = []
            all_labels = []
            for inputs, labels in tqdm(val_loader, desc='validating', leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())  # Convert labels to float
                val_loss += loss.item()
                preds = torch.sigmoid(outputs) > 0.5  # Assuming threshold of 0.5
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # val_balanced_accuracy = accuracy_score(all_labels, all_preds)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {running_loss/len(train_loader):.4f} "
              f"Val Loss: {val_loss/len(val_loader):.4f} "
            #   f"Val Balanced Accuracy: {val_balanced_accuracy:.4f}"
                )

        scheduler.step()
        
        if (epoch + 1) % 3 == 0:
            torch.save(model.state_dict(), f'{save_path}_epoch_{epoch+1}.pth')

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
    
    train_ucb_dataset = UCBDataset(train_split, image_folder='/home/woody/iwso/iwso092h/mad_ucb_kaggle/train_thumbnails')
    val_ucb_dataset = UCBDataset(val_split, image_folder='/home/woody/iwso/iwso092h/mad_ucb_kaggle/train_thumbnails')
    
    batch_size = 4
    train_loader = DataLoader(train_ucb_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ucb_dataset, batch_size=batch_size)
    
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5) 

    criterion = FocalLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)  
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    
    model_save_path = '/home/woody/iwso/iwso092h/mad_ucb_kaggle/saved_models/'
    train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=20, save_path=model_save_path)