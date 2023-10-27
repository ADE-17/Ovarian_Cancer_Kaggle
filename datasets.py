import torch
import pandas as pd
import os
from torch.utils.data import Dataset
from PIL import Image  
from torchvision import transforms


class CustomCancerDataset(Dataset):
    def __init__(self, metadata_df, image_folder, transform=None):
        self.metadata_df = metadata_df
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        image_ids = self.metadata_df.image_id[idx]  
        image_name = os.path.join(self.image_folder, "{}_thumbnail.png".format(image_ids))
        # print(image_name)
        image = Image.open(image_name)
        label = self.metadata_df.is_tma[idx]  
        category = self.metadata_df.label[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, label, category