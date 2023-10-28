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
        label_CC  = self.metadata_df.label_CC[idx]  
        label_EC  = self.metadata_df.label_EC[idx]  
        label_HGSC  = self.metadata_df.label_HGSC[idx]  
        label_LGSC  = self.metadata_df.label_LGSC[idx]  
        label_MC  = self.metadata_df.label_MC[idx]  
        
        if self.transform:
            image = self.transform(image)

        return image, [label_CC, label_EC, label_HGSC, label_LGSC, label_MC]