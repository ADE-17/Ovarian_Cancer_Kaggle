import prepare_data
import pandas as pd
import os
from datasets import CustomCancerDataset
from models import MultiModalModel, ResNetModel
from torchvision import transforms
from torch.utils.data import DataLoader
from trainer import train_classification_model

root = r'C:\Users\ADE17\Desktop\Masters\Projects\Kaggle_UBC_Ocean'
train_images = r'train_thumbnails'
image_folder = r'C:\Users\ADE17\Desktop\Masters\Projects\Kaggle_UBC_Ocean\train_thumbnails'

train_df, val_df = prepare_data.data_split(r'C:\Users\ADE17\Desktop\Masters\Projects\Kaggle_UBC_Ocean\data', )

train_df_f = prepare_data.preprocess_dataframe(train_df, os.path.join(root, train_images))
val_df_f = prepare_data.preprocess_dataframe(val_df, os.path.join(root, train_images))

train_df_pp = prepare_data.custom_label_encode(train_df_f)
val_df_pp = prepare_data.custom_label_encode(val_df_f)

train_reset = train_df_pp.reset_index(drop=True)
val_reset = val_df_pp.reset_index(drop=True)

num_classes_image  = 2
num_classes_feature  = 5
# model = MultiModalModel(num_classes_image, num_classes_feature)
model = ResNetModel(num_classes_image)

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize your data to a specific size
    transforms.ToTensor()  # Convert the data to a PyTorch tensor
])

train_dataset = CustomCancerDataset(train_reset, image_folder, transform = data_transforms)
val_dataset = CustomCancerDataset(val_reset, image_folder, transform = data_transforms)


batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

train_classification_model(model, train_loader, valid_loader, num_epochs=20, lr=0.01, patience=3)