from sklearn.model_selection import train_test_split
import os
import pandas as pd

def data_split(data_path):
    
    ##### Create train Val split
    metadata = pd.read_csv(os.path.join(data_path, "train.csv"))

    train_data, val_data = train_test_split(
        metadata, test_size=0.2, stratify=metadata.is_tma, random_state=12000
    )

    # train_data.to_csv(os.path.join(data_path, "train_dataframe.csv"), index=False)
    # val_data.to_csv(os.path.join(data_path, "val_dataframe.csv"), index=False)

    return train_data, val_data

def custom_label_encode(df):

    label_mapping = {'HGSC': 0, 'LGSC': 1, 'EC': 2, 'CC': 3, 'MC': 4}
    
    df['label'] = df['label'].map(label_mapping)

    # tma_mapping = {'True': 1, 'False': 0}
    
    df['is_tma'] = df['is_tma'].astype(int)
    
    return df

def preprocess_dataframe(dataframe, image_folder):
    # Get a list of image files in the folder
    image_files = os.listdir(image_folder)

    # Create a set of image IDs from the image files
    image_ids_in_folder = {int(filename.split('_')[0]) for filename in image_files}

    # Filter the dataframe to keep only rows with image IDs present in the folder
    dataframe_filtered = dataframe[dataframe['image_id'].isin(image_ids_in_folder)]

    return dataframe_filtered