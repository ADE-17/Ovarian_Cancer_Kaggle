# UBC-OCEAN: Ovarian Cancer Subtype Classification and Outlier Detection

Welcome to the UBC Ovarian Cancer subtypE clAssification and outlier detectioN (UBC-OCEAN) Kaggle competition! In this competition, you will tackle the challenging task of classifying ovarian cancer subtypes using histopathology images from a diverse dataset. Your contributions have the potential to revolutionize ovarian cancer diagnosis and improve patient outcomes.

Competition Details
Start Date: 22 days ago
Deadline: 2 months to go
Competition Hosts: University of British Columbia (UBC), BC Cancer, Ovarian Tumour Tissue Analysis (OTTA) Consortium, TD Bank Group
Problem Statement
Ovarian carcinoma is a highly lethal cancer with distinct subtypes, each requiring specific treatment approaches. The primary task is to classify ovarian cancer subtypes, which include high-grade serous carcinoma, clear-cell ovarian carcinoma, endometrioid, low-grade serous, and mucinous carcinoma. Additionally, there are rare subtypes ("Outliers") with unique characteristics. Currently, subtype identification relies on pathologists, presenting challenges such as observer disagreements and diagnostic reproducibility issues.

Deep learning models offer a promising solution for analyzing histopathology images, and this competition provides access to a vast dataset from over 20 medical centers worldwide.

Competition Impact
Successful solutions from this competition could significantly improve the accuracy of ovarian cancer subtype identification. This would enable clinicians to develop personalized treatment strategies, regardless of geographic location, potentially enhancing treatment efficacy and patient outcomes for this deadly cancer.

Getting Started
Data: Download the competition data from the Kaggle competition data page.

Code Repository: Clone or fork the GitHub repository for this project: Your GitHub Repository URL

Environment Setup: Set up your development environment with the required libraries, including PyTorch, PyTorch Lightning, and any other dependencies. You can use a requirements.txt file or a Conda environment file for reproducibility.

Data Exploration: Explore the dataset to understand its structure and content. Here's how you can create a custom dataset class and data loader for your project using PyTorch and PyTorch Lightning. You can find the complete code in the GitHub repository:

# Code snippet for creating a custom dataset class
class CustomCancerDataset(Dataset):
    # ... (your dataset class code)
    
# Code snippet for creating a PyTorch Lightning DataModule
class ImageClassificationDataModule(pl.LightningDataModule):
    # ... (your DataModule code)

# Initialize the model, data module, and trainer
model = ImageClassificationModel(num_classes=5)
custom_dataset = CustomCancerDataset(metadata_df=your_metadata_df, image_folder=your_image_folder)
data_module = ImageClassificationDataModule(custom_dataset, batch_size=32)
trainer = pl.Trainer(max_epochs=10, gpus=1)

# Start training
trainer.fit(model, data_module)

# Making Predictions
Once you have trained your model, you can make predictions on the test data. Here's a code snippet to help you get started:

# Create a data loader for the test dataset
# ... (create test data loader code)

# Set the model to evaluation mode
model.eval()

# List to store predictions
predictions = []

# Make predictions on the test data
# ... (make predictions on test data code)

# Convert predictions to a list
# ... (convert predictions to a list code)

Guidelines
Please follow the competition rules and guidelines on the Kaggle competition page.
Ensure that you do not use the competition data for external projects or research until the official publication of the competition paper.
Evaluation
Submissions for this competition are evaluated using balanced accuracy.
