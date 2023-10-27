import torch
import torch.nn as nn
from torchvision import models


# Define a multi-modal model
class MultiModalModel(nn.Module):
    def __init__(self, num_classes_image, num_classes_feature):
        super(MultiModalModel, self).__init__()

        self.image_model = models.resnet50(pretrained=True)

        # Modify the last classification layer for the number of image classes
        num_features = self.image_model.fc.in_features
        self.image_model.fc = nn.Linear(num_features, num_classes_image)

        # Create a feature branch with fully connected layers to reshape input
        self.feature_branch = nn.Sequential(
            nn.Linear(1, 256),  # Adjust the input size as needed
            nn.ReLU(),
            nn.Linear(256, num_classes_feature)
        )

    def forward(self, image_input, feature_input):
        # Forward pass for the image branch
        image_output = self.image_model(image_input)

        # Forward pass for the feature branch
        feature_output = self.feature_branch(feature_input)

        # Combine the outputs from both branches
        combined_output = torch.cat((image_output, feature_output), dim=1)

        return combined_output
    
class ResNetModel(nn.Module):
    def __init__(self, num_classes):
        super(ResNetModel, self).__init__()
        # Load a pre-trained ResNet-50 model
        self.resnet = models.resnet50(pretrained=True)
        # Modify the last classification layer for the number of classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # Forward pass through the ResNet model
        output = self.resnet(x)
        return output

