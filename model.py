import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, num_classes):
        super(EncoderCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding="same"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, 3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, 3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.feature_vector = nn.Linear(28*28*64, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.classifier = nn.Linear(embed_size, num_classes)

    def forward(self, images):
        features = self.conv(images)
        features = self.bn(self.feature_vector(features.view(-1, 28*28*64)))
        predictions = self.classifier(features)
        return features, predictions
