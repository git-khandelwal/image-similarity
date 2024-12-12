import torch
import torch.nn as nn
import torchvision.models as models


class AutoEncoder(nn.Module):
    def __init__(self, embed_size, num_classes):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(9408, 1024), # 224*224*3
            nn.ReLU(),
            nn.Linear(1024, embed_size)
        )
        self.bn = nn.BatchNorm1d(embed_size)
        self.classifier = nn.Sequential(
            nn.Linear(embed_size, num_classes)
        )

    def forward(self, images):
        features = self.encoder(images.view(-1, 9408))
        features = features.view(features.size(0), -1)
        features = self.bn(features)
        predictions = self.classifier(features)
        return features, predictions
