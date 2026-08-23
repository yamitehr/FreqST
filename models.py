"""A tiny CNN classifier, small enough to train on CPU in a few minutes.
Used identically across all 4 preprocessing variants in Stage 3 -- only the
input channels' *content* changes, never the architecture.
"""
import torch.nn as nn


class TinyCNN(nn.Module):
    def __init__(self, num_classes=4, in_channels=3, base_channels=16):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c, 2 * c, 3, padding=1), nn.BatchNorm2d(2 * c), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * c, 4 * c, 3, padding=1), nn.BatchNorm2d(4 * c), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(4 * c, 4 * c, 3, padding=1), nn.BatchNorm2d(4 * c), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(4 * c, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)
