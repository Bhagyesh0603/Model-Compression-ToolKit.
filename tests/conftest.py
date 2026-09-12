"""Shared fixtures. Everything here is synthetic and tiny — the test suite must
never download CIFAR-10 or spawn dataloader workers."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

INPUT_SHAPE = (1, 3, 32, 32)


class TinyCNN(nn.Module):
    """Small conv net with BN and a classifier — enough to exercise every stage."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


@pytest.fixture
def model():
    torch.manual_seed(0)
    return TinyCNN().eval()


@pytest.fixture
def example():
    torch.manual_seed(0)
    return torch.randn(*INPUT_SHAPE)


@pytest.fixture
def loader():
    torch.manual_seed(0)
    data = TensorDataset(torch.randn(64, 3, 32, 32), torch.randint(0, 10, (64,)))
    return DataLoader(data, batch_size=16, num_workers=0)
