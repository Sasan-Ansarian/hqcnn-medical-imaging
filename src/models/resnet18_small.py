import torch.nn as nn
from torchvision.models import resnet18


def build_resnet18_small(num_classes: int) -> nn.Module:
    """
    ResNet-18 adapted for 28x28 images:
    - replace 7x7 stride-2 conv with 3x3 stride-1
    - remove maxpool
    """
    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m
