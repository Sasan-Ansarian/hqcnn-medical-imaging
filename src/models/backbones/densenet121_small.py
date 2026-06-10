import torch.nn as nn
import torchvision.models as tv


def build_densenet121_small(num_classes: int) -> nn.Module:
    """
    DenseNet-121 adapted for 28x28 images (MedMNIST / DermaMNIST).
    - Replace initial 7x7 stride-2 conv with 3x3 stride-1 conv
    - Remove initial maxpool
    - Replace classifier head
    """
    model = tv.densenet121(weights=None)

    model.features.conv0 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.features.pool0 = nn.Identity()

    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model
