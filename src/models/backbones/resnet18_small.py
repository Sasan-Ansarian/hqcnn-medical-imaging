import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18SmallBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        m = resnet18(weights=None)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()

        self.features = nn.Sequential(
            m.conv1,
            m.bn1,
            m.relu,
            m.maxpool,
            m.layer1,
            m.layer2,
            m.layer3,
            m.layer4,
            m.avgpool,
        )
        self.out_features = m.fc.in_features  # 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


def build_resnet18_small_backbone() -> nn.Module:
    return ResNet18SmallBackbone()
