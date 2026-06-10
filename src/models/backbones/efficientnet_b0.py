import torch.nn as nn
import torchvision.models as tv


def build_efficientnet_b0(num_classes: int) -> nn.Module:
    """
    EfficientNet-B0 baseline (minimal modification).
    For backbone comparison, keep the default stem; replace classifier head.
    """
    model = tv.efficientnet_b0(weights=None)

    # classifier is typically Sequential(Dropout, Linear)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model
