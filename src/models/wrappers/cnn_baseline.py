import torch
import torch.nn as nn


class CNNBaselineModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        freeze_backbone: bool = False,
        unfreeze_last_block: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head

        if freeze_backbone:
            self.freeze_backbone()

        if unfreeze_last_block:
            self.freeze_all_backbone()
            self.unfreeze_last_block()

    def freeze_all_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def freeze_backbone(self):
        self.freeze_all_backbone()

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def unfreeze_last_block(self):
        if hasattr(self.backbone, "features") and len(self.backbone.features) > 7:
            for p in self.backbone.features[7].parameters():
                p.requires_grad = True
        else:
            raise ValueError("Backbone does not expose features[7] for partial unfreezing")

    def get_backbone_params(self):
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def get_head_params(self):
        return [p for p in self.head.parameters() if p.requires_grad]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        return logits
