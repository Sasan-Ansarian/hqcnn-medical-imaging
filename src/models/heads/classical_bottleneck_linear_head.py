import torch.nn as nn


class ClassicalBottleneckLinearHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        bottleneck_dim: int,
        num_classes: int,
        hidden_dim: int | None = None,
    ):
        super().__init__()

        if hidden_dim is None:
            self.net = nn.Sequential(
                nn.Linear(in_features, bottleneck_dim),
                nn.Linear(bottleneck_dim, num_classes),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.Linear(hidden_dim, bottleneck_dim),
                nn.Linear(bottleneck_dim, num_classes),
            )

    def forward(self, x):
        return self.net(x)
