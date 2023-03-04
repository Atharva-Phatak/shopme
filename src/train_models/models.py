import torchvision
from lightly.models.modules.heads import (
    SimCLRProjectionHead,
)
from torch import nn


class SimCLR(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        (
            hidden_dim,
            self.backbone,
        ) = self.create_backbone(backbone)
        self.projection_head = (
            SimCLRProjectionHead(
                hidden_dim, hidden_dim, 128
            )
        )

    def create_backbone(self, backbone):
        model = getattr(
            torchvision.models, backbone
        )(weights=None)
        if isinstance(
            model,
            torchvision.models.efficientnet.EfficientNet,
        ):
            return model.classifier[
                1
            ].in_features, nn.Sequential(
                *list(model.children())[:-1]
            )
        return (
            model.fc.in_features,
            nn.Sequential(
                *list(model.children())[:-1]
            ),
        )

    def forward(self, x):
        # get representations
        f = self.backbone(x).flatten(start_dim=1)
        # get projections
        return self.projection_head(f)
