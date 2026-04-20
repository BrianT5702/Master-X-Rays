"""DenseNet-121 + Swin-T dual-branch fusion for multilabel CXR (local + global)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models


class DenseSwinHybrid(nn.Module):
    """Parallel DenseNet-121 and Swin-T, concat pooled features, linear head.

    DenseNet emphasizes fine-grained textures (dense connectivity); Swin-T models
    long-range structure via shifted windows. Both see the same tensor (use
    ImageNet normalization). ROI and ASL live outside this module (dataset + Lightning).
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        w_dd = "DEFAULT" if pretrained else None
        w_sw = "DEFAULT" if pretrained else None

        self.densenet = tv_models.densenet121(weights=w_dd)
        d_dim = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()

        self.swin = tv_models.swin_t(weights=w_sw)
        s_dim = self.swin.head.in_features
        self.swin.head = nn.Identity()

        fused = d_dim + s_dim
        if dropout and float(dropout) > 0.0:
            self.head = nn.Sequential(
                nn.Dropout(p=float(dropout)),
                nn.Linear(fused, num_classes),
            )
        else:
            self.head = nn.Linear(fused, num_classes)

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_d = self.densenet(x)
        z_s = self.swin(x)
        z = torch.cat([z_d, z_s], dim=1)
        return self.head(z)
