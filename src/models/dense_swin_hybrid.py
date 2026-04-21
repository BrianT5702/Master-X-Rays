"""DenseNet-121 + Swin-T dual-branch fusion for multilabel CXR (local + global)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models


class DenseSwinHybrid(nn.Module):
    """Parallel DenseNet-121 and Swin-T with gated fusion before the classifier.

    DenseNet branch: local texture; Swin branch: global context. Pooled Swin features
    use the same path as torchvision Swin (norm -> permute -> AdaptiveAvgPool2d(1) -> flatten).
    A learnable gate scales each branch before concatenation for the final linear head.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
        gate_hidden_dim: int = 256,
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
        gh = max(32, min(int(gate_hidden_dim), fused))
        self.fusion_gate = nn.Sequential(
            nn.Linear(fused, gh),
            nn.ReLU(inplace=True),
            nn.Linear(gh, 2),
            nn.Sigmoid(),
        )

        if dropout and float(dropout) > 0.0:
            self.head = nn.Sequential(
                nn.Dropout(p=float(dropout)),
                nn.Linear(fused, num_classes),
            )
        else:
            self.head = nn.Linear(fused, num_classes)

        self.num_classes = num_classes

    def _swin_pooled(self, x: torch.Tensor) -> torch.Tensor:
        """Swin trunk through global average pool (same as full Swin forward without head)."""
        z = self.swin.features(x)
        z = self.swin.norm(z)
        z = self.swin.permute(z)
        z = self.swin.avgpool(z)
        z = self.swin.flatten(z)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_d = self.densenet(x)
        z_s = self._swin_pooled(x)
        h = torch.cat([z_d, z_s], dim=1)
        g = self.fusion_gate(h)
        z_g = torch.cat([z_d * g[:, 0:1], z_s * g[:, 1:2]], dim=1)
        return self.head(z_g)
