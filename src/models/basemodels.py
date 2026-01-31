from __future__ import annotations

from typing import Literal

import torch
import torchvision.models as tv_models


def build_backbone(
    name: Literal["densenet121", "resnet50", "swin_t"], num_classes: int, pretrained: bool = True
) -> torch.nn.Module:
    if name == "densenet121":
        model = tv_models.densenet121(weights="DEFAULT" if pretrained else None)
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, num_classes)
    elif name == "resnet50":
        model = tv_models.resnet50(weights="DEFAULT" if pretrained else None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    elif name == "swin_t":
        model = tv_models.swin_t(weights="DEFAULT" if pretrained else None)
        in_features = model.head.in_features
        model.head = torch.nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    model.num_classes = num_classes
    return model


