from __future__ import annotations

from typing import Literal

import torch
import torchvision.models as tv_models


def build_backbone(
    name: Literal["densenet121", "resnet50", "swin_t"], 
    num_classes: int, 
    pretrained: bool = True,
    dropout: float = 0.0,
) -> torch.nn.Module:
    """Build a backbone model with optional dropout.
    
    Args:
        name: Model architecture name
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout probability (0.0 = no dropout)
    """
    if name == "densenet121":
        model = tv_models.densenet121(weights="DEFAULT" if pretrained else None)
        in_features = model.classifier.in_features
        if dropout > 0:
            model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(in_features, num_classes)
            )
        else:
        model.classifier = torch.nn.Linear(in_features, num_classes)
    elif name == "resnet50":
        model = tv_models.resnet50(weights="DEFAULT" if pretrained else None)
        in_features = model.fc.in_features
        if dropout > 0:
            model.fc = torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(in_features, num_classes)
            )
        else:
        model.fc = torch.nn.Linear(in_features, num_classes)
    elif name == "swin_t":
        model = tv_models.swin_t(weights="DEFAULT" if pretrained else None)
        in_features = model.head.in_features
        if dropout > 0:
            model.head = torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(in_features, num_classes)
            )
        else:
        model.head = torch.nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    model.num_classes = num_classes
    return model


