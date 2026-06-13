"""Test-time augmentation wrappers for evaluation / threshold search."""

from __future__ import annotations

import torch
import torch.nn as nn


class HFlipLogitsTTA(nn.Module):
    """Average logits from original and horizontally flipped inputs (standard CXR TTA)."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z0 = self.inner(x)
        z1 = self.inner(torch.flip(x, dims=[-1]))
        return 0.5 * (z0 + z1)
