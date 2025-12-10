"""Visualization utilities for explainable AI outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def visualize_heatmap(
    image: torch.Tensor,
    heatmap: torch.Tensor,
    save_path: Optional[Path] = None,
    alpha: float = 0.6,
    cmap: str = "jet",
    show: bool = True,
) -> None:
    """
    Visualize heatmap overlaid on original image.

    Args:
        image: Original image tensor (C, H, W) or (H, W, C)
        heatmap: Heatmap tensor (H, W)
        save_path: Path to save visualization
        alpha: Transparency of heatmap overlay
        cmap: Colormap for heatmap
        show: Whether to display the plot
    """
    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        # Denormalize if needed (assuming ImageNet normalization)
        if image.max() <= 1.0 and image.min() >= 0.0:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image * std + mean
        image = image.permute(1, 2, 0).cpu().numpy()
        image = np.clip(image, 0, 1)

    # Convert heatmap to numpy
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()

    # Resize heatmap to match image if needed
    if heatmap.shape != image.shape[:2]:
        # Use PyTorch interpolation instead of scipy
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
        heatmap_tensor = F.interpolate(
            heatmap_tensor,
            size=(image.shape[0], image.shape[1]),
            mode='bilinear',
            align_corners=False
        )
        heatmap = heatmap_tensor.squeeze().detach().numpy()

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Heatmap
    im = axes[1].imshow(heatmap, cmap=cmap)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1])

    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(heatmap, cmap=cmap, alpha=alpha)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def save_heatmap_batch(
    images: torch.Tensor,
    heatmaps: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
    output_dir: Path,
    prefix: str = "heatmap",
) -> None:
    """
    Save batch of heatmap visualizations.

    Args:
        images: Batch of images (B, C, H, W)
        heatmaps: Batch of heatmaps (B, H, W)
        predictions: Model predictions (B, num_classes)
        labels: Ground truth labels (B, num_classes)
        class_names: List of class names
        output_dir: Directory to save visualizations
        prefix: Filename prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = images.shape[0]
    probs = torch.sigmoid(predictions)

    for i in range(batch_size):
        image = images[i]
        heatmap = heatmaps[i]
        pred = probs[i]
        label = labels[i]

        # Create filename with predictions
        pred_str = "_".join(
            [f"{name}_{pred[j]:.2f}" for j, name in enumerate(class_names)]
        )
        label_str = "_".join(
            [f"{name}_{int(label[j])}" for j, name in enumerate(class_names)]
        )

        save_path = output_dir / f"{prefix}_{i:03d}_pred_{pred_str}_gt_{label_str}.png"

        visualize_heatmap(image, heatmap, save_path=save_path, show=False)

