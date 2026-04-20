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
    alpha: float = 0.38,
    cmap: str = "jet",
    show: bool = True,
    ghosted_overlay: bool = True,
    figure_title: Optional[str] = None,
    heatmap_blur_kernel: int = 0,
    heatmap_spread_gamma: float = 1.0,
    heatmap_zero_border_frac: float = 0.0,
    heatmap_percentile_low: float = 0.0,
    heatmap_percentile_high: float = 100.0,
    heatmap_upsample_mode: str = "bicubic",
) -> None:
    """
    Visualize heatmap overlaid on original image.

    Args:
        image: Original image tensor (C, H, W), normalized with mean=0.5, std=0.5 (range [-1,1])
        heatmap: Heatmap tensor (H, W), typically min-max normalized to [0,1] by Grad-CAM
        save_path: Path to save visualization
        alpha: Transparency of heatmap overlay
        cmap: Colormap for heatmap (jet = blue low, red high)
        show: Whether to display the plot
        ghosted_overlay: If True, overlay uses a grayscale copy of the image at full contrast (no dimming).
            If False, uses the same RGB/RGBA tensor as the Original panel.
        figure_title: Optional suptitle (e.g. image id + target class) so saved PNGs are easy to tell apart
        heatmap_blur_kernel: Optional odd kernel size for smoothing the heatmap (0 disables)
        heatmap_spread_gamma: Gamma applied to heatmap in [0,1]; values <1.0 expand highlighted area;
            values >1.0 suppress weak activations (less green wash, peaks stand out).
        heatmap_zero_border_frac: If >0, zero outer fraction of min(H,W) on each side then renormalize.
            Reduces misleading hot spots on the image frame (display only; does not change the model).
        heatmap_percentile_low / heatmap_percentile_high: If not (0, 100), contrast-stretch using
            these percentiles (clamps tails) so mid-level noise maps to cooler colors.
        heatmap_upsample_mode: "bicubic" or "bilinear" when resizing CAM to image size.
    """
    # Convert image to numpy and denormalize
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        # Our dataset uses Normalize(mean=0.5, std=0.5) -> values in [-1, 1]. Reverse: x*0.5+0.5
        if image.min() < 0 or image.max() > 1.01:  # Likely [-1,1] normalized
            image = image * 0.5 + 0.5
        image = np.clip(image, 0, 1)

    # Convert heatmap to numpy; Grad-CAM already min-max normalizes to [0,1]
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
    # Base [0,1] mapping, then optional percentile contrast (reduces vague green regions)
    h_min, h_max = float(heatmap.min()), float(heatmap.max())
    if h_max - h_min > 1e-8:
        heatmap = (heatmap - h_min) / (h_max - h_min)
    heatmap = np.clip(heatmap, 0, 1)

    use_pct = (
        heatmap_percentile_low > 0.0
        or heatmap_percentile_high < 100.0
    ) and heatmap_percentile_high > heatmap_percentile_low + 1e-6
    if use_pct:
        flat = heatmap.ravel()
        lo = float(np.percentile(flat, heatmap_percentile_low))
        hi = float(np.percentile(flat, heatmap_percentile_high))
        if hi - lo > 1e-8:
            heatmap = np.clip((heatmap - lo) / (hi - lo), 0.0, 1.0)
        else:
            heatmap = np.zeros_like(heatmap)

    # Gamma: >1.0 suppresses weak activations (sharper-looking overlays)
    if heatmap_spread_gamma > 0 and abs(heatmap_spread_gamma - 1.0) > 1e-6:
        heatmap = np.power(np.clip(heatmap, 0, 1), heatmap_spread_gamma)
        hm_min, hm_max = heatmap.min(), heatmap.max()
        if hm_max - hm_min > 1e-8:
            heatmap = (heatmap - hm_min) / (hm_max - hm_min)

    # Resize heatmap to match image if needed
    if heatmap.shape != image.shape[:2]:
        heatmap_tensor = torch.from_numpy(heatmap).float().unsqueeze(0).unsqueeze(0)
        mode = (heatmap_upsample_mode or "bicubic").lower()
        if mode not in ("bicubic", "bilinear"):
            mode = "bicubic"
        heatmap_tensor = F.interpolate(
            heatmap_tensor,
            size=(image.shape[0], image.shape[1]),
            mode=mode,
            align_corners=False,
        )
        heatmap = np.clip(heatmap_tensor.squeeze().numpy(), 0.0, 1.0)

    if heatmap_zero_border_frac > 0:
        H, W = heatmap.shape
        t = max(1, int(min(H, W) * heatmap_zero_border_frac))
        heatmap = heatmap.copy()
        heatmap[:t, :] = 0.0
        heatmap[-t:, :] = 0.0
        heatmap[:, :t] = 0.0
        heatmap[:, -t:] = 0.0
        hm_min, hm_max = heatmap.min(), heatmap.max()
        if hm_max - hm_min > 1e-8:
            heatmap = (heatmap - hm_min) / (hm_max - hm_min)

    # Optional spatial smoothing to make highlighted regions appear broader
    if isinstance(heatmap_blur_kernel, int) and heatmap_blur_kernel > 1:
        k = heatmap_blur_kernel if heatmap_blur_kernel % 2 == 1 else heatmap_blur_kernel + 1
        t = torch.from_numpy(heatmap).float().unsqueeze(0).unsqueeze(0)
        t = F.avg_pool2d(t, kernel_size=k, stride=1, padding=k // 2)
        heatmap = t.squeeze().numpy()
        hm_min, hm_max = heatmap.min(), heatmap.max()
        if hm_max - hm_min > 1e-8:
            heatmap = (heatmap - hm_min) / (hm_max - hm_min)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    if figure_title:
        fig.suptitle(figure_title, fontsize=11, y=1.02)

    # Original image (denormalized)
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Heatmap with colorbar (min-max normalized for consistent jet mapping)
    im = axes[1].imshow(heatmap, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM (this target class)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1])

    # Overlay: full-strength base so anatomy matches/clarifies vs Original; heatmap on top
    if ghosted_overlay:
        gray = np.mean(image, axis=-1) if image.ndim == 3 else image
        base = np.stack([gray, gray, gray], axis=-1)
    else:
        base = image
    base = np.clip(base.astype(np.float64), 0, 1)
    axes[2].imshow(base)
    axes[2].imshow(heatmap, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
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

