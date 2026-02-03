"""Visualize ROI extraction results to verify quality."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np

from src.data.roi_extraction import extract_lung_roi


def visualize_roi_extraction(
    csv_path: Path,
    images_root: Path,
    num_samples: int = 12,
    save_dir: Path = Path("roi_visualizations"),
) -> None:
    """
    Visualize ROI extraction on random samples from the dataset.
    
    Args:
        csv_path: Path to NIH metadata CSV
        images_root: Root directory containing images
        num_samples: Number of random samples to visualize
        save_dir: Directory to save visualization images
    """
    # Load metadata
    print(f"[INFO] Loading metadata from {csv_path}...")
    metadata = pd.read_csv(csv_path)
    
    # Filter for frontal views if column exists
    if "View Position" in metadata.columns:
        metadata = metadata[metadata["View Position"].isin(["PA", "AP"])]
        print(f"[INFO] Filtered to {len(metadata):,} frontal views")
    
    # Random sample
    sample_df = metadata.sample(n=min(num_samples, len(metadata)), random_state=42)
    
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Processing {len(sample_df)} samples...")
    
    # Create figure with subplots
    cols = 4
    rows = (len(sample_df) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    axes = axes.flatten()
    
    success_count = 0
    fallback_count = 0
    
    for idx, (_, row) in enumerate(sample_df.iterrows()):
        image_index = row["Image Index"]
        finding_labels = row.get("Finding Labels", "Unknown")
        
        try:
            # Find image
            image_path = _find_image_path(images_root, image_index)
            original = Image.open(image_path).convert("RGB")
            
            # Extract ROI
            cropped, bbox = extract_lung_roi(original.copy())
            
            # Check if fallback was used (bbox covers most of image)
            crop_ratio = (bbox[2] * bbox[3]) / (original.size[0] * original.size[1])
            is_fallback = crop_ratio > 0.7  # If >70% of image, likely fallback
            
            if is_fallback:
                fallback_count += 1
            else:
                success_count += 1
            
            # Create visualization
            ax = axes[idx]
            
            # Draw bounding box on original
            original_with_bbox = original.copy()
            draw = ImageDraw.Draw(original_with_bbox)
            x, y, w, h = bbox
            draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
            
            # Combine original (with bbox) and cropped side by side
            # Resize for display
            display_size = 320
            orig_display = original_with_bbox.resize((display_size, display_size), Image.Resampling.BILINEAR)
            crop_display = cropped.resize((display_size, display_size), Image.Resampling.BILINEAR)
            
            # Combine horizontally
            combined = Image.new("RGB", (display_size * 2, display_size))
            combined.paste(orig_display, (0, 0))
            combined.paste(crop_display, (display_size, 0))
            
            # Add text overlay
            draw_combined = ImageDraw.Draw(combined)
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Label
            label_text = finding_labels[:30] + "..." if len(finding_labels) > 30 else finding_labels
            status = "FALLBACK" if is_fallback else "ROI"
            draw_combined.text((10, 10), f"{image_index}\n{status}", fill="yellow", font=font)
            draw_combined.text((10, display_size - 40), label_text, fill="yellow", font=font)
            
            ax.imshow(combined)
            ax.axis("off")
            ax.set_title(f"{image_index}\n{'FALLBACK' if is_fallback else 'ROI'}", 
                        fontsize=10, color="red" if is_fallback else "green")
            
        except Exception as e:
            ax = axes[idx]
            ax.text(0.5, 0.5, f"Error:\n{str(e)[:50]}", 
                   ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            print(f"[WARNING] Failed to process {image_index}: {e}")
    
    # Hide unused subplots
    for idx in range(len(sample_df), len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    save_path = save_dir / "roi_extraction_samples.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[SUCCESS] Visualization saved to: {save_path}")
    print(f"   Successful ROI extractions: {success_count}")
    print(f"   Fallback crops: {fallback_count}")
    print(f"   Success rate: {100*success_count/(success_count+fallback_count):.1f}%")
    
    # Also save individual images for closer inspection
    print(f"\n[INFO] Saving individual samples to {save_dir / 'individual_samples'}...")
    individual_dir = save_dir / "individual_samples"
    individual_dir.mkdir(exist_ok=True)
    
    for idx, (_, row) in enumerate(sample_df.iterrows()):
        image_index = row["Image Index"]
        try:
            image_path = _find_image_path(images_root, image_index)
            original = Image.open(image_path).convert("RGB")
            cropped, bbox = extract_lung_roi(original.copy())
            
            # Save side-by-side comparison
            w, h = original.size
            combined = Image.new("RGB", (w * 2, h))
            combined.paste(original, (0, 0))
            combined.paste(cropped, (w, 0))
            
            # Draw bbox on original
            draw = ImageDraw.Draw(combined)
            x, y, w_box, h_box = bbox
            draw.rectangle([x, y, x + w_box, y + h_box], outline="red", width=5)
            
            combined.save(individual_dir / f"{image_index}_roi.png")
        except:
            pass
    
    print(f"[SUCCESS] Individual samples saved!")


def _find_image_path(images_root: Path, image_index: str) -> Path:
    """Find image path in subdirectories."""
    # Try most common pattern first
    for i in range(1, 11):
        candidate = images_root / f"images_{i:03d}" / "images" / image_index
        if candidate.exists():
            return candidate
    
    # Fallback: search all patterns
    patterns = [
        f"images_*/images/{image_index}",
        f"images/{image_index}",
        image_index,
    ]
    
    for pattern in patterns:
        matches = list(images_root.glob(pattern))
        if matches:
            return matches[0]
    
    # Direct path
    direct_path = images_root / image_index
    if direct_path.exists():
        return direct_path
    
    raise FileNotFoundError(f"Image not found: {image_index}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize ROI extraction results.")
    parser.add_argument("--csv", type=Path, 
                       default=Path("datasets/NIH Chest X-Rays Master Datasets/archive/Data_Entry_2017.csv"),
                       help="Path to NIH metadata CSV")
    parser.add_argument("--images-root", type=Path,
                       default=Path("datasets/NIH Chest X-Rays Master Datasets/archive"),
                       help="Root directory containing images")
    parser.add_argument("--num-samples", type=int, default=12, help="Number of samples to visualize")
    parser.add_argument("--save-dir", type=Path, default=Path("roi_visualizations"), 
                       help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    visualize_roi_extraction(
        csv_path=args.csv,
        images_root=args.images_root,
        num_samples=args.num_samples,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()











