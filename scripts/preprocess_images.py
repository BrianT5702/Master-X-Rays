"""Pre-process and cache ROI-extracted images for faster training.

This script processes all images once, extracts ROI, and saves them to a cache directory.
Training can then load from cache instead of processing on-the-fly, eliminating bottlenecks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import os
from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageOps, ImageOps

# Fix Windows console encoding for emoji
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.roi_extraction import extract_lung_roi


def preprocess_images(
    csv_path: Path,
    images_root: Path,
    cache_dir: Path,
    image_size: tuple[int, int] = (320, 320),
    use_roi_extraction: bool = True,
    label_columns: list[str] | None = None,
) -> None:
    """
    Pre-process all images: extract ROI, resize, and save to cache.
    
    Args:
        csv_path: Path to NIH metadata CSV
        images_root: Root directory containing original images
        cache_dir: Directory to save preprocessed images
        image_size: Target image size (height, width)
        use_roi_extraction: Whether to extract ROI
        label_columns: List of disease labels (for filtering, optional)
    """
    # Load metadata
    print("[INFO] Loading metadata...")
    metadata = pd.read_csv(csv_path)
    
    # Filter for frontal views only (PA/AP) - exclude lateral views
    # Lateral views have completely different anatomy and should not be used
    if "View Position" in metadata.columns:
        print("[INFO] Filtering for PA/AP views only (excluding lateral views)...")
        initial_count = len(metadata)
        # Only keep Frontal views (PA or AP)
        metadata = metadata[metadata["View Position"].isin(["PA", "AP"])]
        print(f"   Removed {initial_count - len(metadata):,} lateral/oblique views")
        print(f"   Remaining images: {len(metadata):,}")
    elif "View" in metadata.columns:
        # Alternative column name
        print("[INFO] Filtering for frontal views only...")
        initial_count = len(metadata)
        metadata = metadata[metadata["View"].str.contains("PA|AP", case=False, na=False, regex=True)]
        print(f"   Removed {initial_count - len(metadata):,} lateral/oblique views")
        print(f"   Remaining images: {len(metadata):,}")
    
    # For multi-label classification: Include ALL images
    # Positive cases: Images with target labels (Nodule/Fibrosis)
    # Negative cases: Images WITHOUT target labels (includes "No Finding" AND other diseases)
    # This is correct for multi-label classification - other diseases are valid negatives
    if label_columns:
        print(f"[INFO] Processing all images for multi-label classification")
        print(f"   Target labels: {label_columns}")
        print(f"   Positive cases: Images WITH {label_columns}")
        print(f"   Negative cases: Images WITHOUT {label_columns} (includes other diseases)")
        print(f"   Total images to process: {len(metadata):,}")
        # Create binary labels for all images (will be used during training)
        for label in label_columns:
            metadata[label] = (
                metadata["Finding Labels"]
                .str.contains(label, regex=False, na=False)
                .astype(int)
            )
        # Count statistics
        has_target = metadata[label_columns].any(axis=1)
        positive_count = has_target.sum()
        negative_count = len(metadata) - positive_count
        print(f"   Positive cases (with target labels): {positive_count:,}")
        print(f"   Negative cases (without target labels): {negative_count:,}")
        # Keep ALL images - no filtering
        # metadata = metadata.copy()  # Already have all images, no need to filter
    
    # Create cache directory structure
    cache_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = cache_dir / "processed_images"
    processed_dir.mkdir(exist_ok=True)
    
    print("[INFO] Using JPEG format for cache (faster loading, ~3-5x speedup vs PNG)")
    
    # Save metadata to cache
    metadata_cache_path = cache_dir / "metadata.csv"
    metadata.to_csv(metadata_cache_path, index=False)
    print(f"[INFO] Saved metadata to {metadata_cache_path}")
    
    # Process images
    print(f"[INFO] Processing {len(metadata):,} images...")
    print(f"   Cache directory: {processed_dir}")
    print(f"   Format: JPEG (quality=95) - 3-5x faster loading than PNG")
    print(f"   ROI extraction: {'Enabled' if use_roi_extraction else 'Disabled'}")
    print(f"   Target size: {image_size}")
    
    failed = []
    success = 0
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing"):
        image_index = row["Image Index"]
        cache_path = processed_dir / image_index
        
        # Skip if already processed (check both .png and .jpg for backward compatibility)
        cache_path_jpg = cache_path.with_suffix('.jpg')
        cache_path_png = cache_path.with_suffix('.png')
        if cache_path_jpg.exists() or cache_path_png.exists():
            success += 1
            continue
        
        try:
            # Find original image
            image_path = _find_image_path(images_root, image_index)
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Extract ROI if enabled
            if use_roi_extraction:
                image, _ = extract_lung_roi(image)
            
            # Resize to target size
            image = image.resize(image_size, Image.Resampling.BILINEAR)
            
            # Save to cache as JPEG (much faster loading than PNG, ~3-5x speedup)
            # Quality 95 is visually identical to PNG but much smaller and faster
            cache_path = cache_path.with_suffix('.jpg')  # Change extension to .jpg
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(cache_path, format="JPEG", quality=95, optimize=False)
            
            success += 1
            
        except Exception as e:
            failed.append((image_index, str(e)))
            if len(failed) <= 5:  # Print first 5 errors
                print(f"[WARNING] Failed to process {image_index}: {e}")
    
    print(f"\n[SUCCESS] Preprocessing complete!")
    print(f"   Success: {success:,} images")
    print(f"   Failed: {len(failed):,} images")
    if failed:
        print(f"   Failed images saved to: {cache_dir / 'failed_images.txt'}")
        with open(cache_dir / "failed_images.txt", "w", encoding='utf-8') as f:
            for img, err in failed:
                f.write(f"{img}: {err}\n")
    
    print(f"\n[INFO] Next steps:")
    print(f"   1. Config already updated: cache_dir = '{cache_dir}'")
    print(f"   2. ROI extraction auto-disabled when using cache")
    print(f"   3. Run training - it will be much faster!")


def _find_image_path(images_root: Path, image_index: str) -> Path:
    """Find image path in subdirectories. Optimized version."""
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
    parser = argparse.ArgumentParser(description="Pre-process images for faster training.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to NIH metadata CSV")
    parser.add_argument("--images-root", type=Path, required=True, help="Root directory containing images")
    parser.add_argument("--cache-dir", type=Path, default=Path("datasets/cache"), help="Cache directory for processed images")
    parser.add_argument("--image-size", type=int, nargs=2, default=[320, 320], help="Target image size (height width)")
    parser.add_argument("--no-roi", action="store_true", help="Disable ROI extraction")
    parser.add_argument("--labels", nargs="+", default=["Nodule", "Fibrosis"], help="Filter for specific labels")
    
    args = parser.parse_args()
    
    preprocess_images(
        csv_path=args.csv,
        images_root=args.images_root,
        cache_dir=args.cache_dir,
        image_size=tuple(args.image_size),
        use_roi_extraction=not args.no_roi,
        label_columns=args.labels if args.labels else None,
    )


if __name__ == "__main__":
    main()

