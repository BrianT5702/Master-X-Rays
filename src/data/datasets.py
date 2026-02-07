"""Dataset and dataloader implementations for NIH ChestX-ray14."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from src.data.roi_extraction import extract_lung_roi


class AddGaussianNoise:
    """Add Gaussian noise to tensor. Pickle-friendly for Windows multiprocessing."""
    
    def __init__(self, std: float = 0.01):
        self.std = std
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std, 0, 1)


class NIHChestXRayDataset(Dataset):
    """Dataset for NIH ChestX-ray14 with preprocessing and augmentation."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        images_root: Path,
        label_columns: List[str],
        image_size: Tuple[int, int] = (320, 320),
        augment: bool = False,
        augmentation_config: Optional[Dict] = None,
        use_roi_extraction: bool = True,
        cache_dir: Optional[Path] = None,
    ) -> None:
        """
        Args:
            metadata: DataFrame with 'Image Index' and 'Finding Labels' columns
            images_root: Root directory containing image subdirectories
            label_columns: List of disease labels to extract (e.g., ['Nodule', 'Fibrosis'])
            image_size: Target image size (height, width) - default 320x320 for high resolution
            augment: Whether to apply data augmentation
            augmentation_config: Dict with augmentation parameters
            use_roi_extraction: Whether to extract lung ROI to remove artifacts
        """
        self.metadata = metadata.reset_index(drop=True)
        self.images_root = Path(images_root)
        self.label_columns = label_columns
        self.image_size = image_size
        self.augment = augment
        self.augmentation_config = augmentation_config or {}
        self.use_roi_extraction = use_roi_extraction
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Create binary labels
        for label in label_columns:
            self.metadata[label] = (
                self.metadata["Finding Labels"]
                .str.contains(label, regex=False, na=False)
                .astype(int)
            )

        # X-ray appropriate normalization (gentler than ImageNet)
        # X-rays are grayscale medical images, not natural photos
        # Using simple centering normalization: (pixel/255 - 0.5) / 0.5 maps to [-1, 1]
        # This preserves natural X-ray appearance without aggressive contrast changes
        self.mean = [0.5, 0.5, 0.5]  # Center at 0.5 (mid-gray for X-rays)
        self.std = [0.5, 0.5, 0.5]   # Scale to [-1, 1] range

        # Build transform pipeline
        self.transform = self._build_transform()

    def _build_transform(self) -> transforms.Compose:
        """Build image transformation pipeline."""
        transform_list = []

        # Data augmentation (only for training)
        if self.augment:
            # Random Resized Crop (scale 0.8-1.0) - anatomically plausible augmentation
            # This forces the model to recognize pathology based on texture rather than position
            # Using BILINEAR instead of BICUBIC for faster processing (minimal quality difference)
            # IMPORTANT: Use center crop more often to prevent edge focus
            # Scale 0.9-1.0 reduces edge artifacts, ratio 0.95-1.05 keeps it more centered
            transform_list.append(
                transforms.RandomResizedCrop(
                    size=self.image_size,
                    scale=(0.9, 1.0),  # Reduced from 0.8-1.0 to 0.9-1.0 to minimize edge cropping
                    ratio=(0.95, 1.05),  # Tighter ratio to keep focus on center (was 0.9-1.1)
                    interpolation=transforms.InterpolationMode.BILINEAR,  # Faster than BICUBIC
                )
            )
            
            # Horizontal flip
            if self.augmentation_config.get("horizontal_flip_prob", 0.5) > 0:
                transform_list.append(
                    transforms.RandomHorizontalFlip(
                        p=self.augmentation_config.get("horizontal_flip_prob", 0.5)
                    )
                )

            # Rotation (limited to ±10° for anatomical plausibility)
            rotation_degrees = self.augmentation_config.get("rotation_degrees", 10.0)
            # Cap at 10 degrees as per thesis
            rotation_degrees = min(rotation_degrees, 10.0)
            if rotation_degrees > 0:
                transform_list.append(
                    transforms.RandomRotation(degrees=(-rotation_degrees, rotation_degrees))
                )
        else:
            # For validation/test: just resize to target size
            # Using BILINEAR for faster processing
            transform_list.append(transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR))

        # Convert to tensor and normalize
        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

        # Add random noise if specified
        if self.augment and self.augmentation_config.get("random_noise_std", 0) > 0:
            noise_std = self.augmentation_config.get("random_noise_std", 0.01)
            transform_list.append(AddGaussianNoise(std=noise_std))

        return transforms.Compose(transform_list)

    def _load_image(self, image_path: Path) -> Image.Image:
        """Load and preprocess image. Optimized for speed."""
        try:
            # Use lazy loading - don't load full image until needed
            # PIL's open() is lazy, but convert() forces full load
            # For grayscale X-rays, we can optimize further
            with Image.open(image_path) as img:
                # Convert to RGB in one step (faster than open then convert)
                if img.mode != 'RGB':
                    image = img.convert("RGB")
                else:
                    image = img.copy()  # Copy if already RGB
            return image
        except Exception as e:
            raise ValueError(f"Failed to load image from {image_path}: {e}")

    def _find_image_path(self, image_index: str) -> Path:
        """Find image path in subdirectories. Optimized for speed."""
        # NIH dataset structure: images_001/images/00000001_000.png
        # Try most common pattern first (usually images_*/images/)
        # Extract prefix from image_index (e.g., "00000001_000.png" -> "00000001")
        prefix = image_index.split('_')[0] if '_' in image_index else image_index.split('.')[0]
        # Most images are in images_001, images_002, etc. Try first 10 directories
        for i in range(1, 11):
            candidate = self.images_root / f"images_{i:03d}" / "images" / image_index
            if candidate.exists():
                return candidate
        
        # Fallback: search all patterns (slower)
        patterns = [
            f"images_*/images/{image_index}",
            f"images/{image_index}",
            image_index,
        ]

        for pattern in patterns:
            matches = list(self.images_root.glob(pattern))
            if matches:
                return matches[0]

        # If not found, try direct path
        direct_path = self.images_root / image_index
        if direct_path.exists():
            return direct_path

        raise FileNotFoundError(f"Image not found: {image_index}")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        row = self.metadata.iloc[idx]
        image_index = row["Image Index"]

        # Load from cache if available, otherwise load original
        if self.cache_dir:
            # Try both .png and .jpg (for backward compatibility)
            cache_path_png = self.cache_dir / "processed_images" / image_index
            cache_path_jpg = cache_path_png.with_suffix('.jpg')
            cache_path = cache_path_jpg if cache_path_jpg.exists() else cache_path_png
            
            if cache_path.exists():
                # Load preprocessed image (already ROI extracted and resized)
                # JPEG is faster to load than PNG
                image = Image.open(cache_path).convert("RGB")
            else:
                # Fallback to original image processing
                image_path = self._find_image_path(image_index)
                image = self._load_image(image_path)
                if self.use_roi_extraction:
                    image, _ = extract_lung_roi(image)
                # Resize if not already cached at correct size
                if image.size != self.image_size:
                    image = image.resize(self.image_size, Image.Resampling.BILINEAR)
        else:
            # Original processing pipeline
            image_path = self._find_image_path(image_index)
            image = self._load_image(image_path)
            
            # ROI Extraction: Remove artifacts and extract lung fields
            if self.use_roi_extraction:
                image, _ = extract_lung_roi(image)
        
        # Apply transforms (augmentation, normalization)
        # For cached images: skip resize (already 320x320), but still apply augmentation if training
        if self.cache_dir:
            if self.augment:
                # Training: apply augmentation to cached image
                # Use lightweight augmentation (no resize needed)
                transform_list = []
                if self.augmentation_config.get("horizontal_flip_prob", 0) > 0:
                    transform_list.append(transforms.RandomHorizontalFlip(p=self.augmentation_config["horizontal_flip_prob"]))
                # Rotation (limited to ±10° for anatomical plausibility, consistent with main pipeline)
                rotation_degrees = self.augmentation_config.get("rotation_degrees", 10.0)
                rotation_degrees = min(rotation_degrees, 10.0)  # Cap at 10 degrees as per thesis
                if rotation_degrees > 0:
                    transform_list.append(transforms.RandomRotation(degrees=(-rotation_degrees, rotation_degrees)))
                transform_list.extend([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std),
                ])
                image_tensor = transforms.Compose(transform_list)(image)
            else:
                # Validation/Test: just normalize (no augmentation)
                transform_list = [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std),
                ]
                image_tensor = transforms.Compose(transform_list)(image)
        else:
            # Non-cached: full transform pipeline
            image_tensor = self.transform(image)

        # Extract labels
        labels = torch.tensor(
            [row[label] for label in self.label_columns], dtype=torch.float32
        )

        return {"image": image_tensor, "labels": labels}


def _split_by_patients(
    metadata: pd.DataFrame,
    splits: Tuple[float, float, float],
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset by patients to prevent data leakage.
    
    Args:
        metadata: DataFrame with 'Patient ID' column
        splits: (train, val, test) split ratios
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_ratio, val_ratio, test_ratio = splits
    
    # Get unique patients
    unique_patients = metadata["Patient ID"].unique()
    np.random.seed(seed)
    np.random.shuffle(unique_patients)
    
    # Calculate split indices
    n_patients = len(unique_patients)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)
    
    # Split patients
    train_patients = unique_patients[:n_train]
    val_patients = unique_patients[n_train : n_train + n_val]
    test_patients = unique_patients[n_train + n_val :]
    
    # Get all images for each patient group
    train_df = metadata[metadata["Patient ID"].isin(train_patients)].copy()
    val_df = metadata[metadata["Patient ID"].isin(val_patients)].copy()
    test_df = metadata[metadata["Patient ID"].isin(test_patients)].copy()
    
    return train_df, val_df, test_df


def create_dataloaders(
    csv_path: Path,
    images_root: Path,
    label_columns: List[str],
    image_size: Tuple[int, int] = (320, 320),
    batch_size: int = 32,
    splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    num_workers: int = 4,
    seed: int = 42,
    augmentation_config: Optional[Dict] = None,
    use_weighted_sampling: bool = False,
    patient_split: bool = False,
    use_roi_extraction: bool = True,
    cache_dir: Optional[Path] = None,
) -> Dict[str, DataLoader]:
    """
    Create train/val/test dataloaders.

    Args:
        csv_path: Path to NIH metadata CSV
        images_root: Root directory containing images
        label_columns: List of disease labels
        image_size: Target image size
        batch_size: Batch size
        splits: (train, val, test) split ratios
        num_workers: Number of data loading workers
        seed: Random seed
        augmentation_config: Augmentation parameters
        use_weighted_sampling: Whether to use weighted sampling for training
        patient_split: If True, split by patients to prevent data leakage. 
                      Requires 'Patient ID' column in CSV.

    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    # Load metadata
    metadata = pd.read_csv(csv_path)

    # Split data
    train_ratio, val_ratio, test_ratio = splits
    assert abs(sum(splits) - 1.0) < 1e-6, "Splits must sum to 1.0"

    # Choose splitting strategy
    if patient_split:
        # Check if Patient ID column exists
        if "Patient ID" not in metadata.columns:
            print("⚠️  Warning: patient_split=True but 'Patient ID' column not found in CSV.")
            print("   Falling back to random image-level splitting.")
            patient_split = False
        
    if patient_split:
        # Split by patients (prevents data leakage)
        train_df, val_df, test_df = _split_by_patients(metadata, splits, seed)
        print(f"✅ Patient-level splitting:")
        print(f"   Train: {len(train_df):,} images from {train_df['Patient ID'].nunique():,} patients")
        print(f"   Val:   {len(val_df):,} images from {val_df['Patient ID'].nunique():,} patients")
        print(f"   Test:  {len(test_df):,} images from {test_df['Patient ID'].nunique():,} patients")
    else:
        # Random image-level splitting (original method)
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            metadata, test_size=(1 - train_ratio), random_state=seed, shuffle=True
        )

        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df, test_size=(1 - val_size), random_state=seed, shuffle=True
        )
        print(f"📊 Random image-level splitting:")
        print(f"   Train: {len(train_df):,} images")
        print(f"   Val:   {len(val_df):,} images")
        print(f"   Test:  {len(test_df):,} images")

    # Create datasets
    train_dataset = NIHChestXRayDataset(
        metadata=train_df,
        images_root=images_root,
        label_columns=label_columns,
        image_size=image_size,
        augment=True,
        augmentation_config=augmentation_config,
        use_roi_extraction=use_roi_extraction if cache_dir is None else False,  # Disable if using cache
        cache_dir=cache_dir,
    )

    val_dataset = NIHChestXRayDataset(
        metadata=val_df,
        images_root=images_root,
        label_columns=label_columns,
        image_size=image_size,
        augment=False,
        use_roi_extraction=use_roi_extraction if cache_dir is None else False,
        cache_dir=cache_dir,
    )

    test_dataset = NIHChestXRayDataset(
        metadata=test_df,
        images_root=images_root,
        label_columns=label_columns,
        image_size=image_size,
        augment=False,
        use_roi_extraction=use_roi_extraction if cache_dir is None else False,
        cache_dir=cache_dir,
    )
    
    if cache_dir:
        print(f"✅ Using cached images from: {cache_dir}")

    # Create weighted sampler for training if requested
    train_sampler = None
    if use_weighted_sampling:
        # Calculate sample weights based on class distribution
        sample_weights = _calculate_sample_weights(train_df, label_columns)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

    # Windows multiprocessing fix: use persistent_workers and reduce workers if needed
    # ROI extraction is CPU-intensive, so we need to balance workers vs overhead
    import platform
    is_windows = platform.system() == "Windows"
    
    # Optimize for Windows: use 1 worker to avoid multiprocessing overhead but still get async loading
    if num_workers == 0:
        effective_num_workers = 0
        use_persistent_workers = False
        prefetch_factor = None
        print("ℹ️  Using single-threaded data loading (num_workers=0)")
    else:
        # For Windows: Use spawn method and allow more workers
        if is_windows and num_workers > 0:
            import multiprocessing
            try:
                multiprocessing.set_start_method('spawn', force=True)
            except RuntimeError:
                pass  # Already set
        
        # Windows can handle more workers with spawn method
        # 3-4 workers is typically optimal for Windows (spawn overhead)
        # Memory at 92% - can try 4 but monitor closely
        effective_num_workers = min(num_workers, 4) if is_windows else min(num_workers, 6)
        use_persistent_workers = effective_num_workers > 0
        # Moderate prefetch - too high causes memory overhead without benefit
        prefetch_factor = 4 if effective_num_workers > 0 else None
        if is_windows:
            print(f"ℹ️  Windows: Using {effective_num_workers} worker(s) with prefetch_factor={prefetch_factor}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=effective_num_workers,
        pin_memory=True,
        persistent_workers=use_persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=True,
        persistent_workers=use_persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=True,
        persistent_workers=use_persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    
    if is_windows and num_workers > 4:
        print(f"⚠️  Windows detected: Reduced num_workers from {num_workers} to {effective_num_workers} to avoid multiprocessing issues")

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


def _calculate_sample_weights(
    metadata: pd.DataFrame, label_columns: List[str]
) -> torch.Tensor:
    """Calculate sample weights for weighted sampling."""
    # Create binary labels
    for label in label_columns:
        metadata[label] = (
            metadata["Finding Labels"].str.contains(label, regex=False, na=False).astype(int)
        )

    # Calculate weights: inverse of class frequency
    total_samples = len(metadata)
    weights = []

    for idx in range(len(metadata)):
        row = metadata.iloc[idx]
        # Get labels for this sample
        labels = [row[label] for label in label_columns]

        # Calculate weight: higher weight for rare class samples
        weight = 1.0
        for label in label_columns:
            class_count = metadata[label].sum()
            if row[label] == 1:  # Positive sample
                # Weight inversely proportional to class frequency
                weight *= total_samples / (class_count + 1)

        weights.append(weight)

    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum() * len(weights)

    return torch.tensor(weights, dtype=torch.float32)
