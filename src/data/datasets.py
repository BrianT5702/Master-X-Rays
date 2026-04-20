"""Dataset and dataloader implementations for NIH ChestX-ray14."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from src.data.roi_extraction import extract_lung_roi

# Torchvision ImageNet-pretrained CNNs (use ImageNet mean/std when normalization is auto)
_IMAGENET_NORM_BACKBONES = frozenset(
    {"efficientnet_b0", "efficientnet_b2", "resnet50", "swin_t", "dense_swin_hybrid"}
)


def resolve_data_normalization(explicit: Optional[str], backbone: str) -> str:
    """Resolve ``data.normalization`` together with ``model.backbone``.

    - ``None`` or ``"auto"``: DenseNet121 → ``legacy`` (this repo's tuned pipeline); EfficientNet,
      ResNet50, Swin-T, ``dense_swin_hybrid`` → ``imagenet`` (both branches ImageNet-pretrained).
    - ``"legacy"`` / ``"imagenet"``: force that mode (reproducibility or experiments).

    If ``use_normalization`` is false, this value is ignored at transform time.
    """
    if explicit is not None:
        mode = str(explicit).strip().lower()
        if mode == "auto":
            pass
        elif mode in ("legacy", "imagenet"):
            return mode
        raise ValueError(
            "data.normalization must be 'legacy', 'imagenet', or 'auto', "
            f"got {explicit!r}"
        )
    bb = str(backbone or "").strip().lower().replace("-", "_")
    if bb in _IMAGENET_NORM_BACKBONES:
        return "imagenet"
    return "legacy"


class AddGaussianNoise:
    """Add Gaussian noise on [0,1] tensors (apply before Normalize). Pickle-friendly."""

    def __init__(self, std: float = 0.01):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std, 0.0, 1.0)


class RandomLowerArtifactErase:
    """
    Randomly erase a rectangular region near the lower corners of the image.
    
    Motivation:
    - Grad-CAM shows the current model relies heavily on a small artifact region
      near the diaphragm / lower-right corner.
    - This transform makes that shortcut unreliable during training so the model
      is forced to learn lung texture instead of border artifacts.
    """

    def __init__(self, p: float = 0.5, height_ratio: float = 0.20, width_ratio: float = 0.25) -> None:
        self.p = p
        self.height_ratio = height_ratio
        self.width_ratio = width_ratio

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        w, h = img.size

        # Randomly choose left or right lower corner to erase
        erase_right = random.random() < 0.5

        bw = int(w * self.width_ratio)
        bh = int(h * self.height_ratio)
        x1 = w - bw if erase_right else 0
        y1 = h - bh
        x2 = x1 + bw
        y2 = h

        # Draw a dark rectangle (near background) to disrupt shortcut cues
        img = img.copy()
        draw = ImageDraw.Draw(img)
        draw.rectangle([x1, y1, x2, y2], fill=0)
        return img


class RandomLateralEdgeErase:
    """
    Randomly erase thin vertical strips along the left and/or right image edges.

    Motivation:
    - Grad-CAM shows the model often focuses on lateral borders (chest wall / image edges)
      instead of lung parenchyma. This makes that shortcut unreliable during training.
    """

    def __init__(self, p: float = 0.5, width_ratio: float = 0.05) -> None:
        self.p = p
        self.width_ratio = width_ratio

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        w, h = img.size
        strip_w = max(1, int(w * self.width_ratio))
        img = img.copy()
        draw = ImageDraw.Draw(img)

        if random.random() < 0.5:
            draw.rectangle([0, 0, strip_w, h], fill=0)
        if random.random() < 0.5:
            draw.rectangle([w - strip_w, 0, w, h], fill=0)
        return img


class RandomTopEdgeErase:
    """
    Randomly erase a horizontal strip along the top of the image (neck/clavicle/supraclavicular).

    Motivation:
    - Grad-CAM shows the model often focuses on upper chest/neck/clavicle instead of lung.
    - Making this region unreliable during training forces attention on lung parenchyma.
    """

    def __init__(self, p: float = 0.5, height_ratio: float = 0.18) -> None:
        self.p = p
        self.height_ratio = height_ratio

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        w, h = img.size
        strip_h = max(1, int(h * self.height_ratio))
        img = img.copy()
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, w, strip_h], fill=0)
        return img


class RandomBottomBandErase:
    """
    Erase a full-width horizontal strip at the bottom (diaphragm / upper abdomen / collimation).

    Grad-CAM often peaks here even when the task is lung parenchyma; corner-only erases miss a
    centered bottom band. Random full-width erasure makes that shortcut unreliable during training.
    """

    def __init__(
        self,
        p: float = 0.75,
        height_ratio_min: float = 0.08,
        height_ratio_max: float = 0.20,
    ) -> None:
        self.p = p
        self.height_ratio_min = height_ratio_min
        self.height_ratio_max = height_ratio_max

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        w, h = img.size
        hr = random.uniform(self.height_ratio_min, self.height_ratio_max)
        strip_h = max(1, int(h * hr))
        y1 = h - strip_h
        img = img.copy()
        ImageDraw.Draw(img).rectangle([0, y1, w, h], fill=0)
        return img


class RandomCornerErase:
    """
    Erase square patches at image corners (independently per corner).

    Motivation:
    - Grad-CAM often peaks on frame corners / shoulders / collimation outside lungs.
    - Corner erasure breaks reliance on fixed border pixels while preserving most of the field.
    """

    def __init__(
        self,
        p: float = 0.75,
        size_ratio: float = 0.12,
        per_corner_p: float = 0.45,
    ) -> None:
        self.p = p
        self.size_ratio = size_ratio
        self.per_corner_p = per_corner_p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        w, h = img.size
        side = max(4, int(min(w, h) * self.size_ratio))
        img = img.copy()
        draw = ImageDraw.Draw(img)
        boxes = [
            (0, 0, side, side),
            (w - side, 0, w, side),
            (0, h - side, side, h),
            (w - side, h - side, w, h),
        ]
        for box in boxes:
            if random.random() < self.per_corner_p:
                draw.rectangle(box, fill=0)
        return img


class RandomUpperDeviceErase:
    """
    Randomly erase bright, small artifact-like blobs in upper chest (e.g., ports/leads).

    Motivation:
    - New Grad-CAMs show repeated activation on bright device artifacts and corners.
    - This transform weakens that shortcut so the model must rely more on lung texture.
    """

    def __init__(
        self,
        p: float = 0.6,
        upper_ratio: float = 0.65,
        threshold: int = 230,
        min_area: int = 20,
        max_area_ratio: float = 0.01,
        max_components: int = 2,
    ) -> None:
        self.p = p
        self.upper_ratio = upper_ratio
        self.threshold = threshold
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.max_components = max_components

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        arr = np.array(img.convert("L"))
        h, w = arr.shape
        upper_h = max(1, int(h * self.upper_ratio))
        upper = arr[:upper_h, :]

        # Bright blobs candidate mask
        mask = (upper >= self.threshold).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return img

        max_area = int(h * w * self.max_area_ratio)
        candidates = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_area or area > max_area:
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            # favor approximately round-ish device blobs
            if bh > 0:
                aspect = bw / float(bh)
                if 0.4 <= aspect <= 2.5:
                    candidates.append((x, y, bw, bh))

        if not candidates:
            return img

        random.shuffle(candidates)
        out = img.copy()
        draw = ImageDraw.Draw(out)
        for x, y, bw, bh in candidates[: self.max_components]:
            draw.rectangle([x, y, x + bw, y + bh], fill=0)
        return out


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
        apply_roi_mask: bool = True,
        use_normalization: bool = False,
        cache_dir: Optional[Path] = None,
        normalization: str = "legacy",
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
            apply_roi_mask: If True, zero border-touching background (reduces corner shortcuts).
            normalization: "legacy" = (0.5,0.5) on [0,1]; "imagenet" = ImageNet mean/std (EfficientNet/ResNet).
        """
        self.metadata = metadata.reset_index(drop=True)
        self.images_root = Path(images_root)
        self.label_columns = label_columns
        self.image_size = image_size
        self.augment = augment
        self.augmentation_config = augmentation_config or {}
        self.use_roi_extraction = use_roi_extraction
        self.apply_roi_mask = apply_roi_mask
        self.use_normalization = use_normalization
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Create binary labels
        for label in label_columns:
            self.metadata[label] = (
                self.metadata["Finding Labels"]
                .str.contains(label, regex=False, na=False)
                .astype(int)
            )

        norm_mode = (normalization or "legacy").strip().lower()
        if norm_mode not in ("legacy", "imagenet"):
            raise ValueError(f"normalization must be 'legacy' or 'imagenet', got {normalization!r}")
        self._normalization_mode = norm_mode
        # Optional normalization: if use_normalization=False, model sees [0,1] (normal X-rays directly)
        if norm_mode == "imagenet":
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        else:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]

        # Build transform pipeline
        self.transform = self._build_transform()

    def _build_transform(self) -> transforms.Compose:
        """Build image transformation pipeline."""
        transform_list = []

        # Data augmentation (only for training)
        if self.augment:
            # Pre-v28 cache path: training images were already ROI-resized to ~320; applying
            # RandomResizedCrop on top destroyed fidelity and correlated with ~0.5 AUC. When
            # `legacy_cache_train_pipeline` is True and `cache_dir` is set, use Resize only.
            legacy_cache = (
                self.cache_dir is not None
                and bool(self.augmentation_config.get("legacy_cache_train_pipeline", False))
            )
            legacy_minimal = legacy_cache and bool(
                self.augmentation_config.get("legacy_cache_minimal_erases", False)
            )
            if legacy_cache:
                transform_list.append(
                    transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR)
                )
            else:
                transform_list.append(
                    transforms.RandomResizedCrop(
                        size=self.image_size,
                        scale=(0.9, 1.0),
                        ratio=(0.95, 1.05),
                        interpolation=transforms.InterpolationMode.BILINEAR,
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

            # Small translation — omitted in legacy cached pipeline (matches older high-AUC runs)
            if not legacy_cache:
                tr = float(self.augmentation_config.get("random_translate_frac", 0.0))
                if tr > 0:
                    transform_list.append(
                        transforms.RandomAffine(degrees=0, translate=(tr, tr), fill=0)
                    )

            # Stronger lower-corner erase to break shortcut (configurable)
            p_erase = self.augmentation_config.get("lower_artifact_erase_p", 0.85)
            h_ratio = self.augmentation_config.get("lower_artifact_erase_height_ratio", 0.28)
            w_ratio = self.augmentation_config.get("lower_artifact_erase_width_ratio", 0.32)
            transform_list.append(RandomLowerArtifactErase(p=p_erase, height_ratio=h_ratio, width_ratio=w_ratio))
            # Strip/device erases destroy a lot of pixels; v29 still ~0.53 val AUC with legacy+heavy strips.
            if not legacy_minimal:
                p_bb = self.augmentation_config.get("bottom_band_erase_p", 0.0)
                if p_bb > 0:
                    bb_min = self.augmentation_config.get("bottom_band_erase_h_min", 0.08)
                    bb_max = self.augmentation_config.get("bottom_band_erase_h_max", 0.20)
                    transform_list.append(
                        RandomBottomBandErase(p=p_bb, height_ratio_min=bb_min, height_ratio_max=bb_max)
                    )
                p_corner = self.augmentation_config.get("corner_erase_p", 0.0)
                if (not legacy_cache) and p_corner > 0:
                    cr = self.augmentation_config.get("corner_erase_size_ratio", 0.12)
                    cpc = self.augmentation_config.get("corner_erase_per_corner_p", 0.45)
                    transform_list.append(
                        RandomCornerErase(p=p_corner, size_ratio=cr, per_corner_p=cpc)
                    )
                p_lat = self.augmentation_config.get("lateral_edge_erase_p", 0.6)
                lat_w = self.augmentation_config.get("lateral_edge_width_ratio", 0.05)
                transform_list.append(RandomLateralEdgeErase(p=p_lat, width_ratio=lat_w))
                p_top = self.augmentation_config.get("top_edge_erase_p", 0.55)
                top_h = self.augmentation_config.get("top_edge_height_ratio", 0.18)
                transform_list.append(RandomTopEdgeErase(p=p_top, height_ratio=top_h))
                p_dev = self.augmentation_config.get("device_artifact_erase_p", 0.6)
                dev_upper = self.augmentation_config.get("device_artifact_upper_ratio", 0.65)
                dev_thr = self.augmentation_config.get("device_artifact_threshold", 230)
                dev_min_area = self.augmentation_config.get("device_artifact_min_area", 20)
                dev_max_area_ratio = self.augmentation_config.get("device_artifact_max_area_ratio", 0.01)
                dev_max_comp = self.augmentation_config.get("device_artifact_max_components", 2)
                transform_list.append(
                    RandomUpperDeviceErase(
                        p=p_dev,
                        upper_ratio=dev_upper,
                        threshold=dev_thr,
                        min_area=dev_min_area,
                        max_area_ratio=dev_max_area_ratio,
                        max_components=dev_max_comp,
                    )
                )
        else:
            # For validation/test: just resize to target size
            # Using BILINEAR for faster processing
            transform_list.append(transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR))

        # Convert to tensor; optional RandomErasing on [0,1] before Normalize (breaks local texture shortcuts)
        transform_list.append(transforms.ToTensor())
        if self.augment:
            re_p = float(self.augmentation_config.get("random_erasing_p", 0.0))
            if re_p > 0:
                re_scale = tuple(self.augmentation_config.get("random_erasing_scale", (0.02, 0.12)))
                re_ratio = tuple(self.augmentation_config.get("random_erasing_ratio", (0.3, 3.3)))
                transform_list.append(
                    transforms.RandomErasing(
                        p=re_p, scale=re_scale, ratio=re_ratio, value=0.0, inplace=False
                    )
                )
            # Noise on [0,1] before Normalize (avoids destroying negative ImageNet-normalized values)
            if self.augmentation_config.get("random_noise_std", 0) > 0:
                noise_std = self.augmentation_config.get("random_noise_std", 0.01)
                transform_list.append(AddGaussianNoise(std=noise_std))
        if self.use_normalization:
            transform_list.append(transforms.Normalize(mean=self.mean, std=self.std))

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
                    image, _ = extract_lung_roi(image, apply_mask=self.apply_roi_mask)
                # PIL uses (width, height); image_size is (height, width)
                target_wh = (self.image_size[1], self.image_size[0])
                if image.size != target_wh:
                    image = image.resize(target_wh, Image.Resampling.BILINEAR)
        else:
            # Original processing pipeline
            image_path = self._find_image_path(image_index)
            image = self._load_image(image_path)
            
            # ROI Extraction: Remove artifacts and extract lung fields; mask out border to reduce shortcut learning
            if self.use_roi_extraction:
                image, _ = extract_lung_roi(image, apply_mask=self.apply_roi_mask)

        # Match config resolution (cached files may be e.g. 320² while training at 384²)
        target_wh = (self.image_size[1], self.image_size[0])
        if image.size != target_wh:
            image = image.resize(target_wh, Image.Resampling.BILINEAR)

        # Same pipeline for cache and non-cache (all shortcut-focused augs + RandomErasing apply)
        image_tensor = self.transform(image)

        # Extract labels
        labels = torch.tensor(
            [row[label] for label in self.label_columns], dtype=torch.float32
        )

        # NIH image filename (for heatmap titles / debugging; collated as list of str in batches)
        image_index = str(row.get("Image Index", f"row_{idx}"))

        return {"image": image_tensor, "labels": labels, "image_index": image_index}


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
    weighted_sampling_dampen: float = 1.0,
    patient_split: bool = False,
    use_roi_extraction: bool = True,
    apply_roi_mask: bool = True,
    use_normalization: bool = False,
    cache_dir: Optional[Path] = None,
    normalization: str = "legacy",
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
        weighted_sampling_dampen: Exponent on weights (1.0 = full strength; <1 milder).
        patient_split: If True, split by patients to prevent data leakage.
                      Requires 'Patient ID' column in CSV.
        normalization: Passed to NIHChestXRayDataset ("legacy" default, or "imagenet").

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
            print("Warning: patient_split=True but 'Patient ID' column not found in CSV.")
            print("   Falling back to random image-level splitting.")
            patient_split = False
        
    if patient_split:
        # Split by patients (prevents data leakage)
        train_df, val_df, test_df = _split_by_patients(metadata, splits, seed)
        print("Patient-level splitting:")
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
        print("Random image-level splitting:")
        print(f"   Train: {len(train_df):,} images")
        print(f"   Val:   {len(val_df):,} images")
        print(f"   Test:  {len(test_df):,} images")

    # Report label distribution per split (so you can verify test has Nodule/Fibrosis positives)
    for split_name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        nodule_pos = df["Finding Labels"].str.contains("Nodule", regex=False, na=False).sum()
        fibrosis_pos = df["Finding Labels"].str.contains("Fibrosis", regex=False, na=False).sum()
        print(f"   {split_name} label counts: Nodule positives={nodule_pos:,}, Fibrosis positives={fibrosis_pos:,}")

    # Create datasets
    train_dataset = NIHChestXRayDataset(
        metadata=train_df,
        images_root=images_root,
        label_columns=label_columns,
        image_size=image_size,
        augment=True,
        augmentation_config=augmentation_config,
        use_roi_extraction=use_roi_extraction if cache_dir is None else False,  # Disable if using cache
        apply_roi_mask=apply_roi_mask,
        use_normalization=use_normalization,
        cache_dir=cache_dir,
        normalization=normalization,
    )

    val_dataset = NIHChestXRayDataset(
        metadata=val_df,
        images_root=images_root,
        label_columns=label_columns,
        image_size=image_size,
        augment=False,
        use_roi_extraction=use_roi_extraction if cache_dir is None else False,
        apply_roi_mask=apply_roi_mask,
        use_normalization=use_normalization,
        cache_dir=cache_dir,
        normalization=normalization,
    )

    test_dataset = NIHChestXRayDataset(
        metadata=test_df,
        images_root=images_root,
        label_columns=label_columns,
        image_size=image_size,
        augment=False,
        use_roi_extraction=use_roi_extraction if cache_dir is None else False,
        apply_roi_mask=apply_roi_mask,
        use_normalization=use_normalization,
        cache_dir=cache_dir,
        normalization=normalization,
    )

    if cache_dir:
        print(f"Using cached images from: {cache_dir}")
        if use_normalization and normalization.strip().lower() == "imagenet":
            print("   normalization=imagenet (ImageNet mean/std) — matches torchvision pretrained CNNs.")
        if augmentation_config.get("legacy_cache_train_pipeline"):
            extra = ""
            if augmentation_config.get("legacy_cache_minimal_erases"):
                extra = " + minimal erases (lower-corner only; no band/lateral/top/device)."
            print(
                "   legacy_cache_train_pipeline=True: train uses Resize (no RandomResizedCrop) "
                "and skips corner/translate augs — matches pre-unification cache behavior (~0.7 AUC runs)."
                + extra
            )

    # Create weighted sampler for training if requested
    train_sampler = None
    if use_weighted_sampling:
        if abs(float(weighted_sampling_dampen) - 1.0) > 1e-6:
            print(
                f"WeightedRandomSampler: dampen={float(weighted_sampling_dampen):.3f} "
                "(<1.0 = milder positive oversampling than full inverse-frequency weights)"
            )
        # Calculate sample weights based on class distribution
        sample_weights = _calculate_sample_weights(
            train_df, label_columns, dampen=float(weighted_sampling_dampen)
        )
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
        print("Using single-threaded data loading (num_workers=0)")
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
            print(f"Windows: Using {effective_num_workers} worker(s) with prefetch_factor={prefetch_factor}")

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
        print(f"Windows: Reduced num_workers from {num_workers} to {effective_num_workers} to avoid multiprocessing issues")

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


def _calculate_sample_weights(
    metadata: pd.DataFrame,
    label_columns: List[str],
    dampen: float = 1.0,
) -> torch.Tensor:
    """Calculate sample weights for weighted sampling (vectorized).

    The previous row-by-row iloc loop took minutes on ~90k rows and looked hung.

    dampen: raise each weight to this power (after initial normalize), then renormalize.
        Use 0.65–0.85 to oversample positives less aggressively than dampen=1.0, which
        often preserves AUROC better than full push_metrics-style weights.
    """
    n = int(len(metadata))
    if n == 0:
        return torch.zeros(0, dtype=torch.float32)

    flags = np.stack(
        [
            metadata["Finding Labels"]
            .str.contains(lbl, regex=False, na=False)
            .to_numpy(dtype=np.float64)
            for lbl in label_columns
        ],
        axis=1,
    )
    factors = np.array(
        [n / (float(flags[:, j].sum()) + 1.0) for j in range(len(label_columns))],
        dtype=np.float64,
    )
    log_f = np.log(np.maximum(factors, 1e-12))
    log_weights = flags @ log_f
    weights = np.exp(log_weights)
    weights = weights / weights.sum() * n
    d = float(dampen)
    if d <= 0.0:
        d = 1.0
    if abs(d - 1.0) > 1e-6:
        weights = np.power(np.maximum(weights, 1e-12), d)
        weights = weights / weights.sum() * n
    return torch.tensor(weights.astype(np.float32))
