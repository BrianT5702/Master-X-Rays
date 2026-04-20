"""Generate Grad-CAM heatmaps for trained model predictions.

This script loads a trained model checkpoint and generates heatmaps for test/validation
samples to visualize which regions the model focuses on for predictions.

Output layout (default): heatmaps/<split>/<method>/<backbone>/version_<N>/.

Alternative (--layout model_version): <output_dir>/<ModelName>/version_<N>/ e.g. heatmaps/test/Swin-T/version_0/
(use --model_folder_name to override ModelName).

Note: Nodule vs Fibrosis maps for the *same* image can look similar if the two output
heads use correlated features (same backbone, different final weights). They are not
identical: Grad-CAM backprops w.r.t. a different logit each time. Filenames and the
figure suptitle (image id + target class) distinguish runs.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pytorch_lightning as pl
import yaml
from torch.utils.data import DataLoader

from src.data.datasets import create_dataloaders, resolve_data_normalization
from src.models.basemodels import build_backbone
from src.training.lightning_module import RareDiseaseModule
from src.xai.gradcam import generate_heatmap
from src.xai.visualize import visualize_heatmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM heatmaps for trained model."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (.ckpt file).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gpu.yaml"),
        help="Path to YAML config used for training.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to generate heatmaps for.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of samples (X-rays) to generate heatmaps for. Ignored if --all_positives is set.",
    )
    parser.add_argument(
        "--all_positives",
        action="store_true",
        help="Generate heatmaps for every test/val image that has at least one positive (Nodule or Fibrosis). Ignores --num_samples.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("heatmaps"),
        help="Root directory for heatmaps. With --layout model_version, use e.g. heatmaps/test.",
    )
    parser.add_argument(
        "--layout",
        type=str,
        choices=["default", "model_version"],
        default="default",
        help=(
            "default: output_dir/split/method/backbone/version_N. "
            "model_version: output_dir/MODEL_FOLDER/version_N (flat; good for heatmaps/test/Swin-T/version_0)."
        ),
    )
    parser.add_argument(
        "--model_folder_name",
        type=str,
        default=None,
        help="With layout=model_version, subfolder name under output_dir (e.g. Swin-T). Default: derived from backbone.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["gradcam", "gradcam++", "hires_cam"],
        default="hires_cam",
        help="CAM method: hires_cam (default, sharper); gradcam++; gradcam.",
    )
    parser.add_argument(
        "--target_layer_mode",
        type=str,
        choices=["default", "mid_res", "high_res", "ultra_high_res"],
        default="high_res",
        help="CAM layer: high_res = finer grid; ultra_high_res = earlier conv (sharper, can be noisier); mid_res/default = coarser.",
    )
    parser.add_argument(
        "--class_idx",
        type=int,
        default=None,
        help="Specific class index to visualize (0=Nodule, 1=Fibrosis). If None, generates for all classes.",
    )
    parser.add_argument(
        "--heatmap_blur_kernel",
        type=int,
        default=0,
        help="Odd kernel size for spatial smoothing of heatmap (default 0 = no smoothing, clinically strict).",
    )
    parser.add_argument(
        "--heatmap_spread_gamma",
        type=float,
        default=1.35,
        help="Gamma > 1 suppresses weak green/yellow (sharper peaks); < 1 expands highlighted area.",
    )
    parser.add_argument(
        "--heatmap_zero_border_frac",
        type=float,
        default=0.02,
        help="Zero outer fraction of CAM (each side) before plotting; reduces frame-edge peaks. Use 0 for raw.",
    )
    parser.add_argument(
        "--heatmap_pct_low",
        type=float,
        default=3.0,
        help="Lower percentile for display contrast (0 = min); pairs with --heatmap_pct_high.",
    )
    parser.add_argument(
        "--heatmap_pct_high",
        type=float,
        default=97.0,
        help="Upper percentile for display contrast (100 = max). Use 0 and 100 for plain min-max.",
    )
    parser.add_argument(
        "--heatmap_upsample",
        type=str,
        choices=["bicubic", "bilinear"],
        default="bicubic",
        help="Interpolation when upsampling CAM to image size.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Display names for --layout model_version (override with --model_folder_name).
_BACKBONE_FOLDER_DISPLAY: dict[str, str] = {
    "swin_t": "Swin-T",
    "efficientnet_b2": "EfficientNet-B2",
    "densenet121": "DenseNet-121",
    "dense_swin_hybrid": "DenseSwin-Hybrid",
}


def backbone_to_model_folder_name(backbone_name: str) -> str:
    if backbone_name in _BACKBONE_FOLDER_DISPLAY:
        return _BACKBONE_FOLDER_DISPLAY[backbone_name]
    tag = re.sub(r"[^\w\-.]+", "_", str(backbone_name)).strip("_")
    return tag or "model"


def load_model_from_checkpoint(checkpoint_path: Path, config: dict) -> torch.nn.Module:
    """Load trained model from Lightning checkpoint."""
    # Build model architecture (needed for checkpoint loading)
    model_cfg = config["model"]
    training_cfg = config["training"]
    loss_cfg = config.get("loss", {})
    
    backbone = build_backbone(
        model_cfg["backbone"],
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
        dropout=model_cfg.get("dropout", 0.0),  # Must match training config
    )
    
    # Create Lightning module with same config as training
    samples_per_class = model_cfg.get("samples_per_class", None)
    class_weights = model_cfg.get("class_weights", None)
    prediction_threshold = training_cfg.get("prediction_threshold", 0.3)
    
    # Load checkpoint (Lightning will restore the model state)
    f1_metric_threshold = training_cfg.get("f1_metric_threshold", 0.05)
    per_class_thr = training_cfg.get("per_class_thresholds")

    checkpoint_module = RareDiseaseModule.load_from_checkpoint(
        str(checkpoint_path),
        model=backbone,
        learning_rate=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
        class_weights=class_weights,
        loss_config=loss_cfg,
        max_epochs=training_cfg["max_epochs"],
        prediction_threshold=prediction_threshold,
        samples_per_class=samples_per_class,
        warmup_epochs=training_cfg.get("warmup_epochs", 0),
        f1_metric_threshold=f1_metric_threshold,
        per_class_thresholds=per_class_thr,
        backbone_lr_mult=float(training_cfg.get("backbone_lr_mult", 1.0)),
        map_location="cpu",
    )
    
    # Extract the underlying PyTorch model
    model = checkpoint_module.model
    model.eval()
    
    return model


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    
    # Setup paths
    data_cfg = config["data"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    
    label_columns = ["Nodule", "Fibrosis"]
    class_names = label_columns
    
    images_root = Path(data_cfg["nih_root"])
    csv_path = Path(data_cfg["csv_path"])
    
    # Create dataloader for the specified split
    augmentation_config = config.get("augmentations", {})
    use_roi_extraction = data_cfg.get("use_roi_extraction", True)
    apply_roi_mask = data_cfg.get("apply_roi_mask", True)
    use_normalization = data_cfg.get("use_normalization", False)
    normalization = resolve_data_normalization(
        data_cfg.get("normalization"), model_cfg["backbone"]
    )
    cache_dir = data_cfg.get("cache_dir", None)
    if cache_dir:
        cache_dir = Path(cache_dir)
    
    dataloaders = create_dataloaders(
        csv_path=csv_path,
        images_root=images_root,
        label_columns=label_columns,
        image_size=tuple(data_cfg["image_size"]),
        batch_size=1,  # Process one at a time for heatmaps
        splits=(
            data_cfg["train_split"],
            data_cfg["val_split"],
            data_cfg["test_split"],
        ),
        num_workers=0,  # No multiprocessing for visualization
        seed=config["experiment"]["seed"],
        augmentation_config={},  # No augmentation for visualization
        use_weighted_sampling=False,
        patient_split=data_cfg.get("patient_split", True),
        use_roi_extraction=use_roi_extraction,
        apply_roi_mask=apply_roi_mask,
        use_normalization=use_normalization,
        cache_dir=cache_dir,
        normalization=normalization,
    )
    
    dataloader = dataloaders[args.split]
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model_from_checkpoint(args.checkpoint, config)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Get backbone name (from config; must match checkpoint)
    backbone_name = model_cfg["backbone"]
    backbone_tag = re.sub(r"[^\w\-.]+", "_", str(backbone_name)).strip("_") or "model"

    # Extract Lightning version from checkpoint path (e.g. .../csv_metrics_efficientnet_b2/version_1/...)
    version_match = re.search(r"version_(\d+)", str(args.checkpoint))
    if version_match:
        version_num = version_match.group(1)
    else:
        checkpoint_str = str(args.checkpoint)
        version_match = re.search(r"version[_-]?(\d+)", checkpoint_str, re.IGNORECASE)
        version_num = version_match.group(1) if version_match else "unknown"

    if args.layout == "model_version":
        model_folder = args.model_folder_name or backbone_to_model_folder_name(backbone_name)
        safe_folder = re.sub(r'[<>:"/\\|?*]', "_", model_folder).strip(" .")
        if not safe_folder:
            safe_folder = backbone_tag
        output_dir = Path(args.output_dir) / safe_folder / f"version_{version_num}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving heatmaps under (model_version): {safe_folder}/version_{version_num}")
    else:
        # heatmaps/<split>/<method>/<backbone>/version_<N>/ — avoids mixing runs across backbones
        output_dir = (
            args.output_dir / args.split / args.method / backbone_tag / f"version_{version_num}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving heatmaps under: {backbone_tag}/version_{version_num}")
    
    # Determine which classes to visualize
    if args.class_idx is not None:
        class_indices = [args.class_idx]
    else:
        class_indices = list(range(len(class_names)))
    
    if args.all_positives:
        print("Generating heatmaps for ALL images with at least one positive (Nodule or Fibrosis)...")
    else:
        print(f"Generating heatmaps for up to {args.num_samples} X-ray images (x {len(class_indices)} class targets each)...")
    print(f"Classes: {[class_names[i] for i in class_indices]}")
    print(f"Output directory: {output_dir}")
    print(f"Target layer mode: {args.target_layer_mode}")
    print(
        f"Render options: blur_kernel={args.heatmap_blur_kernel}, "
        f"spread_gamma={args.heatmap_spread_gamma}, "
        f"pct=({args.heatmap_pct_low},{args.heatmap_pct_high}), "
        f"upsample={args.heatmap_upsample}, "
        f"zero_border_frac={args.heatmap_zero_border_frac}"
    )
    
    # Generate heatmaps (--num_samples = number of X-rays, not number of PNG files)
    heatmap_files_written = 0
    positives_processed = 0
    images_processed = 0
    for batch_idx, batch in enumerate(dataloader):
        if not args.all_positives and images_processed >= args.num_samples:
            break

        if isinstance(batch, dict):
            images = batch["image"]
            labels = batch["labels"]
            raw_id = batch.get("image_index")
            if raw_id is not None:
                img_id = raw_id[0] if isinstance(raw_id, (list, tuple)) else raw_id
            else:
                img_id = f"batch{batch_idx}"
        else:
            images, labels = batch
            img_id = f"batch{batch_idx}"

        if args.all_positives:
            has_nodule = labels[0, 0].item() >= 0.5
            has_fibrosis = labels[0, 1].item() >= 0.5
            if not (has_nodule or has_fibrosis):
                continue
            positives_processed += 1

        images = images.to(device)

        with torch.no_grad():
            logits = model(images)
            probs = torch.sigmoid(logits)

        # One row per image; Grad-CAM target_class makes Nodule vs Fibrosis maps differ in ∂logit/∂A
        for class_idx in class_indices:
            class_name = class_names[class_idx]

            heatmap = generate_heatmap(
                model=model,
                input_tensor=images,
                backbone_name=backbone_name,
                target_class=class_idx,
                method=args.method,
                target_layer_mode=args.target_layer_mode,
            )

            pred_prob = probs[0, class_idx].item()
            gt_label = labels[0, class_idx].item()

            idx_label = positives_processed if args.all_positives else images_processed
            filename = (
                f"sample_{idx_label:04d}_class_{class_name}_"
                f"pred_{pred_prob:.3f}_gt_{int(gt_label)}.png"
            )
            save_path = output_dir / filename

            figure_title = (
                f"Image: {img_id}  |  Grad-CAM target: {class_name} (logit index {class_idx})"
            )
            visualize_heatmap(
                image=images[0].cpu(),
                heatmap=heatmap[0].cpu(),
                save_path=save_path,
                show=False,
                figure_title=figure_title,
                heatmap_blur_kernel=args.heatmap_blur_kernel,
                heatmap_spread_gamma=args.heatmap_spread_gamma,
                heatmap_zero_border_frac=args.heatmap_zero_border_frac,
                heatmap_percentile_low=args.heatmap_pct_low,
                heatmap_percentile_high=args.heatmap_pct_high,
                heatmap_upsample_mode=args.heatmap_upsample,
            )

            heatmap_files_written += 1

        images_processed += 1

        n_so_far = positives_processed if args.all_positives else images_processed
        if n_so_far > 0 and n_so_far % 20 == 0:
            print(f"Processed {n_so_far} images...")

    total_images = positives_processed if args.all_positives else images_processed
    print(f"\nGenerated {heatmap_files_written} heatmap files ({total_images} X-rays x {len(class_indices)} classes).")
    print(f"   Saved to: {output_dir}")


if __name__ == "__main__":
    main()

