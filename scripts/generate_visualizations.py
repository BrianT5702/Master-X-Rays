"""Generate Grad-CAM heatmaps for trained model predictions.

This script loads a trained checkpoint and generates visualization heatmaps
for XAI (Explainable AI) analysis. Useful for thesis Chapter 4 (Results).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import create_dataloaders
from src.models.basemodels import build_backbone
from src.training.lightning_module import RareDiseaseModule
from src.xai.gradcam import generate_heatmap
from src.xai.visualize import save_heatmap_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM heatmaps for trained model."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (.ckpt file). Example: checkpoints/best-f1-epoch=12-val_f1=0.45.ckpt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/base.yaml",
        help="Path to YAML config file (for dataset paths and model settings).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "heatmaps",
        help="Directory to save heatmap visualizations.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for visualization (smaller = less memory).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=32,
        help="Number of samples to visualize (from test set).",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Target class index (0=Nodule, 1=Fibrosis). If None, uses highest prediction.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="gradcam++",
        choices=["gradcam", "gradcam++"],
        help="Grad-CAM method to use.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of data loading workers (0 for Windows compatibility).",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    """Load YAML configuration file."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_heatmaps_for_checkpoint(
    checkpoint_path: Path,
    config_path: Path,
    output_dir: Path,
    model_type: str = "unknown",
    version: str = "version_unknown",
    batch_size: int = 16,
    num_samples: int = 32,
    target_class: int | None = None,
    method: str = "gradcam++",
    num_workers: int = 0,
) -> None:
    """
    Generate Grad-CAM heatmaps for a trained checkpoint (callable function).
    
    Args:
        checkpoint_path: Path to model checkpoint (.ckpt file)
        config_path: Path to YAML config file
        output_dir: Base directory to save heatmap visualizations
        model_type: Type of model (e.g., "best-auc", "best-f1", "best-loss") - used for folder name
        version: Version identifier (e.g., "version_01", "version_20240115_143022") - creates version subfolder
        batch_size: Batch size for visualization
        num_samples: Number of samples to visualize
        target_class: Target class index (0=Nodule, 1=Fibrosis). If None, uses highest prediction.
        method: Grad-CAM method ('gradcam' or 'gradcam++')
        num_workers: Number of data loading workers
    """
    # Validate checkpoint exists
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found: {checkpoint_path}")
        return
    
    # Load config
    config = load_config(config_path)
    data_cfg = config["data"]
    model_cfg = config["model"]
    
    # Create version-specific output directory structure: outputs/heatmaps/version_XX/model_type/
    version_output_dir = output_dir / version
    model_output_dir = version_output_dir / model_type
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"🔬 GRAD-CAM HEATMAP GENERATION - {version.upper()} / {model_type.upper()}")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Version: {version}")
    print(f"Output directory: {model_output_dir}")
    print(f"Method: {method}")
    print(f"Target class: {target_class if target_class is not None else 'Auto (highest prediction)'}")
    print()
    
    # 1. LOAD DATA
    print("📂 Loading test dataset...")
    csv_path = PROJECT_ROOT / data_cfg["csv_path"]
    images_root = PROJECT_ROOT / data_cfg["nih_root"]
    
    dataloaders = create_dataloaders(
        csv_path=csv_path,
        images_root=images_root,
        label_columns=["Nodule", "Fibrosis"],
        image_size=tuple(data_cfg["image_size"]),
        batch_size=batch_size,
        splits=(0.8, 0.1, 0.1),
        num_workers=num_workers,
        seed=config["experiment"]["seed"],
        patient_split=data_cfg.get("patient_split", False),
    )
    test_loader = dataloaders["test"]
    print(f"✅ Test dataset loaded: {len(test_loader.dataset):,} images")
    print()
    
    # 2. LOAD TRAINED MODEL
    print(f"🤖 Loading model from checkpoint...")
    
    # Build backbone structure (needed for checkpoint loading)
    backbone = build_backbone(
        name=model_cfg["backbone"],
        num_classes=model_cfg["num_classes"],
        pretrained=False,  # We'll load weights from checkpoint
        dropout=model_cfg.get("dropout", 0.0),
    )
    
    # Load Lightning module from checkpoint
    lightning_module = RareDiseaseModule.load_from_checkpoint(
        str(checkpoint_path),
        model=backbone,
        learning_rate=0.0,  # Not used for inference
        weight_decay=0.0,  # Not used for inference
    )
    
    # Set to evaluation mode
    lightning_module.eval()
    lightning_module.freeze()
    
    # Get the actual PyTorch model
    model = lightning_module.model
    
    print(f"✅ Model loaded: {model_cfg['backbone']}")
    print(f"   Device: {next(model.parameters()).device}")
    print()
    
    # 3. GENERATE HEATMAPS
    print(f"🎨 Generating heatmaps...")
    
    # Collect samples from test set
    all_images = []
    all_labels = []
    all_predictions = []
    samples_collected = 0
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(next(model.parameters()).device)
            labels = batch["labels"]
            
            # Get model predictions
            logits = model(images)
            probs = torch.sigmoid(logits)
            
            all_images.append(images.cpu())
            all_labels.append(labels)
            all_predictions.append(logits.cpu())
            
            samples_collected += len(images)
            if samples_collected >= num_samples:
                break
    
    # Concatenate batches
    images_batch = torch.cat(all_images[:num_samples], dim=0)
    labels_batch = torch.cat(all_labels[:num_samples], dim=0)
    predictions_batch = torch.cat(all_predictions[:num_samples], dim=0)
    
    print(f"   Collected {len(images_batch)} samples")
    print(f"   Generating heatmaps for each sample...")
    
    # Generate heatmaps for each image
    heatmap_list = []
    device = next(model.parameters()).device
    
    for i in range(len(images_batch)):
        if (i + 1) % 8 == 0:
            print(f"   Progress: {i + 1}/{len(images_batch)}")
        
        image_tensor = images_batch[i].unsqueeze(0).to(device)
        
        # Generate heatmap
        heatmap = generate_heatmap(
            model=model,
            input_tensor=image_tensor,
            backbone_name=model_cfg["backbone"],
            target_class=target_class,
            method=method,
        )
        
        # Move back to CPU and store
        heatmap_list.append(heatmap.cpu().squeeze(0))
    
    heatmaps_batch = torch.stack(heatmap_list)
    print(f"✅ Generated {len(heatmaps_batch)} heatmaps")
    print()
    
    # 4. SAVE VISUALIZATIONS
    print(f"💾 Saving visualizations to {model_output_dir}...")
    
    class_names = ["Nodule", "Fibrosis"]
    prefix = f"heatmap_{method}"
    if target_class is not None:
        prefix += f"_class{target_class}"
    
    save_heatmap_batch(
        images=images_batch,
        heatmaps=heatmaps_batch,
        predictions=predictions_batch,
        labels=labels_batch,
        class_names=class_names,
        output_dir=model_output_dir,
        prefix=prefix,
    )
    
    print()
    print("=" * 60)
    print(f"✅ DONE - {model_type.upper()}!")
    print("=" * 60)
    print(f"📁 Heatmaps saved to: {model_output_dir}")
    print(f"📊 Generated {len(heatmaps_batch)} visualizations")
    print()


def main() -> None:
    args = parse_args()
    
    # Generate version from timestamp for manual CLI usage
    from datetime import datetime
    version = f"version_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Call the programmatic function
    generate_heatmaps_for_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        model_type="manual",  # For manual CLI usage
        version=version,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        target_class=args.target_class,
        method=args.method,
        num_workers=args.num_workers,
    )
    
    print("💡 Tips:")
    print("   - Red/yellow areas = model attention (important regions)")
    print("   - Filenames include predictions and ground truth labels")
    print("   - Use --target-class 0 for Nodule, 1 for Fibrosis")
    print("   - Use --target-class None to visualize highest prediction")


if __name__ == "__main__":
    main()

