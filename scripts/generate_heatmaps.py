"""Generate Grad-CAM heatmaps for trained model predictions.

This script loads a trained model checkpoint and generates heatmaps for test/validation
samples to visualize which regions the model focuses on for predictions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pytorch_lightning as pl
import yaml
from torch.utils.data import DataLoader

from src.data.datasets import create_dataloaders
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
        help="Number of samples to generate heatmaps for.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("heatmaps"),
        help="Directory to save heatmap visualizations.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["gradcam", "gradcam++"],
        default="gradcam",
        help="Grad-CAM method to use.",
    )
    parser.add_argument(
        "--class_idx",
        type=int,
        default=None,
        help="Specific class index to visualize (0=Nodule, 1=Fibrosis). If None, generates for all classes.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
        cache_dir=cache_dir,
    )
    
    dataloader = dataloaders[args.split]
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model_from_checkpoint(args.checkpoint, config)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Get backbone name
    backbone_name = model_cfg["backbone"]
    
    # Create output directory
    output_dir = args.output_dir / args.split / args.method
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which classes to visualize
    if args.class_idx is not None:
        class_indices = [args.class_idx]
    else:
        class_indices = list(range(len(class_names)))
    
    print(f"Generating heatmaps for {args.num_samples} samples...")
    print(f"Classes: {[class_names[i] for i in class_indices]}")
    print(f"Output directory: {output_dir}")
    
    # Generate heatmaps
    samples_processed = 0
    for batch_idx, batch in enumerate(dataloader):
        if samples_processed >= args.num_samples:
            break
        
        # Handle dictionary batch format
        if isinstance(batch, dict):
            images = batch["image"]
            labels = batch["labels"]
        else:
            images, labels = batch
        
        images = images.to(device)
        
        # Forward pass to get predictions (no_grad for prediction only)
        with torch.no_grad():
            logits = model(images)
            probs = torch.sigmoid(logits)
        
        # Generate heatmap for each specified class
        # CRITICAL: Must be OUTSIDE no_grad context for gradients to work!
        for class_idx in class_indices:
            class_name = class_names[class_idx]
            
            # Generate heatmap (generate_heatmap handles gradients internally)
            # This MUST be outside no_grad() context
            heatmap = generate_heatmap(
                model=model,
                input_tensor=images,
                backbone_name=backbone_name,
                target_class=class_idx,
                method=args.method,
            )
            
            # Get predictions and ground truth
            pred_prob = probs[0, class_idx].item()
            gt_label = labels[0, class_idx].item()
            
            # Create filename
            filename = (
                f"sample_{batch_idx:04d}_class_{class_name}_"
                f"pred_{pred_prob:.3f}_gt_{int(gt_label)}.png"
            )
            save_path = output_dir / filename
            
            # Visualize and save
            visualize_heatmap(
                image=images[0].cpu(),
                heatmap=heatmap[0].cpu(),
                save_path=save_path,
                show=False,
            )
            
            samples_processed += 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} samples...")
    
    print(f"\n✅ Generated {samples_processed} heatmap visualizations!")
    print(f"   Saved to: {output_dir}")


if __name__ == "__main__":
    main()

