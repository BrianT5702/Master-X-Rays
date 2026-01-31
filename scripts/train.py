from __future__ import annotations

import argparse
from pathlib import Path
import sys
import os

# Set PyTorch cache to D drive before importing torch
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TORCH_CACHE_DIR = PROJECT_ROOT / ".torch_cache"
os.environ["TORCH_HOME"] = str(TORCH_CACHE_DIR)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
# Optimize Tensor Core usage for RTX 3050 Ti (optional performance boost)
torch.set_float32_matmul_precision('medium')  # 'medium' = good balance, 'high' = faster but less precise

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
import subprocess

from src.data.datasets import create_dataloaders
from src.models.basemodels import build_backbone
from src.training.lightning_module import RareDiseaseModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train rare lung disease detection model.")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"), help="Path to YAML config.")
    parser.add_argument("--accelerator", type=str, default="auto", help="PyTorch Lightning accelerator.")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use.")
    # No checkpoint argument: always start a fresh run
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_cfg = config["data"]
    training_cfg = config["training"]
    model_cfg = config["model"]
    loss_cfg = config.get("loss", {})

    label_columns = ["Nodule", "Fibrosis"]

    images_root = Path(data_cfg["nih_root"])
    csv_path = Path(data_cfg["csv_path"])

    augmentation_config = config.get("augmentations", {})
    use_weighted_sampling = training_cfg.get("use_weighted_sampling", False)
    use_roi_extraction = data_cfg.get("use_roi_extraction", True)
    patient_split = data_cfg.get("patient_split", True)
    cache_dir = data_cfg.get("cache_dir", None)
    if cache_dir:
        cache_dir = Path(cache_dir)
    
    dataloaders = create_dataloaders(
        csv_path=csv_path,
        images_root=images_root,
        label_columns=label_columns,
        image_size=tuple(data_cfg["image_size"]),
        batch_size=training_cfg["batch_size"],
        splits=(
            data_cfg["train_split"],
            data_cfg["val_split"],
            data_cfg["test_split"],
        ),
        num_workers=data_cfg["num_workers"],
        seed=config["experiment"]["seed"],
        augmentation_config=augmentation_config,
        use_weighted_sampling=use_weighted_sampling,
        patient_split=patient_split,
        use_roi_extraction=use_roi_extraction,
        cache_dir=cache_dir,
    )

    # Build model architecture
    backbone = build_backbone(
        model_cfg["backbone"],
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
    )
    
    # Get prediction threshold from config (default 0.3 for imbalanced rare disease detection)
    prediction_threshold = training_cfg.get("prediction_threshold", 0.3)
    class_weights = model_cfg.get("class_weights", None)  # Optional: ASL handles imbalance internally
    
    # Always start a fresh LightningModule for a new training run
    print("🆕 Starting new training run (no checkpoint resumption)")
    lightning_module = RareDiseaseModule(
        model=backbone,
        learning_rate=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
        class_weights=class_weights,
        loss_config=loss_cfg,
        max_epochs=training_cfg["max_epochs"],
        prediction_threshold=prediction_threshold,
        samples_per_class=None,
    )

    # Auto-detect GPU, fall back to CPU if not available
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"🔍 Debug - PyTorch version: {torch.__version__}")
    print(f"🔍 Debug - CUDA available: {cuda_available}")
    if cuda_available:
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"❌ CUDA not available. PyTorch version: {torch.__version__}")
        print(f"   This might be the CPU-only version. Check: pip show torch")
    
    # Determine accelerator
    if args.accelerator == "gpu" or args.accelerator == "cuda":
        if cuda_available:
            accelerator = "gpu"  # PyTorch Lightning accepts "gpu" or "cuda"
        else:
            print("⚠️  GPU requested but not available. Falling back to CPU.")
            accelerator = "cpu"
    elif args.accelerator == "auto":
        accelerator = "gpu" if cuda_available else "cpu"
    else:
        accelerator = args.accelerator
    
    # Set precision based on accelerator
    if accelerator == "cpu":
        precision = 32  # Must use 32-bit on CPU
        if isinstance(training_cfg["precision"], str) and "mixed" in training_cfg["precision"]:
            print("⚠️  Using CPU: switching to 32-bit precision (mixed precision not supported on CPU)")
    else:
        precision = training_cfg["precision"]  # Use config precision for GPU

    # Gradient accumulation for high-resolution training (320x320)
    # Allows effective larger batch size when GPU memory is limited
    accumulate_grad_batches = training_cfg.get("accumulate_grad_batches", 1)
    
    # Model checkpointing:
    # - Primary: governed by Macro-Averaged F1-Score (per thesis)
    # - Secondary: also track best AUROC and best validation loss for analysis
    checkpoint_f1 = ModelCheckpoint(
        monitor="val_f1",
        mode="max",
        filename="best-f1-{epoch:02d}-{val_f1:.4f}",
        save_top_k=1,
        save_last=True,
    )
    checkpoint_auc = ModelCheckpoint(
        monitor="val_auc",
        mode="max",
        filename="best-auc-{epoch:02d}-{val_auc:.4f}",
        save_top_k=1,
        save_last=False,
    )
    checkpoint_loss = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="best-loss-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        save_last=False,
    )
    
    # Explicitly disable early stopping - train for full max_epochs
    # No EarlyStopping callback is added, so training will run for all epochs
    trainer = pl.Trainer(
        max_epochs=training_cfg["max_epochs"],
        accelerator=accelerator,
        devices=args.devices,
        precision=precision,
        log_every_n_steps=10,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=[checkpoint_f1, checkpoint_auc, checkpoint_loss],
        enable_progress_bar=True,
    )

    trainer.fit(
        lightning_module,
        train_dataloaders=dataloaders["train"],
        val_dataloaders=dataloaders["val"],
    )
    trainer.test(lightning_module, dataloaders["test"])
    
    # Generate heatmaps automatically after training completes
    print("\n" + "="*60)
    print("Training completed! Generating heatmaps...")
    print("="*60)
    
    # Get the best checkpoint path
    # Use best AUC checkpoint for heatmaps (better metric for rare disease detection)
    # Fallback to best F1 if AUC checkpoint not available
    best_checkpoint = checkpoint_auc.best_model_path if checkpoint_auc.best_model_path else checkpoint_f1.best_model_path
    if checkpoint_auc.best_model_path:
        print(f"📊 Using best AUC checkpoint (val_auc={checkpoint_auc.best_model_score:.4f})")
    else:
        print(f"📊 Using best F1 checkpoint (val_f1={checkpoint_f1.best_model_score:.4f})")
    if best_checkpoint:
        # Convert to absolute path if relative
        if not Path(best_checkpoint).is_absolute():
            best_checkpoint = Path(best_checkpoint).resolve()
        else:
            best_checkpoint = Path(best_checkpoint)
        
        print(f"Using best checkpoint: {best_checkpoint}")
        
        # Generate heatmaps for test set
        heatmap_script = PROJECT_ROOT / "scripts" / "generate_heatmaps.py"
        if heatmap_script.exists():
            try:
                # Use sys.executable to ensure we use the same Python interpreter
                # This fixes ModuleNotFoundError when subprocess uses different Python
                subprocess.run([
                    sys.executable, str(heatmap_script),
                    "--checkpoint", str(best_checkpoint),
                    "--config", str(args.config),
                    "--split", "test",
                    "--num_samples", "20",
                    "--output_dir", str(PROJECT_ROOT / "heatmaps"),
                ], check=True, cwd=str(PROJECT_ROOT))
                print(f"\n✅ Heatmaps generated successfully!")
                print(f"   Location: {PROJECT_ROOT / 'heatmaps' / 'test' / 'gradcam'}")
            except subprocess.CalledProcessError as e:
                print(f"\n⚠️  Warning: Failed to generate heatmaps: {e}")
                print("   You can generate them manually using:")
                print(f"   python scripts/generate_heatmaps.py --checkpoint {best_checkpoint} --config {args.config}")
        else:
            print(f"\n⚠️  Warning: Heatmap script not found at {heatmap_script}")
    else:
        print("\n⚠️  Warning: No best checkpoint found. Heatmaps not generated.")
        print("   You can generate them manually using:")
        print("   python scripts/generate_heatmaps.py --checkpoint <checkpoint_path> --config <config_path>")


if __name__ == "__main__":
    main()

