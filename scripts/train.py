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

from src.data.datasets import create_dataloaders
from src.models.basemodels import build_backbone
from src.training.lightning_module import RareDiseaseModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train rare lung disease detection model.")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"), help="Path to YAML config.")
    parser.add_argument("--accelerator", type=str, default="auto", help="PyTorch Lightning accelerator.")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use.")
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
    patient_split = data_cfg.get("patient_split", False)
    
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
    )

    model = build_backbone(
        model_cfg["backbone"],
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
        dropout=model_cfg.get("dropout", 0.0),
    )
    # Get prediction threshold from config (default 0.3 for imbalanced rare disease detection)
    prediction_threshold = training_cfg.get("prediction_threshold", 0.3)
    
    samples_per_class = model_cfg.get("samples_per_class", None)
    
    lightning_module = RareDiseaseModule(
        model=model,
        learning_rate=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
        class_weights=model_cfg["class_weights"],
        loss_config=loss_cfg,
        max_epochs=training_cfg["max_epochs"],
        prediction_threshold=prediction_threshold,
        samples_per_class=samples_per_class,
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

    # --- CHECKPOINT CALLBACKS ---
    # Create checkpoints directory if it doesn't exist
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # 1. Best AUC (Standard stability metric)
    checkpoint_auc = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="best-auc-{epoch:02d}-{val_auc:.4f}",
        monitor="val_auc",
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    
    # 2. Best F1 Score (Critical for rare disease balance)
    checkpoint_f1 = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="best-f1-{epoch:02d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    
    # 3. Lowest Validation Loss (Safest against overfitting)
    checkpoint_loss = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="best-loss-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )

    trainer = pl.Trainer(
        max_epochs=training_cfg["max_epochs"],
        accelerator=accelerator,
        devices=args.devices,
        precision=precision,
        log_every_n_steps=10,
        callbacks=[checkpoint_auc, checkpoint_f1, checkpoint_loss],  # Add callbacks here
    )
    
    # 1. RUN TRAINING
    trainer.fit(
        lightning_module,
        train_dataloaders=dataloaders["train"],
        val_dataloaders=dataloaders["val"],
    )
    
    # 2. TEST AUTOMATICALLY (Best Models)
    print("\n" + "="*60)
    print("TRAINING COMPLETE. STARTING EVALUATION OF BEST CHECKPOINTS.")
    print("="*60)
    
    # Test Best AUC Model
    if checkpoint_auc.best_model_path:
        print(f"\n>>> Testing Model with Best AUC: {checkpoint_auc.best_model_path}")
        trainer.test(dataloaders=dataloaders["test"], ckpt_path=checkpoint_auc.best_model_path)
    else:
        print("\n>>> Warning: No Best AUC checkpoint found (metric might be NaN).")
    
    # Test Best F1 Model (Most important for your thesis)
    if checkpoint_f1.best_model_path:
        print(f"\n>>> Testing Model with Best F1 Score: {checkpoint_f1.best_model_path}")
        trainer.test(dataloaders=dataloaders["test"], ckpt_path=checkpoint_f1.best_model_path)
    else:
        print("\n>>> Warning: No Best F1 checkpoint found.")
    
    # Test Best Loss Model
    if checkpoint_loss.best_model_path:
        print(f"\n>>> Testing Model with Best (Lowest) Loss: {checkpoint_loss.best_model_path}")
        trainer.test(dataloaders=dataloaders["test"], ckpt_path=checkpoint_loss.best_model_path)
    else:
        print("\n>>> Warning: No Best Loss checkpoint found.")


if __name__ == "__main__":
    main()

