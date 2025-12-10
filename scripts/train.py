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
from scripts.generate_visualizations import generate_heatmaps_for_checkpoint


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
    # Optional speedup (PyTorch 2.x): compile model for RTX 3000+
    # Guarded: skip if Triton/Inductor is not available/working (common on Windows)
    try:
        import importlib
        if importlib.util.find_spec("triton") is None:
            raise ImportError("triton not available; skipping torch.compile")
        model = torch.compile(model, mode="max-autotune")
        print("✅ torch.compile enabled")
    except Exception as e:
        print(f"⚠️ torch.compile skipped: {e}")
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
    # Let PL place checkpoints in the logger directory (lightning_logs/version_x/checkpoints)
    checkpoint_auc = ModelCheckpoint(
        filename="best-auc-{epoch:02d}-{val_auc:.4f}",
        monitor="val_auc",
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    
    checkpoint_f1 = ModelCheckpoint(
        filename="best-f1-{epoch:02d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    
    checkpoint_loss = ModelCheckpoint(
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
        default_root_dir=str(PROJECT_ROOT),  # ensure logger/checkpoints under project root
        callbacks=[checkpoint_auc, checkpoint_f1, checkpoint_loss],  # Add callbacks here
    )

    # 1. RUN TRAINING
    trainer.fit(
        lightning_module,
        train_dataloaders=dataloaders["train"],
        val_dataloaders=dataloaders["val"],
    )
    
    # Helper for heatmaps
    def run_heatmap_generation(
        ckpt_path: str,
        output_name: str,
        version_dir: Path,
    ) -> None:
        try:
            generate_heatmaps_for_checkpoint(
                checkpoint_path=Path(ckpt_path),
                config_path=args.config,
                output_dir=Path("outputs") / "heatmaps",
                model_type=output_name,
                version=version_dir.name,
                batch_size=16,
                num_samples=32,
                target_class=None,
                method="gradcam++",
                num_workers=0,
            )
        except Exception as e:
            print(f"    [Error] Heatmap generation failed for {output_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*50)
    print("TRAINING COMPLETE. STARTING EVALUATION & VISUALIZATION.")
    print("="*50)

    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Get Version Info (Matches lightning_logs/version_X)
    try:
        version = trainer.logger.version if trainer.logger else "unknown"
    except Exception:
        version = "unknown"

    version_str = f"version_{version}"
    print(f"[*] Current Run Version: {version_str}")

    # 3. Setup Heatmap Directory: outputs/heatmaps/version_X
    heatmap_version_dir = Path("outputs") / "heatmaps" / version_str
    heatmap_version_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Heatmaps will be saved to: {heatmap_version_dir}")

    checkpoints_to_process = [
        ("best_auc_model", checkpoint_auc),
        ("best_f1_model", checkpoint_f1),
        ("best_loss_model", checkpoint_loss),
    ]

    # Helper to print where files actually went
    if trainer.logger and trainer.logger.log_dir:
        print(f"[*] Artifacts are saved in: {trainer.logger.log_dir}")

    for name, callback in checkpoints_to_process:
        ckpt_path = callback.best_model_path

        # ROBUST SAFETY CHECK
        if ckpt_path and os.path.isfile(ckpt_path):
            print(f"\n>>> PROCESSING: {name}")
            print(f"    Checkpoint: {ckpt_path}")

            # A. Test Metrics (Numerical)
            try:
                print("    Running numerical test...")
                trainer.test(dataloaders=dataloaders["test"], ckpt_path=ckpt_path)
            except Exception as e:
                print(f"    [Error] Numerical testing failed: {e}")

            # B. Generate Heatmaps (Visual)
            run_heatmap_generation(
                ckpt_path=ckpt_path,
                output_name=name,
                version_dir=heatmap_version_dir,
            )
        else:
            print(f"\n>>> SKIPPING: {name}")
            if not ckpt_path:
                print("    Reason: No checkpoint was saved (metric likely never improved).")
            else:
                print(f"    Reason: File not found at '{ckpt_path}'")

    print("\n" + "="*60)
    print("✅ ALL DONE! Training, Testing, and Heatmap Generation Complete!")
    print("="*40)
    print(f"📁 Heatmaps saved in: {heatmap_version_dir}")
    print("   Structure:")
    print("   - best_auc_model/   (Best AUC model heatmaps)")
    print("   - best_f1_model/    (Best F1 model heatmaps)")
    print("   - best_loss_model/  (Best Loss model heatmaps)")
    print("="*40)


if __name__ == "__main__":
    main()

