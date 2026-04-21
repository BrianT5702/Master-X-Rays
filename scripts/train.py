from __future__ import annotations

import argparse
import re
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
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
from pytorch_lightning.loggers import CSVLogger
import yaml
import subprocess

from src.data.datasets import create_dataloaders, resolve_data_normalization
from src.models.basemodels import build_backbone
from src.training.lightning_module import RareDiseaseModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train rare lung disease detection model.")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"), help="Path to YAML config.")
    parser.add_argument("--accelerator", type=str, default="auto", help="PyTorch Lightning accelerator.")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use.")
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume training from (e.g., last.ckpt).")
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
    weighted_sampling_dampen = float(training_cfg.get("weighted_sampling_dampen", 1.0))
    use_roi_extraction = data_cfg.get("use_roi_extraction", True)
    apply_roi_mask = data_cfg.get("apply_roi_mask", True)
    use_normalization = data_cfg.get("use_normalization", False)
    norm_cfg = data_cfg.get("normalization")
    normalization = resolve_data_normalization(norm_cfg, model_cfg["backbone"])
    patient_split = data_cfg.get("patient_split", True)
    cache_dir = data_cfg.get("cache_dir", None)
    if cache_dir:
        cache_dir = Path(cache_dir)

    if use_normalization:
        nc = None if norm_cfg is None else str(norm_cfg).strip().lower()
        src = (
            "config"
            if nc in ("legacy", "imagenet")
            else f"auto (backbone={model_cfg['backbone']})"
        )
        print(f"Data normalization: {normalization} ({src})")

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
        weighted_sampling_dampen=weighted_sampling_dampen,
        patient_split=patient_split,
        use_roi_extraction=use_roi_extraction,
        apply_roi_mask=apply_roi_mask,
        use_normalization=use_normalization,
        cache_dir=cache_dir,
        normalization=normalization,
    )

    # Build model architecture
    backbone = build_backbone(
        model_cfg["backbone"],
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
        dropout=model_cfg.get("dropout", 0.0),
    )
    
    # Get prediction threshold from config (default 0.3 for imbalanced rare disease detection)
    prediction_threshold = training_cfg.get("prediction_threshold", 0.3)
    f1_metric_threshold = training_cfg.get("f1_metric_threshold", 0.05)  # Low threshold so logged F1 responds (ASL probs modest)
    per_class_thresholds = training_cfg.get("per_class_thresholds")  # Optional [Nodule, Fibrosis] for Binary F1/acc per label
    class_weights = model_cfg.get("class_weights", None)  # Optional: ASL handles imbalance internally
    
    # Check if resuming from checkpoint
    ckpt_path = None
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.is_absolute():
            ckpt_path = PROJECT_ROOT / ckpt_path
        if not ckpt_path.exists():
            print(f"Error: Checkpoint not found at {ckpt_path}")
            print("   Starting fresh training run instead.")
            ckpt_path = None
        else:
            print(f"Resuming training from checkpoint: {ckpt_path}")
    
    if ckpt_path is None:
        print("Starting new training run (no checkpoint resumption)")
        warmup_epochs = training_cfg.get("warmup_epochs", 0)
        backbone_lr_mult = float(training_cfg.get("backbone_lr_mult", 1.0))
        lightning_module = RareDiseaseModule(
            model=backbone,
            learning_rate=training_cfg["learning_rate"],
            weight_decay=training_cfg["weight_decay"],
            class_weights=class_weights,
            loss_config=loss_cfg,
            max_epochs=training_cfg["max_epochs"],
            prediction_threshold=prediction_threshold,
            samples_per_class=None,
            warmup_epochs=warmup_epochs,
            f1_metric_threshold=f1_metric_threshold,
            per_class_thresholds=per_class_thresholds,
            backbone_lr_mult=backbone_lr_mult,
        )
    else:
        # Load from checkpoint - Lightning will restore model, optimizer, and epoch
        print("Loading model from checkpoint...")
        warmup_epochs = training_cfg.get("warmup_epochs", 0)
        backbone_lr_mult = float(training_cfg.get("backbone_lr_mult", 1.0))
        lightning_module = RareDiseaseModule.load_from_checkpoint(
            str(ckpt_path),
            model=backbone,
            learning_rate=training_cfg["learning_rate"],
            weight_decay=training_cfg["weight_decay"],
            class_weights=class_weights,
            loss_config=loss_cfg,
            max_epochs=training_cfg["max_epochs"],
            prediction_threshold=prediction_threshold,
            samples_per_class=None,
            warmup_epochs=warmup_epochs,
            f1_metric_threshold=f1_metric_threshold,
            per_class_thresholds=per_class_thresholds,
            backbone_lr_mult=backbone_lr_mult,
        )
        print("Checkpoint loaded successfully!")

    # Auto-detect GPU, fall back to CPU if not available
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"Debug - PyTorch version: {torch.__version__}")
    print(f"Debug - CUDA available: {cuda_available}")
    if cuda_available:
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"❌ CUDA not available. PyTorch version: {torch.__version__}")
        print(f"   This might be the CPU-only version. Check: pip show torch")
    
    # Determine accelerator
    if args.accelerator == "gpu" or args.accelerator == "cuda":
        if cuda_available:
            accelerator = "gpu"  # PyTorch Lightning accepts "gpu" or "cuda"
        else:
            print("GPU requested but not available. Falling back to CPU.")
            accelerator = "cpu"
    elif args.accelerator == "auto":
        accelerator = "gpu" if cuda_available else "cpu"
    else:
        accelerator = args.accelerator
    
    # Set precision based on accelerator
    if accelerator == "cpu":
        precision = 32  # Must use 32-bit on CPU
        if isinstance(training_cfg["precision"], str) and "mixed" in training_cfg["precision"]:
            print("Using CPU: switching to 32-bit precision (mixed precision not supported on CPU)")
    else:
        precision = training_cfg["precision"]  # Use config precision for GPU

    # Gradient accumulation for high-resolution training (320x320)
    # Allows effective larger batch size when GPU memory is limited
    accumulate_grad_batches = training_cfg.get("accumulate_grad_batches", 1)
    
    # Model checkpointing: prefer rare-disease clinical metrics when per-class thresholds are set.
    checkpoint_sensitivity = None
    if per_class_thresholds is not None:
        checkpoint_sensitivity = ModelCheckpoint(
            monitor="val_macro_sensitivity",
            mode="max",
            filename="best-sensitivity-{epoch:02d}-{val_macro_sensitivity:.4f}",
            save_top_k=1,
            save_last=False,
        )
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
    # Periodic checkpoint: save every 5 epochs to prevent data loss from crashes (saves space)
    checkpoint_periodic = ModelCheckpoint(
        monitor="val_auc",  # Monitor AUC to include in filename
        filename="epoch-{epoch:02d}-{val_auc:.4f}",
        every_n_epochs=5,  # Changed from 2 to 5 to save disk space
        save_top_k=-1,  # Keep all periodic checkpoints
        save_last=False,
    )

    callbacks_list: list = []
    if checkpoint_sensitivity is not None:
        callbacks_list.append(checkpoint_sensitivity)
    callbacks_list.extend(
        [checkpoint_f1, checkpoint_auc, checkpoint_loss, checkpoint_periodic]
    )
    
    # CSV logger: separate run folders per backbone (e.g. lightning_logs/csv_metrics_efficientnet_b0/version_2/)
    log_save_dir = PROJECT_ROOT / "lightning_logs"
    backbone_tag = str(model_cfg.get("backbone", "model")).replace(" ", "_")
    log_name = f"csv_metrics_{backbone_tag}"
    log_version = None
    if ckpt_path is not None:
        path_str = str(ckpt_path).replace("\\", "/")
        m_new = re.search(r"csv_metrics_([^/]+)/version_(\d+)", path_str, re.IGNORECASE)
        m_old = re.search(r"(?:^|/)csv_metrics/version_(\d+)", path_str, re.IGNORECASE)
        if m_new:
            log_name = f"csv_metrics_{m_new.group(1)}"
            log_version = int(m_new.group(2))
            print(f"Resuming in same version folder: {log_name}/version_{log_version}")
        elif m_old:
            log_name = "csv_metrics"
            log_version = int(m_old.group(1))
            print(f"Resuming (legacy layout): {log_name}/version_{log_version}")
    csv_logger = CSVLogger(
        save_dir=str(log_save_dir),
        name=log_name,
        version=log_version,  # None = new version; int = same folder when resuming
    )

    early_stop_patience = training_cfg.get("early_stopping_patience", None)
    early_stop_monitor = training_cfg.get("early_stopping_monitor", "val_auc")
    early_stop_mode = training_cfg.get("early_stopping_mode")
    if early_stop_mode is None:
        early_stop_mode = "max" if early_stop_monitor != "val_loss" else "min"
    if early_stop_patience is not None and early_stop_patience > 0:
        early_stop = EarlyStopping(
            monitor=early_stop_monitor,
            mode=str(early_stop_mode),
            patience=int(early_stop_patience),
            verbose=True,
        )
        callbacks_list.append(early_stop)
        print(
            f"Early stopping enabled: monitor={early_stop_monitor} ({early_stop_mode}), "
            f"patience={early_stop_patience}"
        )
    swa_start = training_cfg.get("swa_epoch_start")
    if swa_start is not None:
        callbacks_list.append(
            StochasticWeightAveraging(
                swa_lrs=training_cfg["learning_rate"],
                swa_epoch_start=float(swa_start),
            )
        )
        print(f"Stochastic Weight Averaging enabled from epoch fraction {float(swa_start)}")
    _blr = float(training_cfg.get("backbone_lr_mult", 1.0))
    if _blr < 1.0:
        _lr = float(training_cfg["learning_rate"])
        print(f"Discriminative LR: backbone {_lr * _blr:.2e}, head {_lr:.2e} (backbone_lr_mult={_blr})")
    gradient_clip_val = training_cfg.get("gradient_clip_val", None)
    trainer = pl.Trainer(
        max_epochs=training_cfg["max_epochs"],
        accelerator=accelerator,
        devices=args.devices,
        precision=precision,
        log_every_n_steps=10,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,  # Prevent gradient explosion
        callbacks=callbacks_list,
        logger=csv_logger,
        enable_progress_bar=True,
    )

    trainer.fit(
        lightning_module,
        train_dataloaders=dataloaders["train"],
        val_dataloaders=dataloaders["val"],
        ckpt_path=str(ckpt_path) if ckpt_path else None,  # Resume from checkpoint if provided
    )
    # Test on best clinical weights when available (macro sensitivity), else macro F1, else AUC
    best_test_ckpt = None
    if checkpoint_sensitivity is not None and checkpoint_sensitivity.best_model_path:
        best_test_ckpt = checkpoint_sensitivity.best_model_path
    if not best_test_ckpt:
        best_test_ckpt = checkpoint_f1.best_model_path or checkpoint_auc.best_model_path
    if best_test_ckpt:
        best_test_ckpt = str(Path(best_test_ckpt).resolve())
        print(f"\nTest set: using best checkpoint weights (not last epoch): {best_test_ckpt}")
        trainer.test(lightning_module, dataloaders=dataloaders["test"], ckpt_path=best_test_ckpt)
    else:
        print("\nTest set: no best checkpoint saved; evaluating current (last epoch) weights.")
        trainer.test(lightning_module, dataloaders=dataloaders["test"])
    
    # Generate heatmaps automatically after training completes
    print("\n" + "="*60)
    print("Training completed! Generating heatmaps...")
    print("="*60)
    
    # Get the best checkpoint path
    best_checkpoint = None
    if checkpoint_sensitivity is not None and checkpoint_sensitivity.best_model_path:
        best_checkpoint = checkpoint_sensitivity.best_model_path
        print(
            f"Using best sensitivity checkpoint (val_macro_sensitivity={checkpoint_sensitivity.best_model_score:.4f})"
        )
    if not best_checkpoint and checkpoint_f1.best_model_path:
        best_checkpoint = checkpoint_f1.best_model_path
        print(f"Using best F1 checkpoint (val_f1={checkpoint_f1.best_model_score:.4f})")
    if not best_checkpoint and checkpoint_auc.best_model_path:
        best_checkpoint = checkpoint_auc.best_model_path
        print(f"Using best AUC checkpoint (val_auc={checkpoint_auc.best_model_score:.4f})")
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
                    "--method", "hires_cam",
                    "--target_layer_mode", "high_res",
                    "--heatmap_blur_kernel", "0",
                    "--heatmap_spread_gamma", "1.35",
                    "--heatmap_pct_low", "3",
                    "--heatmap_pct_high", "97",
                    "--heatmap_upsample", "bicubic",
                    "--heatmap_zero_border_frac", "0.02",
                ], check=True, cwd=str(PROJECT_ROOT))
                print("\nHeatmaps generated successfully!")
                bb = str(model_cfg.get("backbone", "model")).replace("/", "_")
                print(
                    f"   Location: {PROJECT_ROOT / 'heatmaps' / 'test' / 'hires_cam' / bb} "
                    f"(version_* from checkpoint path)"
                )
            except subprocess.CalledProcessError as e:
                print(f"\nWarning: Failed to generate heatmaps: {e}")
                print("   You can generate them manually using:")
                print(f"   python scripts/generate_heatmaps.py --checkpoint {best_checkpoint} --config {args.config}")
        else:
            print(f"\nWarning: Heatmap script not found at {heatmap_script}")
    else:
        print("\nWarning: No best checkpoint found. Heatmaps not generated.")
        print("   You can generate them manually using:")
        print("   python scripts/generate_heatmaps.py --checkpoint <checkpoint_path> --config <config_path>")


if __name__ == "__main__":
    main()

