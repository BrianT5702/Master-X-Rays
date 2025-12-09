"""Quick test runner for a saved checkpoint with custom prediction threshold.

Usage (PowerShell):
    python scripts/test_checkpoint.py --ckpt "D:\Master File\Start\checkpoints\best-auc-epoch=05-val_auc=0.7229.ckpt" --config configs/gpu.yaml

Notes:
- Uses prediction_threshold from the config (defaults to 0.5 in gpu.yaml/base.yaml).
- Does NOT retrain; just loads the checkpoint and runs test().
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
import pytorch_lightning as pl

from src.data.datasets import create_dataloaders
from src.models.basemodels import build_backbone
from src.training.lightning_module import RareDiseaseModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a saved checkpoint with custom threshold.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to checkpoint (.ckpt).")
    parser.add_argument("--config", type=Path, default=Path("configs/gpu.yaml"), help="Config YAML (default: configs/gpu.yaml).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    config = yaml.safe_load(args.config.read_text())
    data_cfg = config["data"]
    training_cfg = config["training"]
    model_cfg = config["model"]

    # Build dataloaders
    dataloaders = create_dataloaders(
        csv_path=Path(data_cfg["csv_path"]),
        images_root=Path(data_cfg["nih_root"]),
        label_columns=["Nodule", "Fibrosis"],
        image_size=tuple(data_cfg["image_size"]),
        batch_size=training_cfg["batch_size"],
        splits=(data_cfg["train_split"], data_cfg["val_split"], data_cfg["test_split"]),
        num_workers=data_cfg["num_workers"],
        seed=config["experiment"]["seed"],
        patient_split=data_cfg.get("patient_split", False),
    )

    # Build backbone and load checkpoint with overridden threshold
    backbone = build_backbone(
        model_cfg["backbone"],
        num_classes=model_cfg["num_classes"],
        pretrained=False,  # weights will be loaded from checkpoint
        dropout=model_cfg.get("dropout", 0.0),
    )

    pred_thresh = training_cfg.get("prediction_threshold", 0.5)

    module = RareDiseaseModule.load_from_checkpoint(
        args.ckpt,
        model=backbone,
        learning_rate=0.0,
        weight_decay=0.0,
        prediction_threshold=pred_thresh,
    )

    # Select accelerator/precision
    accel = "gpu" if torch.cuda.is_available() else "cpu"
    prec = 32 if accel == "cpu" else training_cfg["precision"]

    trainer = pl.Trainer(
        accelerator=accel,
        devices=1,
        precision=prec,
        log_every_n_steps=10,
    )

    print(f"\n=== TESTING CHECKPOINT: {args.ckpt} ===")
    print(f"Prediction threshold: {pred_thresh}")
    trainer.test(module, dataloaders=dataloaders["test"])


if __name__ == "__main__":
    main()
