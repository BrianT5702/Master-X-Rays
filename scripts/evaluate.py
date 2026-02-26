"""Run test set evaluation and optional heatmaps for a trained checkpoint (e.g. best AUC)."""

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

from src.data.datasets import create_dataloaders
from src.models.basemodels import build_backbone
from src.training.lightning_module import RareDiseaseModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on the test set (and optionally generate heatmaps)."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint (e.g. best-auc-*.ckpt).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gpu.yaml"),
        help="Config used for training.",
    )
    parser.add_argument(
        "--heatmaps",
        action="store_true",
        help="Also run heatmap generation after evaluation.",
    )
    parser.add_argument(
        "--heatmap_samples",
        type=int,
        default=20,
        help="Number of samples for heatmaps (default 20).",
    )
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
        use_weighted_sampling=False,
        patient_split=patient_split,
        use_roi_extraction=use_roi_extraction,
        cache_dir=cache_dir,
    )

    backbone = build_backbone(
        model_cfg["backbone"],
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
        dropout=model_cfg.get("dropout", 0.0),
    )

    class_weights = model_cfg.get("class_weights", None)
    prediction_threshold = training_cfg.get("prediction_threshold", 0.3)
    warmup_epochs = training_cfg.get("warmup_epochs", 0)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    lightning_module = RareDiseaseModule.load_from_checkpoint(
        str(ckpt_path),
        model=backbone,
        learning_rate=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
        class_weights=class_weights,
        loss_config=loss_cfg,
        max_epochs=training_cfg["max_epochs"],
        prediction_threshold=prediction_threshold,
        samples_per_class=model_cfg.get("samples_per_class", None),
        warmup_epochs=warmup_epochs,
        map_location="cpu",
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    precision = 32 if accelerator == "cpu" else training_cfg.get("precision", "16-mixed")

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        precision=precision,
        enable_progress_bar=True,
    )

    print("Running test set evaluation...")
    trainer.test(lightning_module, dataloaders["test"])

    if args.heatmaps:
        import subprocess
        heatmap_script = PROJECT_ROOT / "scripts" / "generate_heatmaps.py"
        if heatmap_script.exists():
            print("\nGenerating heatmaps...")
            subprocess.run([
                sys.executable, str(heatmap_script),
                "--checkpoint", str(ckpt_path),
                "--config", str(args.config),
                "--split", "test",
                "--num_samples", str(args.heatmap_samples),
                "--output_dir", str(PROJECT_ROOT / "heatmaps"),
            ], check=True, cwd=str(PROJECT_ROOT))
            print("Heatmaps saved under heatmaps/test/gradcam/version_XX/")
        else:
            print("Heatmap script not found; skipping heatmaps.")


if __name__ == "__main__":
    main()
