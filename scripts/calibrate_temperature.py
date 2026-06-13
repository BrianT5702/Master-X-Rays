"""Fit temperature scaling on validation logits (multi-label BCE NLL). Saves JSON for find_threshold / evaluate.

Example (replace the checkpoint path with your real .ckpt file — not the literal word "..."):
  python scripts/calibrate_temperature.py \\
    --checkpoint lightning_logs/csv_metrics_dense_swin_hybrid/version_6/checkpoints/best-auc-epoch=21-val_auc=0.7642.ckpt \\
    --config configs/gpu_dense_swin_hybrid.yaml \\
    --out calibration/dense_swin_v6.json

Then:
  python scripts/find_threshold.py --checkpoint <same.ckpt> --config ... --calibration calibration/dense_swin_v6.json
  python scripts/evaluate.py --checkpoint <same.ckpt> --config ... --calibration calibration/dense_swin_v6.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
from torch.nn import functional as F

from src.data.datasets import create_dataloaders, resolve_data_normalization
from src.models.basemodels import build_backbone
from src.training.calibration import calibration_dict, fit_temperature, save_calibration_json
from src.training.lightning_module import RareDiseaseModule
from torchmetrics.classification import MultilabelAUROC


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit temperature scaling on val split; write JSON sidecar.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/gpu.yaml"))
    p.add_argument("--out", type=Path, required=True, help="Output JSON path.")
    p.add_argument(
        "--mode",
        choices=["scalar", "per_class"],
        default="per_class",
        help="scalar: one T; per_class: one T per label (default).",
    )
    p.add_argument("--max-iter", type=int, default=80, help="LBFGS max_iter per step.")
    return p.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    if not ckpt_path.is_file():
        print(f"Error: checkpoint is not a readable file: {ckpt_path}")
        print(
            "The --checkpoint argument must be the real path to your .ckpt file.\n"
            "Do not use `...` as a placeholder — paste the full path, for example:\n"
            r'  lightning_logs\csv_metrics_dense_swin_hybrid\version_6\checkpoints\best-auc-epoch=21-val_auc=0.7642.ckpt'
        )
        sys.exit(1)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.is_file():
        print(f"Error: config not found: {cfg_path}")
        sys.exit(1)

    config = load_config(cfg_path)
    data_cfg = config["data"]
    training_cfg = config["training"]
    model_cfg = config["model"]
    loss_cfg = config.get("loss", {})

    label_columns = ["Nodule", "Fibrosis"]
    images_root = Path(data_cfg["nih_root"])
    csv_path = Path(data_cfg["csv_path"])
    cache_dir = data_cfg.get("cache_dir")
    if cache_dir:
        cache_dir = Path(cache_dir)

    dataloaders = create_dataloaders(
        csv_path=csv_path,
        images_root=images_root,
        label_columns=label_columns,
        image_size=tuple(data_cfg["image_size"]),
        batch_size=training_cfg["batch_size"],
        splits=(data_cfg["train_split"], data_cfg["val_split"], data_cfg["test_split"]),
        num_workers=data_cfg["num_workers"],
        seed=config["experiment"]["seed"],
        augmentation_config=config.get("augmentations", {}),
        use_weighted_sampling=False,
        patient_split=data_cfg.get("patient_split", True),
        use_roi_extraction=data_cfg.get("use_roi_extraction", True),
        apply_roi_mask=data_cfg.get("apply_roi_mask", True),
        use_normalization=data_cfg.get("use_normalization", False),
        cache_dir=cache_dir,
        normalization=resolve_data_normalization(
            data_cfg.get("normalization"), model_cfg["backbone"]
        ),
    )
    loader = dataloaders["val"]

    backbone = build_backbone(
        model_cfg["backbone"],
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
        dropout=model_cfg.get("dropout", 0.0),
    )
    module = RareDiseaseModule.load_from_checkpoint(
        str(ckpt_path),
        model=backbone,
        learning_rate=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
        class_weights=model_cfg.get("class_weights"),
        loss_config=loss_cfg,
        max_epochs=training_cfg["max_epochs"],
        prediction_threshold=training_cfg.get("prediction_threshold", 0.3),
        samples_per_class=model_cfg.get("samples_per_class"),
        warmup_epochs=training_cfg.get("warmup_epochs", 0),
        f1_metric_threshold=training_cfg.get("f1_metric_threshold", 0.05),
        per_class_thresholds=training_cfg.get("per_class_thresholds"),
        backbone_lr_mult=float(training_cfg.get("backbone_lr_mult", 1.0)),
        map_location="cpu",
    )
    module.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device)

    n_batches = len(loader)
    print(
        f"Collecting val logits: {n_batches} batches "
        f"(batch_size={training_cfg['batch_size']}, device={device})..."
    )
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            images = batch["image"].to(device)
            labels = batch["labels"]
            logits = module(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
            if (i + 1) % 50 == 0 or (i + 1) == n_batches:
                print(f"  ... batch {i + 1}/{n_batches}")
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).int()

    probs_uncal = torch.sigmoid(logits)
    auc0 = MultilabelAUROC(num_labels=logits.shape[1], average="macro")
    print(f"Macro AUC (uncalibrated probs, threshold-free): {auc0(probs_uncal, labels).item():.4f}")

    T = fit_temperature(logits, labels, args.mode, device, max_iter=args.max_iter)
    logits_cal = logits / T.view(1, -1)
    nll_before = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="mean").item()
    nll_after = F.binary_cross_entropy_with_logits(logits_cal, labels.float(), reduction="mean").item()
    print(f"Val NLL (BCE logits): before={nll_before:.6f}  after={nll_after:.6f}")
    print(f"Fitted temperature(s) ({args.mode}): {T.squeeze().tolist()}")

    probs_cal = torch.sigmoid(logits_cal)
    auc1 = MultilabelAUROC(num_labels=logits.shape[1], average="macro")
    print(f"Macro AUC (calibrated probs): {auc1(probs_cal, labels).item():.4f}  (should match uncalibrated)")

    spreads = [(probs_cal[:, c].max() - probs_cal[:, c].min()).item() for c in range(probs_cal.shape[1])]
    print(f"Per-class calibrated prob spread (max - min): {spreads}")
    min_spread = min(spreads)
    if min_spread < 0.02:
        print(
            "\nERROR: Calibration collapsed — probabilities are nearly constant (~0.5) per class.\n"
            "BCE can be minimized there, but thresholds become meaningless. JSON was NOT written.\n"
            "Use uncalibrated scores instead: run find_threshold.py and evaluate.py without --calibration.\n",
            flush=True,
        )
        sys.exit(1)

    out_path = args.out
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    payload = calibration_dict(args.mode, T)
    save_calibration_json(out_path, payload)
    print(f"Wrote {out_path}")
    print("\nNext: find_threshold.py --calibration <that.json>, then evaluate.py --calibration <that.json>")


if __name__ == "__main__":
    main()
