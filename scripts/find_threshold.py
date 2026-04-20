"""Find optimal prediction threshold for F1 on validation (or test) set.

After a coarse grid, refines each class in ±fine-window with fine-step (default 0.005)
unless --no-refine. Prints precision/recall at the chosen per-class thresholds.

Example:
  py scripts/find_threshold.py --checkpoint path/to/best.ckpt --config configs/gpu_efficientnet_b2_shortcut_lite.yaml
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

from src.data.datasets import create_dataloaders, resolve_data_normalization
from src.models.basemodels import build_backbone
from src.training.lightning_module import RareDiseaseModule
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    MultilabelAUROC,
    MultilabelF1Score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find threshold that maximizes macro F1 on val (or test) set."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .ckpt file.")
    parser.add_argument("--config", type=Path, default=Path("configs/gpu.yaml"), help="Config used for training.")
    parser.add_argument("--split", choices=["val", "test"], default="val",
                        help="Split to search threshold on (default: val).")
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
        help="Comma-separated thresholds to try (default: 0.05..0.95).",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Disable fine per-class search (default: refine ±window around coarse best in 0.005 steps).",
    )
    parser.add_argument(
        "--fine-step",
        type=float,
        default=0.005,
        help="Step size for per-class refinement (default 0.005).",
    )
    parser.add_argument(
        "--fine-window",
        type=float,
        default=0.06,
        help="Half-width around coarse best threshold for refinement (default 0.06).",
    )
    return parser.parse_args()


def _best_f1_threshold_1d(
    probs_1d: torch.Tensor,
    labels_1d: torch.Tensor,
    thresholds: list[float],
) -> tuple[float, float]:
    best_f1, best_t = -1.0, thresholds[0]
    for t in thresholds:
        m = BinaryF1Score(threshold=t)
        m.update(probs_1d, labels_1d)
        f1_val = m.compute().item()
        if f1_val > best_f1:
            best_f1 = f1_val
            best_t = t
    return best_t, best_f1


def _refine_threshold_f1(
    probs_1d: torch.Tensor,
    labels_1d: torch.Tensor,
    coarse_t: float,
    fine_step: float,
    half_width: float,
) -> tuple[float, float]:
    lo = max(0.0, coarse_t - half_width)
    hi = min(1.0, coarse_t + half_width)
    fine_list: list[float] = []
    t = lo
    while t <= hi + 1e-9:
        fine_list.append(float(round(t, 6)))
        t += fine_step
    if not fine_list:
        fine_list = [float(coarse_t)]
    if not any(abs(x - coarse_t) < 1e-5 for x in fine_list):
        fine_list.append(float(coarse_t))
    return _best_f1_threshold_1d(probs_1d, labels_1d, sorted(set(fine_list)))


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
    loader = dataloaders[args.split]

    backbone = build_backbone(
        model_cfg["backbone"],
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
        dropout=model_cfg.get("dropout", 0.0),
    )
    module = RareDiseaseModule.load_from_checkpoint(
        str(args.checkpoint),
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

    # Collect all logits and labels (no Lightning progress bar — can look "stuck" on large val sets)
    n_batches = len(loader)
    print(
        f"Scoring {args.split} split: {n_batches} batches "
        f"(batch_size={training_cfg['batch_size']}, device={device}) — this may take several minutes..."
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

    probs = torch.sigmoid(logits)
    num_classes = logits.shape[1]

    # AUC is threshold-independent
    auroc = MultilabelAUROC(num_labels=num_classes, average="macro")
    auc_val = auroc(probs, labels).item()
    print(f"Split: {args.split}  |  Macro AUC (threshold-independent): {auc_val:.4f}\n")

    thresholds = [float(x.strip()) for x in args.thresholds.split(",")]
    print("Threshold  |  Macro F1")
    print("-----------|----------")
    best_f1 = -1.0
    best_thr = None
    for t in sorted(thresholds):
        f1_metric = MultilabelF1Score(num_labels=num_classes, average="macro", threshold=t)
        f1_val = f1_metric(probs, labels).item()
        print(f"  {t:.2f}     |  {f1_val:.4f}")
        if f1_val > best_f1:
            best_f1 = f1_val
            best_thr = t
    print("-----------|----------")
    print(f"\nBest single threshold for multilabel macro F1: {best_thr}  (F1 = {best_f1:.4f})")
    print("(Same threshold on both labels — often suboptimal when Nodule vs Fibrosis behave differently.)\n")

    # Per-class thresholds → paste into training.per_class_thresholds in your training config
    label_names = ["Nodule", "Fibrosis"]
    suggested: list[float] = []
    print("--- Per-class F1 (independent threshold per label) ---")
    for ci, name in enumerate(label_names):
        pcol = probs[:, ci]
        ycol = labels[:, ci]
        best_cf1, best_ct = -1.0, sorted(thresholds)[0]
        print(f"\n{name}:")
        print("Thr   | F1")
        print("------|-------")
        for t in sorted(thresholds):
            m = BinaryF1Score(threshold=t)
            m.update(pcol, ycol)
            f1_val = m.compute().item()
            print(f"{t:.2f}  | {f1_val:.4f}")
            if f1_val > best_cf1:
                best_cf1 = f1_val
                best_ct = t
        if not args.no_refine:
            best_ct, best_cf1 = _refine_threshold_f1(
                pcol, ycol, best_ct, args.fine_step, args.fine_window
            )
        print(f"  => Chosen threshold={best_ct:.4f}, F1={best_cf1:.4f}")

        pr_m = BinaryPrecision(threshold=best_ct)
        rc_m = BinaryRecall(threshold=best_ct)
        pr_m.update(pcol, ycol)
        rc_m.update(pcol, ycol)
        print(
            f"  => At chosen thr: precision={pr_m.compute().item():.4f}, "
            f"recall={rc_m.compute().item():.4f}"
        )
        suggested.append(best_ct)

    m0 = BinaryF1Score(threshold=suggested[0])
    m0.update(probs[:, 0], labels[:, 0])
    m1 = BinaryF1Score(threshold=suggested[1])
    m1.update(probs[:, 1], labels[:, 1])
    macro_pc = (m0.compute() + m1.compute()) / 2.0
    print(f"\nMacro F1 using best per-class thresholds {suggested}: {macro_pc.item():.4f}")
    cfg_rel = args.config.as_posix() if hasattr(args.config, "as_posix") else str(args.config)
    print(f"Add to {cfg_rel} under training:")
    print(
        f"  per_class_thresholds: [{suggested[0]:.4f}, {suggested[1]:.4f}]  # [Nodule, Fibrosis]"
    )


if __name__ == "__main__":
    main()
