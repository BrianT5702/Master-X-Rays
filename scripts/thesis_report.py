"""Chapter-4 metrics: AUROC, F1, precision, recall, specificity, confusion counts per class.

Requires per_class_thresholds in config (from find_threshold.py on val). Example:

  py scripts/thesis_report.py \\
    --checkpoint lightning_logs/.../best-auc-epoch=24-val_auc=0.7670.ckpt \\
    --config configs/gpu_dense_swin_hybrid_v9.yaml \\
    --split test \\
    --out results/thesis_v9_test.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
    MultilabelAUROC,
)

from src.data.datasets import create_dataloaders, resolve_data_normalization
from src.models.basemodels import build_backbone
from src.training.calibration import load_calibration_json, temperatures_tensor_from_dict
from src.training.lightning_module import RareDiseaseModule
from src.training.tta import HFlipLogitsTTA

LABELS = ("Nodule", "Fibrosis")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Thesis-ready per-class metrics at tuned thresholds.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--calibration", type=Path, default=None)
    p.add_argument("--tta-hflip", action="store_true")
    p.add_argument("--out", type=Path, default=None, help="Optional CSV path for thesis table.")
    return p.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def confusion_counts(probs_1d: torch.Tensor, labels_1d: torch.Tensor, thr: float) -> dict[str, int]:
    pred = probs_1d >= thr
    y = labels_1d.bool()
    tp = int((pred & y).sum().item())
    tn = int((~pred & ~y).sum().item())
    fp = int((pred & ~y).sum().item())
    fn = int((~pred & y).sum().item())
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def main() -> None:
    args = parse_args()
    ckpt = args.checkpoint if args.checkpoint.is_absolute() else PROJECT_ROOT / args.checkpoint
    cfg_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    if not ckpt.is_file():
        print(f"Error: checkpoint not found: {ckpt}")
        sys.exit(1)
    if not cfg_path.is_file():
        print(f"Error: config not found: {cfg_path}")
        sys.exit(1)

    config = load_config(cfg_path)
    data_cfg = config["data"]
    training_cfg = config["training"]
    model_cfg = config["model"]
    loss_cfg = config.get("loss", {})

    thresholds = training_cfg.get("per_class_thresholds")
    if not thresholds or len(thresholds) != 2:
        print(
            "Error: training.per_class_thresholds missing in config.\n"
            "Run find_threshold.py on val first, then paste thresholds into the yaml."
        )
        sys.exit(1)
    thr_list = [float(t) for t in thresholds]

    cache_dir = data_cfg.get("cache_dir")
    if cache_dir:
        cache_dir = Path(cache_dir)

    loader = create_dataloaders(
        csv_path=Path(data_cfg["csv_path"]),
        images_root=Path(data_cfg["nih_root"]),
        label_columns=list(LABELS),
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
    )[args.split]

    backbone = build_backbone(
        model_cfg["backbone"],
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
        dropout=model_cfg.get("dropout", 0.0),
    )
    module = RareDiseaseModule.load_from_checkpoint(
        str(ckpt),
        model=backbone,
        learning_rate=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
        class_weights=model_cfg.get("class_weights"),
        loss_config=loss_cfg,
        max_epochs=training_cfg["max_epochs"],
        prediction_threshold=training_cfg.get("prediction_threshold", 0.3),
        warmup_epochs=training_cfg.get("warmup_epochs", 0),
        f1_metric_threshold=training_cfg.get("f1_metric_threshold", 0.05),
        per_class_thresholds=thr_list,
        backbone_lr_mult=float(training_cfg.get("backbone_lr_mult", 1.0)),
        map_location="cpu",
    )
    module.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device)
    if args.tta_hflip:
        module.model = HFlipLogitsTTA(module.model)

    print(f"Scoring {args.split} ({len(loader)} batches)...")
    logits_list, labels_list = [], []
    with torch.no_grad():
        for batch in loader:
            logits_list.append(module(batch["image"].to(device)).cpu())
            labels_list.append(batch["labels"])
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0).int()

    if args.calibration:
        cal_path = args.calibration if args.calibration.is_absolute() else PROJECT_ROOT / args.calibration
        cal = load_calibration_json(cal_path)
        T = temperatures_tensor_from_dict(cal)
        logits = logits / T.view(1, -1)

    probs = torch.sigmoid(logits)
    macro_auc = MultilabelAUROC(num_labels=2, average="macro")(probs, labels).item()

    print("\n" + "=" * 72)
    print(f"THESIS METRICS — {args.split.upper()} — {config['experiment'].get('name', 'model')}")
    print(f"Checkpoint: {ckpt.name}")
    print(f"Thresholds [Nodule, Fibrosis]: {thr_list}")
    print("=" * 72)

    rows: list[dict[str, str | float | int]] = []
    for ci, name in enumerate(LABELS):
        p = probs[:, ci]
        y = labels[:, ci]
        thr = thr_list[ci]
        pos_mask = y == 1
        neg_mask = ~pos_mask
        print(f"\n--- {name} ---")
        pos_mean = p[pos_mask].mean().item() if pos_mask.any() else float("nan")
        neg_mean = p[neg_mask].mean().item() if neg_mask.any() else float("nan")
        print(
            f"  Prob: min={p.min():.4f} max={p.max():.4f} mean={p.mean():.4f} "
            f"pos_mean={pos_mean:.4f} neg_mean={neg_mean:.4f}"
        )

        auc = BinaryAUROC()(p, y).item()
        f1 = BinaryF1Score(threshold=thr)(p, y).item()
        prec = BinaryPrecision(threshold=thr)(p, y).item()
        rec = BinaryRecall(threshold=thr)(p, y).item()
        spec = BinarySpecificity(threshold=thr)(p, y).item()
        counts = confusion_counts(p, y, thr)

        print(f"  AUROC:       {auc:.4f}")
        print(f"  Threshold:   {thr:.4f}")
        print(f"  F1:          {f1:.4f}")
        print(f"  Precision:   {prec:.4f}")
        print(f"  Recall:      {rec:.4f}  (sensitivity — catch real cases)")
        print(f"  Specificity: {spec:.4f}  (correctly clear negatives)")
        print(f"  Confusion:   TP={counts['TP']} TN={counts['TN']} FP={counts['FP']} FN={counts['FN']}")

        rows.append(
            {
                "split": args.split,
                "class": name,
                "threshold": thr,
                "auroc": round(auc, 4),
                "f1": round(f1, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "specificity": round(spec, 4),
                "TP": counts["TP"],
                "TN": counts["TN"],
                "FP": counts["FP"],
                "FN": counts["FN"],
            }
        )

    f1s = [r["f1"] for r in rows]
    macro_f1 = sum(f1s) / len(f1s)
    print(f"\n--- MACRO ---")
    print(f"  Macro AUROC: {macro_auc:.4f}")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print("=" * 72)

    collapse_ok = all(
        probs[:, ci][labels[:, ci] == 0].mean().item() < 0.45
        for ci in range(2)
        if (labels[:, ci] == 0).any()
    )
    if collapse_ok:
        print("Probability check: PASS (negative mean < 0.45 for both classes)")
    else:
        print(
            "Probability check: FAIL — scores still bunched high on negatives. "
            "Train v9b or raise gamma_neg before trusting heatmaps."
        )

    if args.out:
        out_path = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
            f.write(f"\nmacro_auroc,{macro_auc:.4f}\nmacro_f1,{macro_f1:.4f}\n")
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
