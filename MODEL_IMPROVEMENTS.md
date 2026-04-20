# Model improvement guide

This document summarizes changes made to improve the rare lung disease (Nodule / Fibrosis) model and what to try next.

## Changes made

### 1. **Accuracy metric uses same threshold as F1**
- **Issue:** Val/test accuracy used the default 0.5 threshold while ASL keeps probabilities modest (often 0.05–0.2 for negatives). That made accuracy look very low (~3.4%) even when the model had reasonable AUC (~0.72).
- **Fix:** `val_acc` and `test_acc` now use the same threshold as F1 (`f1_metric_threshold` or `prediction_threshold` from config). Logged accuracy is now comparable to F1 and reflects the same decision boundary.

### 2. **F1 threshold passed for new runs**
- **Issue:** `f1_metric_threshold` was only passed when resuming from checkpoint, so new runs used only `prediction_threshold` for the F1 metric.
- **Fix:** `f1_metric_threshold` is now passed when creating a new `RareDiseaseModule`, so configs like `gpu.yaml` (with `f1_metric_threshold: 0.05`) apply from the start.

### 3. **Updates to `configs/gpu.yaml`**
- **Weighted sampling:** `use_weighted_sampling: true` to oversample rare positives (Nodule/Fibrosis) and improve recall.
- **Longer training:** `max_epochs: 55`, `early_stopping_patience: 10`.
- Same anti-shortcut augmentations (lower-corner, lateral-edge, top-edge erase), ASL + BCE aux + label smoothing, and gradient clipping.

### 4. **EfficientNet-B0 backbone option**
- **Code:** `build_backbone` and Grad-CAM support `efficientnet_b0`.
- **Usage:** In any config, set `model.backbone: efficientnet_b0` to try a different capacity/feature set.

### 5. **Per-class thresholds (F1 / accuracy)**
- **Issue:** One threshold for both Nodule and Fibrosis often yields flat macro F1 across a grid search.
- **Fix:** `training.per_class_thresholds: [t_nodule, t_fibrosis]` in `configs/gpu.yaml`. The Lightning module logs `val_f1_nodule`, `val_f1_fibrosis`, `val_f1` (macro of the two at those thresholds), and matching accuracies. Run `scripts/find_threshold.py` — it prints a per-class table and a suggested YAML line.

### 6. **Heatmaps: clearer PNGs + correct sample count**
- Each batch item includes `image_index` (NIH filename). Saved figures use a **suptitle** (image id + target class). `--num_samples` now means **number of X-rays**, not number of PNG files (each X-ray still gets one PNG per class target).

## What to do next

1. **Train with the GPU config**
   ```bash
   python scripts/train.py --config configs/gpu.yaml
   ```
   This uses weighted sampling, anti-shortcut augmentations, and ASL + aux loss.

2. **Find thresholds after training**
   ```bash
   python scripts/find_threshold.py --checkpoint "lightning_logs/csv_metrics/version_XX/checkpoints/best-auc-*.ckpt" --config configs/gpu.yaml --split val
   ```
   Copy the printed `per_class_thresholds: [a, b]` into `training` in `gpu.yaml`. Optionally tune `prediction_threshold` for inference defaults.

3. **Try EfficientNet**
   - In `configs/gpu.yaml` set `model.backbone: efficientnet_b0`, then train. Compare val/test AUC and F1 to DenseNet-121.

4. **If AUC is still below target (~0.80+)**
   - **Data:** Ensure `cache_dir` is set and preprocessed (ROI + mask) so training is stable and consistent.
   - **Class weights:** Tune `model.class_weights` (e.g. `[1.0, 4.0]` or `[1.0, 5.0]`) and re-run threshold search.
   - **Loss:** Try `loss.name: focal` with `alpha: 0.75`, `gamma: 2.0` as an alternative to ASL.
   - **Resolution:** If GPU memory allows, try `image_size: [384, 384]` for finer detail (and adjust batch size / accumulation).

## Quick reference: config files

| Config            | Use case                                                      |
|-------------------|---------------------------------------------------------------|
| `configs/base.yaml` | Baseline, no artifact erasure                                 |
| `configs/gpu.yaml`  | GPU-optimized, anti-shortcut aug, ASL + aux, weighted sampling |
