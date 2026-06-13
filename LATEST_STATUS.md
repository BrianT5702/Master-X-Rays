# Latest Project Status (for new chat context)

**Last updated:** 2026-06-07  
**Repo:** `D:\Master File\Start`  
**Paste this file (or link to it) at the start of a new chat to restore context.**

---

## Goal

Multi-label chest X-ray classification for **two rare lung diseases** on NIH ChestX-ray14:

| Class     | Index | Train positives | Val positives | Test positives |
|-----------|-------|-----------------|---------------|----------------|
| Nodule    | 0     | 5,050           | 649           | 632            |
| Fibrosis  | 1     | 1,392           | 155           | 139            |

**Thesis targets** (2-class, patient split, val-tuned thresholds):

| Tier   | Macro AUROC | Macro F1 | Nodule F1 | Fibrosis F1 |
|--------|-------------|----------|-----------|-------------|
| Strong | ≥ 0.82      | ≥ 0.35   | ≥ 0.40    | ≥ 0.22      |
| Good   | ≥ 0.80      | ≥ 0.28   | ≥ 0.35    | ≥ 0.18      |

Literature reference (ChestX-ray14 per-disease AUROC, not 2-class F1): Nodule/Fibrosis often ~0.78–0.85 mid-tier.

---

## Active model & config

**Primary backbone:** `dense_swin_hybrid` — DenseNet-121 + Swin-T with **gated branch fusion** (~34.9M params).  
**Active config:** `configs/gpu_dense_swin_hybrid_v9.yaml`

| Config | Role | Notes |
|--------|------|-------|
| `gpu_dense_swin_hybrid.yaml` | v6 baseline | Best calibration recipe; val thresholds `[0.8200, 0.8150]` (ep21 best-auc) |
| `gpu_dense_swin_hybrid_v7.yaml` | v7 experiment | Weighted sampling ON → **regression**; thresholds `[0.9300, 0.9350]` |
| `gpu_dense_swin_hybrid_v8.yaml` | v8 experiment | v6 + longer schedule; **logit collapse** (~0.81–0.91 all samples); thresholds `[0.8250, 0.8300]` |
| `gpu_dense_swin_hybrid_v9.yaml` | **current** | Anti-collapse + fibrosis push; **train fresh from ImageNet** (do not resume v8) |

**v9 key hyperparams vs v6:**

- Stronger easy-negative down-weight: `gamma_neg: 2.0` (was 1.5)
- Lower BCE aux / label smoothing: `0.03` / `0.015` (was `0.05` / `0.02`)
- Fibrosis push: `class_weights: [1.0, 5.5]`, `gamma_pos[1]: 2.7`, `clip[1]: 0.16`
- Schedule: 80 epochs, patience 8, 2-epoch warmup, `backbone_lr_mult: 0.10`

---

## Data pipeline

- **Dataset:** NIH ChestX-ray14 (`Data_Entry_2017.csv`)
- **Image size:** 320×320
- **Cache:** `data/processed_320_clahe` (CLAHE preprocessed ROI+mask)
- **Normalization:** `use_normalization: false` — model sees raw `[0,1]` RGB (matches cache; distribution shift vs ImageNet pretrain)
- **Split:** 80/10/10, **patient-level** (no leakage)
  - Train: 89,703 images / 24,644 patients
  - Val: 11,221 / 3,080
  - Test: 11,196 / 3,081
- **ROI:** extraction + mask applied (`use_roi_extraction: true`, `apply_roi_mask: true`)
- **Augmentations:** `legacy_cache_train_pipeline: true` (Resize, no RandomResizedCrop); artifact erases (lower-corner, band, lateral, top, device); flip/rotate/noise

---

## Training setup

- **Loss:** Asymmetric Loss (ASL) + BCE auxiliary + label smoothing
- **Optimizer:** AdamW, LR `3e-5`, discriminative LR (backbone × `backbone_lr_mult`)
- **Batch:** 4 × accumulate 4 = effective 16
- **Precision:** 16-mixed AMP
- **Early stopping:** monitor `val_auc` (max), patience 8
- **F1 metric threshold:** `0.05` (ASL keeps probs modest; do not use 0.5)
- **Per-class thresholds:** `null` in v9 — must tune on val after training

**Hardware (last run):** NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4.3 GB VRAM, Windows, `.venv` Python env.

**VRAM note:** Two full backbones per forward. If OOM: `batch_size: 2`, `accumulate_grad_batches: 24`.

---

## v9 training status (as of last session)

**Log dir:** `lightning_logs/csv_metrics_dense_swin_hybrid/version_9`

| Metric | Value | Notes |
|--------|-------|-------|
| Best val_auc | **0.767** | Epoch 24 |
| val_f1 (logged) | 0.0683 | Flat — no `per_class_thresholds` tuned yet |
| Last epoch | 31 (interrupted) | KeyboardInterrupt at ~12% through epoch 31 |
| val_auc at interrupt | 0.751 | Epoch 31 in progress |

**Important:** v9 config says train **fresh from ImageNet**. Last terminal session used `--resume last.ckpt` (continuing an earlier v9 run). Decide whether to resume or restart fresh before next run.

**Checkpoints (expected path):**
```
lightning_logs/csv_metrics_dense_swin_hybrid/version_9/checkpoints/
  best-auc-*.ckpt
  last.ckpt
```

---

## Post-training workflow

```powershell
# Activate env
& "D:\Master File\Start\.venv\Scripts\Activate.ps1"

# 1. Train (fresh — recommended for v9)
python scripts/train.py --config configs/gpu_dense_swin_hybrid_v9.yaml

# 2. Tune thresholds on val
python scripts/find_threshold.py `
  --checkpoint "lightning_logs/csv_metrics_dense_swin_hybrid/version_9/checkpoints/best-auc-*.ckpt" `
  --config configs/gpu_dense_swin_hybrid_v9.yaml `
  --split val

# 3. Paste printed per_class_thresholds into v9.yaml, then evaluate on test
python scripts/evaluate.py `
  --checkpoint "<same ckpt>" `
  --config configs/gpu_dense_swin_hybrid_v9.yaml `
  --split test
```

**Optional inference boosts (no retrain):**

```powershell
# Horizontal-flip TTA (use same flag on find_threshold and evaluate)
python scripts/find_threshold.py ... --tta-hflip
python scripts/evaluate.py ... --tta-hflip

# Temperature scaling (fit on val first)
python scripts/calibrate_temperature.py --checkpoint <ckpt> --config configs/gpu_dense_swin_hybrid_v9.yaml --split val --out calibration/dense_swin_v9_t.json
python scripts/find_threshold.py ... --calibration calibration/dense_swin_v9_t.json
python scripts/evaluate.py ... --calibration calibration/dense_swin_v9_t.json
```

Existing v6 calibration files (likely stale for v9): `calibration/dense_swin_v6_t.json`, `calibration/dense_swin_v6_t_scalar.json` (temperatures ~2000+ — severe miscalibration signal).

---

## Known issues & lessons

1. **Logit collapse (v8):** All samples scored ~0.81–0.91 → F1 only works in a narrow threshold band (~0.82–0.83). v9 targets this via stronger `gamma_neg`, lower BCE aux/smoothing.
2. **Weighted sampling hurts (v7):** `use_weighted_sampling: true` caused regression; keep `false` for hybrid runs.
3. **Logged F1 vs tuned F1:** Training logs `val_f1` at `f1_metric_threshold: 0.05` without per-class tuning — expect ~0.06–0.07 until `find_threshold.py` is run. Real F1 comes from val-tuned `per_class_thresholds`.
4. **Accuracy ~3.4% is normal** at default thresholds when ASL keeps negative probs low; use same threshold as F1 metric.
5. **Old concat-only hybrid checkpoints incompatible** with gated fusion — always train from ImageNet weights for architecture changes.

---

## Recent code changes (uncommitted in git)

| Area | Files | What |
|------|-------|------|
| Calibration | `src/training/calibration.py`, `scripts/calibrate_temperature.py` | Post-hoc temperature scaling (scalar or per-class) |
| TTA | `src/training/tta.py` | `--tta-hflip` for evaluate / find_threshold |
| Threshold search | `scripts/find_threshold.py` | Per-class grid, calibration + TTA support |
| Evaluation | `scripts/evaluate.py` | Calibration + TTA; passes calibration to heatmap gen |
| Training | `scripts/train.py`, `src/training/lightning_module.py` | Resume, discriminative LR, per-class metrics |
| XAI | `src/xai/gradcam.py` | Hybrid backbone target layers, calibration wrapper |
| Preprocessing | `scripts/preprocess_images.py`, `scripts/organize_data_by_label.py` | Cache pipeline utilities |
| Configs | v7/v8/v9 yaml, `gpu_dense_swin_hybrid.yaml`, `gpu.yaml` | Experiment lineage |

---

## Key file map

```
configs/
  gpu_dense_swin_hybrid_v9.yaml   ← active experiment
  gpu_dense_swin_hybrid.yaml      ← v6 baseline
  gpu.yaml                        ← DenseNet-only GPU config (separate track)

src/
  models/dense_swin_hybrid.py     ← gated DenseNet+Swin fusion
  training/lightning_module.py    ← RareDiseaseModule, ASL loss, metrics
  training/calibration.py         ← temperature scaling
  training/tta.py                 ← h-flip TTA
  data/datasets.py                ← NIH dataset, augmentations, cache
  xai/gradcam.py                  ← Grad-CAM for hybrid backbone

scripts/
  train.py
  find_threshold.py
  evaluate.py
  calibrate_temperature.py
  preprocess_images.py
  generate_heatmaps.py

data/processed_320_clahe/         ← preprocessed image cache
calibration/                      ← temperature JSON outputs
lightning_logs/csv_metrics_dense_swin_hybrid/version_9/  ← current run logs
```

---

## What to do next

1. **Finish or restart v9 training** — target val_auc ≥ 0.80 before threshold tuning matters much.
2. **Run `find_threshold.py` on val** with best-auc checkpoint; paste `per_class_thresholds` into v9.yaml.
3. **Evaluate on test** — check macro AUROC, macro F1, per-class F1 against thesis targets.
4. **Try TTA + calibration** if raw metrics are close but F1 is threshold-sensitive.
5. **If v9 still collapses:** compare logit/probability histograms to v6; consider reverting toward v6 ASL params with only fibrosis-specific tweaks.

---

## Quick prompt for a new chat

> I'm working on a DenseNet+Swin hybrid for NIH ChestX-ray14 Nodule/Fibrosis detection. Read `LATEST_STATUS.md` for full context. Active config is `configs/gpu_dense_swin_hybrid_v9.yaml`. v9 best val_auc was 0.767 at epoch 24; training interrupted at epoch 31. Help me with: [your task here].
