# Training Process Explanation

## 📊 Data Split (80/10/10)

Your dataset is split into 3 parts:

1. **Training Set (80%)** - ~89,696 images
   - Used to **train** the model (update weights)
   - Model learns patterns from these images
   - **Augmentations applied** (flip, rotate, etc.)

2. **Validation Set (10%)** - ~11,212 images
   - Used to **monitor** training progress
   - **NO weight updates** - just evaluation
   - Shows if model is learning or overfitting
   - Used to select best checkpoints

3. **Test Set (10%)** - ~11,212 images
   - **ONLY used at the END** after training completes
   - Final evaluation to see real-world performance
   - Never touched during training

## 🔄 What Happens Each Epoch

### Epoch 0 (Special):
1. **Validation First** (pretrained model baseline)
2. **Then Training** (update weights)

### Epochs 1-24 (Normal):
1. **Training Phase**:
   - Process all training batches (~3,737 batches)
   - Update model weights
   - Compute training metrics (train_loss, train_f1)

2. **Validation Phase**:
   - Process all validation batches (~467 batches)
   - **NO weight updates** - just evaluation
   - Compute validation metrics (val_loss, val_f1, val_auc)
   - These metrics determine "best" checkpoints

3. **Test Phase**:
   - **NOT RUN** during training
   - Only runs once at the very end

## 📈 Why Metrics Drop After Epoch 0/5?

### Problem: Overfitting
- Model learns training data too well
- Stops generalizing to new data
- Validation metrics get worse even though training metrics improve

### Solutions Applied:
1. **Lower Learning Rate**: 0.0001 → 0.00005 (more stable)
2. **More Regularization**: 
   - Dropout: 0.2 → 0.3 (prevents overfitting)
   - Weight Decay: 0.01 → 0.02 (stronger regularization)
3. **Faster Warmup**: 5 → 3 epochs (reach full LR faster, then stabilize)

## 🎯 What to Expect Now

With the fixes:
- **Metrics should improve more gradually**
- **Less overfitting** (validation metrics stay closer to training)
- **Best metrics should appear later** (epochs 8-15 instead of 0-5)
- **More stable training** (less jumping around)

## 📝 Key Points

1. **Test set is NOT used during training** - only at the end
2. **Validation is used to monitor** - helps detect overfitting
3. **Best checkpoints are saved** - based on validation metrics
4. **Overfitting is normal** - but we try to minimize it

## 🔍 How to Read Training Logs

```
Epoch 0:
  val_f1=0.65, val_auc=0.72  ← Pretrained model baseline

Epoch 1-3:
  train_loss decreasing
  val_f1 improving
  val_auc improving
  ← Warmup phase (LR increasing)

Epoch 4-15:
  train_loss still decreasing
  val_f1/val_auc should improve or stay stable
  ← Main learning phase

Epoch 16-24:
  train_loss continues decreasing
  val_f1/val_auc may plateau or slightly decrease
  ← Fine-tuning phase
```

## ✅ What Changed

1. **Learning Rate**: 0.0001 → 0.00005 (50% reduction)
2. **Dropout**: 0.2 → 0.3 (50% increase)
3. **Weight Decay**: 0.01 → 0.02 (100% increase)
4. **Warmup**: 5 → 3 epochs (faster warmup)

These changes should help the model learn more gradually and prevent early overfitting.








