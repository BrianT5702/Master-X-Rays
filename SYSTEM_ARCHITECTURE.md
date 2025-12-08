# System Architecture Documentation
## Rare Lung Disease Detection (Nodules and Fibrosis) in Chest X-Rays

---

## 1. System Overview

This document describes the complete operational framework for improving the detection of pulmonary nodules and pulmonary fibrosis—two highly underrepresented and visually subtle conditions in the ChestX-ray14 dataset. The system implements a deep learning pipeline with class-imbalance handling and explainable AI capabilities.

### 1.1 Objectives

The system is designed to:
1. Effectively preprocess incoming raw chest X-ray images
2. Manage extreme class imbalance during model training
3. Extract features using robust deep learning backbones
4. Generate reliable predictions for multi-label classification
5. Produce interpretable heatmaps to support transparency and reliability

### 1.2 Target Diseases

- **Pulmonary Nodules**: Small, round growths in the lungs
- **Pulmonary Fibrosis**: Scarring of lung tissue

Both conditions are rare and visually subtle, making them challenging for automated detection.

---

## 2. System Architecture Diagram

```
                +------------------------------------------------+
                |              Input X-Ray Image                 |
                |      (Raw frontal-view CXR from dataset)       |
                |         Format: PNG, various resolutions        |
                +------------------------------------------------+
                                  |
                                  v
                +------------------------------------------------+
                |              Preprocessing Module              |
                |  Location: src/data/datasets.py                |
                |                                                 |
                |  Operations:                                   |
                |  ✓ Resize to 224×224 pixels                    |
                |  ✓ Grayscale → 3-channel replication            |
                |  ✓ Normalization (ImageNet mean/std)           |
                |  ✓ Label vector encoding (binary)              |
                |  ✓ Data augmentation:                          |
                |    - Horizontal flip (prob: 0.5)                |
                |    - Rotation (±7.5 degrees)                   |
                |    - Random noise (std: 0.01)                  |
                |  ✓ Train/Val/Test splitting (80/10/10)         |
                +------------------------------------------------+
                                  |
                                  v
                +------------------------------------------------+
                |        Imbalance-Aware Training Module         |
                |  Location: src/training/lightning_module.py    |
                |                                                 |
                |  Loss Functions (configurable):                |
                |  ✓ Weighted BCE                                |
                |  ✓ Class-Balanced Loss                         |
                |  ✓ Focal Loss (default)                        |
                |                                                 |
                |  Data-Level Techniques:                         |
                |  ✓ Batch-balanced sampling (optional)           |
                |  ✓ Class weights: [16.71, 65.50]               |
                |    [Nodule, Fibrosis]                          |
                |                                                 |
                |  (Minority-sensitive optimisation)              |
                +------------------------------------------------+
                                  |
                                  v
                +------------------------------------------------+
                |          DenseNet-121 Feature Extractor        |
                |  Location: src/models/basemodels.py             |
                |                                                 |
                |  Architecture:                                 |
                |  ✓ Dense blocks (4 blocks)                      |
                |  ✓ Transition layers                           |
                |  ✓ Global average pooling                      |
                |  ✓ Fully-connected classifier (2 outputs)     |
                |                                                 |
                |  Alternatives:                                 |
                |  - ResNet-50                                   |
                |  - Swin Transformer (Swin-T)                   |
                |                                                 |
                |  Pretrained: ImageNet weights                    |
                +------------------------------------------------+
                                  |
                                  v
                +------------------------------------------------+
                |              Multi-Label Prediction            |
                |  Location: src/training/lightning_module.py    |
                |                                                 |
                |  Output:                                       |
                |  ✓ Sigmoid activation for 2 classes            |
                |  ✓ Independent binary predictions             |
                |  ✓ Probability scores: [P(Nodule), P(Fibrosis)]|
                |                                                 |
                |  Metrics:                                       |
                |  ✓ Multilabel F1-Score (macro)                 |
                |  ✓ Multilabel AUROC (macro)                    |
                |  ✓ Multilabel Accuracy (macro)                |
                +------------------------------------------------+
                                  |
                                  v
                +------------------------------------------------+
                |             Explainable AI Module             |
                |  Location: src/xai/gradcam.py                   |
                |                                                 |
                |  Methods:                                      |
                |  ✓ Grad-CAM                                    |
                |  ✓ Grad-CAM++ (enhanced)                       |
                |                                                 |
                |  Output:                                       |
                |  ✓ Heatmap generation                         |
                |  ✓ Activation region visualisation            |
                |  ✓ Overlay on original images                 |
                |  ✓ Interpretability output                    |
                +------------------------------------------------+
                                  |
                                  v
                +------------------------------------------------+
                |               Final Output                     |
                |                                                 |
                |  Predictions:                                  |
                |  - Probability scores for each class           |
                |  - Binary predictions (threshold: 0.3)         |
                |                                                 |
                |  Visualizations:                               |
                |  - Heatmaps showing model focus regions        |
                |  - Overlaid heatmaps on original images       |
                |  - Class-specific attention maps              |
                +------------------------------------------------+
```

---

## 3. Component Details

### 3.1 Preprocessing Module (`src/data/datasets.py`)

**Purpose**: Transform raw chest X-ray images into standardized, normalized tensors ready for model input.

**Key Functions**:
- `NIHChestXRayDataset`: PyTorch Dataset class for NIH ChestX-ray14
- `create_dataloaders()`: Factory function for creating train/val/test dataloaders

**Preprocessing Steps**:

1. **Image Loading**
   - Load PNG images from NIH dataset structure
   - Convert to RGB format (3 channels)
   - Handle various subdirectory structures

2. **Resizing**
   - Resize all images to 224×224 pixels (standard for ImageNet-pretrained models)
   - Maintains aspect ratio during resize

3. **Normalization**
   - Apply ImageNet statistics:
     - Mean: [0.485, 0.456, 0.406]
     - Std: [0.229, 0.224, 0.225]
   - Converts pixel values to normalized range

4. **Label Encoding**
   - Extract binary labels from "Finding Labels" column
   - Create binary vectors: [Nodule, Fibrosis]
   - Example: "Nodule|Fibrosis" → [1, 1]

5. **Data Augmentation** (training only)
   - Horizontal flip: 50% probability
   - Rotation: ±7.5 degrees
   - Random noise: Gaussian noise with std=0.01

6. **Data Splitting**
   - Train: 80%
   - Validation: 10%
   - Test: 10%
   - Random seed: 42 (reproducible)

**Configuration** (in `configs/base.yaml`):
```yaml
data:
  image_size: [224, 224]
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1

augmentations:
  horizontal_flip_prob: 0.5
  rotation_degrees: 7.5
  random_noise_std: 0.01
```

---

### 3.2 Imbalance-Aware Training Module (`src/training/lightning_module.py`)

**Purpose**: Handle extreme class imbalance through multiple loss functions and sampling strategies.

#### 3.2.1 Loss Functions

**1. Focal Loss** (Default)
- **Formula**: FL = α_t * (1 - p_t)^γ * BCE
- **Parameters**:
  - `alpha`: 0.75 (weighting factor for rare class)
  - `gamma`: 2.0 (focusing parameter)
- **Advantage**: Automatically down-weights easy examples, focuses on hard examples
- **Usage**: Combined with class weights for additional rebalancing

**2. Class-Balanced Loss**
- **Formula**: CB = (1 - β) / (1 - β^n_i) * Loss
- **Parameters**:
  - `beta`: 0.9999 (hyperparameter)
  - `samples_per_class`: Number of samples per class
- **Advantage**: Based on effective number of samples theory
- **Reference**: Cui et al., CVPR 2019

**3. Weighted BCE**
- **Formula**: WBCE = w_i * BCE
- **Parameters**: Class weights [16.71, 65.50]
- **Advantage**: Simple, interpretable reweighting

**4. Standard BCE**
- Baseline loss function (no reweighting)

#### 3.2.2 Class Weights

- **Nodule**: 16.71 (1 positive per ~17 negatives)
- **Fibrosis**: 65.50 (1 positive per ~66 negatives)

These weights are calculated as: `weight = total_negative / total_positive`

#### 3.2.3 Batch-Balanced Sampling

- **Implementation**: `WeightedRandomSampler` from PyTorch
- **Purpose**: Oversample rare class examples during training
- **Method**: Sample weights inversely proportional to class frequency
- **Configuration**: `use_weighted_sampling: true/false` in config

**Configuration**:
```yaml
model:
  class_weights: [16.71, 65.50]

loss:
  name: focal  # Options: focal, class_balanced, weighted_bce, bce
  alpha: 0.75
  gamma: 2.0

training:
  use_weighted_sampling: false  # Set to true to enable
```

---

### 3.3 Deep Learning Backbone (`src/models/basemodels.py`)

**Purpose**: Extract discriminative features from preprocessed chest X-ray images.

#### 3.3.1 DenseNet-121 (Default)

**Architecture**:
- **Input**: 224×224×3 normalized images
- **Dense Blocks**: 4 blocks with dense connections
- **Transition Layers**: Compression between blocks
- **Global Pooling**: Average pooling
- **Classifier**: Fully-connected layer → 2 outputs

**Features**:
- Pretrained on ImageNet
- Dense connections improve gradient flow
- Parameter efficient
- Good for medical imaging tasks

**Alternative Backbones**:
- **ResNet-50**: Residual connections, proven architecture
- **Swin Transformer**: Vision transformer, state-of-the-art performance

**Configuration**:
```yaml
model:
  backbone: densenet121  # Options: densenet121, resnet50, swin_t
  pretrained: true
  num_classes: 2
```

---

### 3.4 Multi-Label Prediction Module

**Purpose**: Generate independent binary predictions for each disease class.

**Output Format**:
- **Logits**: Raw model outputs (2 values)
- **Probabilities**: After sigmoid activation [0, 1]
- **Predictions**: Binary after threshold (default: 0.3)

**Metrics**:
- **F1-Score**: Macro-averaged across classes
- **AUROC**: Area under ROC curve (macro)
- **Accuracy**: Macro-averaged accuracy

**Threshold Selection**:
- Default: 0.3 (optimized for rare disease detection)
- Lower threshold = higher recall (fewer false negatives)
- Important for medical applications where missing a disease is costly

---

### 3.5 Explainable AI Module (`src/xai/`)

**Purpose**: Generate interpretable visualizations showing which image regions the model focuses on.

#### 3.5.1 Grad-CAM

**Method**: Gradient-weighted Class Activation Mapping

**Algorithm**:
1. Forward pass through model
2. Backward pass to compute gradients w.r.t. target class
3. Global average pooling of gradients → weights
4. Weighted combination of activation maps
5. Apply ReLU and normalize

**Output**: Heatmap showing important regions (H×W tensor)

#### 3.5.2 Grad-CAM++

**Enhanced Version**:
- Better localization
- Improved handling of multiple object instances
- More accurate attention maps

**Algorithm**:
1. Compute alpha coefficients (gradient weighting)
2. Apply ReLU to gradients
3. Weighted combination with improved weighting scheme

#### 3.5.3 Visualization

**Functions** (in `src/xai/visualize.py`):
- `visualize_heatmap()`: Single image visualization
- `save_heatmap_batch()`: Batch processing

**Output Format**:
- Original image
- Heatmap alone
- Overlaid heatmap on original image

**Target Layers**:
- **DenseNet-121**: `features.norm5` (last normalization layer)
- **ResNet-50**: `layer4[-1].conv3` (last conv layer)
- **Swin-T**: `features[-1][-1].norm2` (last norm layer)

---

## 4. Training Pipeline

### 4.1 Training Configuration

**Optimizer**: AdamW
- Learning rate: 0.0003
- Weight decay: 0.01

**Scheduler**: Cosine Annealing
- T_max: max_epochs
- Warmup: 5 epochs (optional)

**Training Parameters**:
- Batch size: 32
- Max epochs: 40
- Precision: 16-mixed (GPU) / 32 (CPU)
- Mixed precision training for efficiency

### 4.2 Training Flow

1. **Data Loading**
   - Load images from disk
   - Apply preprocessing and augmentation
   - Create batches

2. **Forward Pass**
   - Pass batch through model
   - Get logits (raw predictions)

3. **Loss Calculation**
   - Apply selected loss function
   - Apply class weights if configured

4. **Backward Pass**
   - Compute gradients
   - Update model parameters

5. **Metrics Calculation**
   - Compute F1, AUROC, Accuracy
   - Log to TensorBoard

6. **Validation**
   - Evaluate on validation set
   - No augmentation
   - Track best model

---

## 5. Inference Pipeline

### 5.1 Prediction Workflow

1. **Load Model**
   - Load trained checkpoint
   - Set to evaluation mode

2. **Preprocess Image**
   - Resize to 224×224
   - Normalize
   - Convert to tensor

3. **Forward Pass**
   - Get logits
   - Apply sigmoid → probabilities

4. **Post-process**
   - Apply threshold → binary predictions
   - Return probabilities and predictions

### 5.2 Explainability Workflow

1. **Load Model & Image**
   - Same as prediction workflow

2. **Generate Heatmap**
   - Forward pass with gradient tracking
   - Compute Grad-CAM/Grad-CAM++
   - Get heatmap tensor

3. **Visualize**
   - Overlay heatmap on original image
   - Save visualization
   - Display results

---

## 6. File Structure

```
project_root/
├── configs/
│   ├── base.yaml              # Main configuration
│   ├── cpu_optimized.yaml     # CPU-specific config
│   └── gpu.yaml               # GPU-specific config
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── datasets.py        # Preprocessing module
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── basemodels.py      # Backbone models
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── lightning_module.py  # Training logic & losses
│   │
│   └── xai/
│       ├── __init__.py
│       ├── gradcam.py         # Grad-CAM implementation
│       └── visualize.py       # Visualization utilities
│
├── scripts/
│   ├── train.py               # Training script
│   ├── calculate_class_weights.py
│   └── check_class_distribution.py
│
└── datasets/
    └── NIH Chest X-Rays Master Datasets/
        └── archive/
            ├── Data_Entry_2017.csv
            └── images_*/images/*.png
```

---

## 7. Key Design Decisions

### 7.1 Why 2 Classes Instead of 14?

- **Research Focus**: Specifically targeting rare diseases (nodules and fibrosis)
- **Class Imbalance**: Extreme imbalance (1:16 and 1:65 ratios) requires specialized handling
- **Model Specialization**: Focused model performs better on rare classes
- **Computational Efficiency**: Smaller model, faster training

### 7.2 Why Focal Loss + Class Weights?

- **Complementary Approaches**: Focal loss handles hard examples, class weights handle frequency imbalance
- **Proven Effectiveness**: Works well for extreme imbalance scenarios
- **Flexibility**: Can experiment with other loss functions

### 7.3 Why DenseNet-121?

- **Medical Imaging**: Proven performance on chest X-ray tasks
- **Parameter Efficiency**: Fewer parameters than ResNet-50
- **Feature Reuse**: Dense connections improve feature learning
- **Pretrained Weights**: Available ImageNet pretraining

### 7.4 Why Grad-CAM?

- **Interpretability**: Shows which regions model focuses on
- **Medical Trust**: Critical for clinical acceptance
- **Standard Method**: Well-established, widely used
- **Implementation**: Relatively simple, efficient

---

## 8. Performance Considerations

### 8.1 Class Imbalance Handling

**Problem**: Extreme imbalance (1:16 and 1:65)

**Solutions Implemented**:
1. **Loss-level**: Focal loss + class weights
2. **Data-level**: Weighted sampling (optional)
3. **Threshold**: Lower threshold (0.3) for higher recall

### 8.2 Training Efficiency

- **Mixed Precision**: 16-bit on GPU (2x speedup)
- **Data Loading**: Multi-worker loading (4 workers)
- **Batch Size**: 32 (balance between speed and memory)

### 8.3 Memory Optimization

- **Gradient Checkpointing**: Can be enabled for large models
- **Batch Size**: Adjustable based on GPU memory
- **Mixed Precision**: Reduces memory usage

---

## 9. Usage Examples

### 9.1 Training

```bash
python scripts/train.py --config configs/base.yaml --devices 1
```

### 9.2 Generate Heatmaps

```python
from src.xai.gradcam import generate_heatmap
from src.models.basemodels import build_backbone

# Load model
model = build_backbone("densenet121", num_classes=2, pretrained=False)
model.load_state_dict(torch.load("checkpoint.ckpt"))

# Generate heatmap
heatmap = generate_heatmap(
    model=model,
    input_tensor=image_tensor,
    backbone_name="densenet121",
    target_class=0,  # Nodule
    method="gradcam"
)
```

---

## 10. Future Enhancements

1. **Additional Backbones**: Vision Transformers, EfficientNet
2. **Advanced Augmentation**: GAN-based synthetic data
3. **Ensemble Methods**: Combine multiple models
4. **Patient-Level Splitting**: Prevent data leakage
5. **Multi-Dataset Training**: Combine NIH, CheXpert, MIMIC
6. **Active Learning**: Intelligent sample selection
7. **Uncertainty Quantification**: Prediction confidence scores

---

## 11. References

1. **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
2. **Class-Balanced Loss**: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019
3. **Grad-CAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks", ICCV 2017
4. **Grad-CAM++**: Chattopadhay et al., "Grad-CAM++: Improved Visual Explanations", WACV 2018
5. **DenseNet**: Huang et al., "Densely Connected Convolutional Networks", CVPR 2017
6. **NIH ChestX-ray14**: Wang et al., "ChestX-ray8: Hospital-scale Chest X-ray Database", CVPR 2017

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Author**: System Architecture Documentation

