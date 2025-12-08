# Documentation for AI Tools - System Architecture Update

## Purpose
This document provides comprehensive information about the implemented system architecture for rare lung disease detection (nodules and fibrosis) in chest X-rays. Use this with other AI tools to update your research documents, proposals, or technical reports.

---

## Executive Summary

The system implements a complete deep learning pipeline for detecting pulmonary nodules and pulmonary fibrosis in chest X-ray images. The architecture addresses extreme class imbalance (1:16 and 1:65 ratios) through multiple techniques and provides explainable AI capabilities via Grad-CAM heatmaps.

**Key Achievements:**
- ✅ Complete preprocessing pipeline with data augmentation
- ✅ Multiple loss functions for class imbalance (Focal Loss, Class-Balanced Loss, Weighted BCE)
- ✅ Batch-balanced sampling support
- ✅ DenseNet-121 backbone with pretrained weights
- ✅ Multi-label classification (2 classes: Nodule, Fibrosis)
- ✅ Explainable AI with Grad-CAM and Grad-CAM++
- ✅ Comprehensive visualization utilities

---

## System Architecture Flow

### Stage 1: Input
- **Input**: Raw frontal-view chest X-ray images (PNG format, various resolutions)
- **Source**: NIH ChestX-ray14 dataset
- **Format**: Single-channel or multi-channel images

### Stage 2: Preprocessing Module
**Location**: `src/data/datasets.py`

**Operations Performed:**
1. **Image Loading**: Load PNG images from dataset structure
2. **Resizing**: Resize all images to 224×224 pixels (standard for ImageNet models)
3. **Channel Conversion**: Convert grayscale to 3-channel RGB (replication)
4. **Normalization**: Apply ImageNet statistics
   - Mean: [0.485, 0.456, 0.406]
   - Standard deviation: [0.229, 0.224, 0.225]
5. **Label Encoding**: Extract binary labels from metadata
   - "Nodule" → [1, 0] or [0, 1] or [1, 1] (multi-label)
   - "Fibrosis" → Binary encoding
6. **Data Augmentation** (training only):
   - Horizontal flip: 50% probability
   - Rotation: ±7.5 degrees
   - Random Gaussian noise: std=0.01
7. **Data Splitting**: 80% train, 10% validation, 10% test

**Key Classes/Functions:**
- `NIHChestXRayDataset`: PyTorch Dataset class
- `create_dataloaders()`: Factory function for dataloaders

### Stage 3: Imbalance-Aware Training Module
**Location**: `src/training/lightning_module.py`

**Loss Functions Implemented:**

1. **Focal Loss** (Default)
   - Formula: FL = α_t × (1 - p_t)^γ × BCE
   - Parameters: alpha=0.75, gamma=2.0
   - Purpose: Focuses on hard examples, down-weights easy examples
   - Combined with class weights for additional rebalancing

2. **Class-Balanced Loss**
   - Formula: CB = (1 - β) / (1 - β^n_i) × Loss
   - Parameters: beta=0.9999, samples_per_class
   - Purpose: Based on effective number of samples theory
   - Reference: Cui et al., CVPR 2019

3. **Weighted BCE**
   - Formula: WBCE = w_i × BCE
   - Class weights: [16.71, 65.50] for [Nodule, Fibrosis]
   - Purpose: Simple reweighting based on class frequency

4. **Standard BCE**
   - Baseline loss function

**Data-Level Techniques:**
- **Class Weights**: [16.71, 65.50] calculated as total_negative / total_positive
- **Batch-Balanced Sampling**: Optional WeightedRandomSampler
  - Oversamples rare class examples
  - Sample weights inversely proportional to class frequency

**Configuration Options:**
```yaml
loss:
  name: focal  # Options: focal, class_balanced, weighted_bce, bce
  alpha: 0.75
  gamma: 2.0
  beta: 0.9999  # For class_balanced loss

training:
  use_weighted_sampling: false  # Enable batch-balanced sampling
```

### Stage 4: Deep Learning Backbone
**Location**: `src/models/basemodels.py`

**DenseNet-121 Architecture** (Default):
- **Input**: 224×224×3 normalized images
- **Structure**:
  - Initial convolution: 7×7 conv, stride 2
  - Dense Block 1: 6 layers
  - Transition Layer 1: 1×1 conv + 2×2 avg pool
  - Dense Block 2: 12 layers
  - Transition Layer 2: 1×1 conv + 2×2 avg pool
  - Dense Block 3: 24 layers
  - Transition Layer 3: 1×1 conv + 2×2 avg pool
  - Dense Block 4: 16 layers
  - Global Average Pooling
  - Classifier: Fully-connected → 2 outputs

**Features:**
- Pretrained on ImageNet (transfer learning)
- Dense connections improve gradient flow
- Parameter efficient (~8M parameters)
- Proven performance on medical imaging

**Alternative Backbones:**
- ResNet-50: Residual connections
- Swin Transformer (Swin-T): Vision transformer architecture

### Stage 5: Multi-Label Prediction
**Location**: `src/training/lightning_module.py`

**Output Format:**
- **Logits**: Raw model outputs (2 values)
- **Probabilities**: After sigmoid activation [0, 1]
  - P(Nodule) = sigmoid(logit_0)
  - P(Fibrosis) = sigmoid(logit_1)
- **Binary Predictions**: After threshold (default: 0.3)
  - Optimized for rare disease detection (higher recall)

**Metrics Computed:**
- **F1-Score**: Macro-averaged across classes
- **AUROC**: Area under ROC curve (macro-averaged)
- **Accuracy**: Macro-averaged accuracy

**Threshold Selection:**
- Default: 0.3 (lower than standard 0.5)
- Rationale: Higher recall for rare diseases (fewer false negatives)
- Critical for medical applications where missing a disease is costly

### Stage 6: Explainable AI Module
**Location**: `src/xai/gradcam.py` and `src/xai/visualize.py`

**Grad-CAM Implementation:**

**Algorithm:**
1. Forward pass through model with gradient tracking enabled
2. Backward pass to compute gradients w.r.t. target class
3. Global average pooling of gradients → weights
4. Weighted combination of activation maps
5. Apply ReLU to remove negative contributions
6. Normalize to [0, 1] range

**Grad-CAM++ (Enhanced):**
- Improved localization accuracy
- Better handling of multiple object instances
- Alpha coefficients for gradient weighting
- More accurate attention maps

**Target Layers:**
- **DenseNet-121**: `features.norm5` (last normalization layer)
- **ResNet-50**: `layer4[-1].conv3` (last convolutional layer)
- **Swin-T**: `features[-1][-1].norm2` (last normalization layer)

**Visualization:**
- Original image display
- Heatmap alone (colormap: jet)
- Overlaid heatmap on original image (transparency: 0.6)
- Batch processing support

**Key Functions:**
- `GradCAM`: Base implementation
- `GradCAMPlusPlus`: Enhanced version
- `generate_heatmap()`: Convenience function
- `visualize_heatmap()`: Single image visualization
- `save_heatmap_batch()`: Batch processing

### Stage 7: Final Output

**Predictions:**
- Probability scores: [P(Nodule), P(Fibrosis)]
- Binary predictions: [0/1, 0/1] after threshold
- Confidence scores

**Visualizations:**
- Heatmaps showing model attention regions
- Overlaid heatmaps on original images
- Class-specific attention maps
- Batch visualizations saved to disk

---

## Technical Specifications

### Data Preprocessing
- **Image Size**: 224×224 pixels
- **Normalization**: ImageNet statistics
- **Augmentation**: Horizontal flip (50%), rotation (±7.5°), noise (std=0.01)
- **Train/Val/Test Split**: 80/10/10

### Model Architecture
- **Backbone**: DenseNet-121 (default)
- **Pretrained**: ImageNet weights
- **Output Classes**: 2 (Nodule, Fibrosis)
- **Parameters**: ~8M

### Training Configuration
- **Optimizer**: AdamW
- **Learning Rate**: 0.0003
- **Weight Decay**: 0.01
- **Scheduler**: Cosine Annealing
- **Batch Size**: 32
- **Max Epochs**: 40
- **Precision**: 16-mixed (GPU) / 32 (CPU)

### Class Imbalance Handling
- **Class Weights**: [16.71, 65.50]
- **Loss Function**: Focal Loss (alpha=0.75, gamma=2.0)
- **Threshold**: 0.3 (optimized for rare diseases)
- **Optional**: Batch-balanced sampling

### Explainable AI
- **Method**: Grad-CAM / Grad-CAM++
- **Target Layer**: Last normalization/convolution layer
- **Output**: Heatmap tensor (H×W)
- **Visualization**: Overlay on original image

---

## File Structure

```
project_root/
├── configs/
│   ├── base.yaml              # Main configuration
│   ├── cpu_optimized.yaml     # CPU-specific settings
│   └── gpu.yaml               # GPU-specific settings
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── datasets.py        # Preprocessing module
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── basemodels.py      # Backbone models (DenseNet, ResNet, Swin-T)
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── lightning_module.py  # Training logic, loss functions
│   │
│   └── xai/
│       ├── __init__.py
│       ├── gradcam.py         # Grad-CAM implementation
│       └── visualize.py       # Visualization utilities
│
├── scripts/
│   ├── train.py               # Main training script
│   ├── calculate_class_weights.py
│   └── check_class_distribution.py
│
├── SYSTEM_ARCHITECTURE.md     # Detailed documentation
├── SYSTEM_DIAGRAM.txt          # ASCII diagram
└── DOCUMENTATION_FOR_AI_TOOLS.md  # This file
```

---

## Key Design Decisions

### Why 2 Classes Instead of 14?
- **Research Focus**: Specifically targeting rare diseases
- **Class Imbalance**: Extreme ratios (1:16, 1:65) require specialized handling
- **Model Specialization**: Focused model performs better on rare classes
- **Computational Efficiency**: Smaller model, faster training

### Why Focal Loss + Class Weights?
- **Complementary**: Focal loss handles hard examples, class weights handle frequency
- **Proven Effectiveness**: Works well for extreme imbalance
- **Flexibility**: Can experiment with other loss functions

### Why DenseNet-121?
- **Medical Imaging**: Proven performance on chest X-ray tasks
- **Parameter Efficiency**: Fewer parameters than ResNet-50
- **Feature Reuse**: Dense connections improve feature learning
- **Pretrained Weights**: Available ImageNet pretraining

### Why Grad-CAM?
- **Interpretability**: Shows which regions model focuses on
- **Medical Trust**: Critical for clinical acceptance
- **Standard Method**: Well-established, widely used
- **Implementation**: Relatively simple, efficient

---

## Implementation Status

### ✅ Completed Components

1. **Preprocessing Module** (`src/data/datasets.py`)
   - Complete dataset class implementation
   - Image loading and preprocessing
   - Data augmentation pipeline
   - Train/val/test splitting
   - Dataloader creation

2. **Explainable AI Module** (`src/xai/`)
   - Grad-CAM implementation
   - Grad-CAM++ implementation
   - Visualization utilities
   - Batch processing support

3. **Loss Functions** (`src/training/lightning_module.py`)
   - Focal Loss
   - Class-Balanced Loss
   - Weighted BCE
   - Standard BCE

4. **Batch-Balanced Sampling**
   - WeightedRandomSampler integration
   - Sample weight calculation
   - Configurable via config file

5. **Documentation**
   - System architecture document
   - ASCII diagram
   - This AI tools documentation

---

## Usage Instructions for AI Tools

When updating your documents with this information:

1. **For Methodology Section**: Use the "System Architecture Flow" section
2. **For Implementation Details**: Use the "Technical Specifications" section
3. **For Diagrams**: Use the ASCII diagram from `SYSTEM_DIAGRAM.txt`
4. **For Design Rationale**: Use the "Key Design Decisions" section
5. **For Code References**: Use the "File Structure" section

**Suggested Prompts for AI Tools:**

- "Update the methodology section with the complete preprocessing pipeline details"
- "Add the explainable AI module description to the results section"
- "Include the class imbalance handling techniques in the methods"
- "Update the system architecture diagram with the new components"
- "Add technical specifications for the model architecture"

---

## References

1. **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
2. **Class-Balanced Loss**: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019
3. **Grad-CAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks", ICCV 2017
4. **Grad-CAM++**: Chattopadhay et al., "Grad-CAM++: Improved Visual Explanations", WACV 2018
5. **DenseNet**: Huang et al., "Densely Connected Convolutional Networks", CVPR 2017
6. **NIH ChestX-ray14**: Wang et al., "ChestX-ray8: Hospital-scale Chest X-ray Database", CVPR 2017

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Purpose**: For use with AI tools to update research documents

