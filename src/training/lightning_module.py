from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score, MultilabelAccuracy


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal loss for multilabel classification.
    
    Args:
        logits: Raw model outputs (before sigmoid)
        targets: Binary target labels
        alpha: Weighting factor for rare class
        gamma: Focusing parameter
        reduction: 'mean' or 'none'
    """
    probs = torch.sigmoid(logits)
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal_loss = alpha_t * focal_weight * bce_loss
    
    if reduction == "mean":
        return focal_loss.mean()
    return focal_loss


def class_balanced_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    samples_per_class: torch.Tensor,
    beta: float = 0.9999,
    reduction: str = "mean",
) -> torch.Tensor:
    """Class-Balanced Loss based on effective number of samples.
    
    Reference: "Class-Balanced Loss Based on Effective Number of Samples"
    by Cui et al., CVPR 2019.
    
    Args:
        logits: Raw model outputs (before sigmoid)
        targets: Binary target labels
        samples_per_class: Number of samples per class (shape: [num_classes])
        beta: Hyperparameter (typically 0.9999)
        reduction: 'mean' or 'none'
    
    Returns:
        Class-balanced loss
    """
    # Calculate effective number of samples
    effective_num = 1.0 - torch.pow(beta, samples_per_class)
    weights = (1.0 - beta) / (effective_num + 1e-8)
    weights = weights / weights.sum() * len(weights)  # Normalize
    
    # Compute BCE loss
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    
    # Apply class-balanced weights
    weights = weights.to(targets.device)
    weight_tensor = weights.unsqueeze(0).expand_as(targets)
    cb_loss = weight_tensor * bce_loss
    
    if reduction == "mean":
        return cb_loss.mean()
    return cb_loss


class RareDiseaseModule(LightningModule):
    """LightningModule encapsulating the training loop for multilabel classification."""

    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float,
        weight_decay: float,
        class_weights: Optional[Iterable[float]] = None,
        loss_config: Optional[Dict[str, Any]] = None,
        max_epochs: int = 50,
        prediction_threshold: float = 0.3,
        samples_per_class: Optional[Iterable[float]] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        )
        self.samples_per_class = (
            torch.tensor(samples_per_class, dtype=torch.float32) if samples_per_class is not None else None
        )
        self.loss_config = loss_config or {"name": "bce"}
        self.max_epochs = max_epochs
        self.prediction_threshold = prediction_threshold

        num_classes = getattr(model, "num_classes", None)
        if num_classes is None:
            raise ValueError("Model must expose `num_classes` attribute for metrics initialization.")

        # Macro-averaged metrics (overall performance)
        self.train_f1 = MultilabelF1Score(num_labels=num_classes, average="macro", threshold=prediction_threshold)
        self.val_f1 = MultilabelF1Score(num_labels=num_classes, average="macro", threshold=prediction_threshold)
        self.test_f1 = MultilabelF1Score(num_labels=num_classes, average="macro", threshold=prediction_threshold)

        self.val_auc = MultilabelAUROC(num_labels=num_classes, average="macro")
        self.test_auc = MultilabelAUROC(num_labels=num_classes, average="macro")

        self.val_acc = MultilabelAccuracy(num_labels=num_classes, average="macro")
        self.test_acc = MultilabelAccuracy(num_labels=num_classes, average="macro")
        
        # Per-class metrics for diagnosis (to see which class is failing)
        self.val_f1_per_class = MultilabelF1Score(num_labels=num_classes, average="none", threshold=prediction_threshold)
        self.val_auc_per_class = MultilabelAUROC(num_labels=num_classes, average="none")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        images = batch["image"]
        labels = batch["labels"]
        logits = self(images)

        # Select loss function based on config
        loss_name = self.loss_config.get("name", "bce")
        
        if loss_name == "focal":
            alpha = self.loss_config.get("alpha", 0.75)
            gamma = self.loss_config.get("gamma", 2.0)
            # Compute focal loss per element (reduction="none")
            loss_per_element = focal_loss(logits, labels, alpha=alpha, gamma=gamma, reduction="none")
            # Apply class weights if provided
            if self.class_weights is not None:
                weights = self.class_weights.to(labels.device)
                # Expand weights to match batch and class dimensions: [num_classes] -> [batch, num_classes]
                weight_tensor = weights.unsqueeze(0).expand_as(labels)
                loss = (loss_per_element * weight_tensor).mean()
            else:
                loss = loss_per_element.mean()
        elif loss_name == "class_balanced":
            beta = self.loss_config.get("beta", 0.9999)
            if self.samples_per_class is None:
                raise ValueError("samples_per_class required for class-balanced loss")
            loss = class_balanced_loss(
                logits, labels, self.samples_per_class, beta=beta, reduction="mean"
            )
        elif loss_name == "weighted_bce":
            # Weighted BCE as separate option
            loss = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                weight=self.class_weights.to(labels.device) if self.class_weights is not None else None,
            )
        else:  # Default: standard BCE
            loss = F.binary_cross_entropy_with_logits(logits, labels)

        preds = torch.sigmoid(logits)
        metrics = {
            "f1": getattr(self, f"{stage}_f1"),
            "auc": getattr(self, f"{stage}_auc", None),
            "acc": getattr(self, f"{stage}_acc", None),
        }
        metrics["f1"](preds, labels.int())
        if metrics["auc"] is not None:
            metrics["auc"](preds, labels.int())
        if metrics["acc"] is not None:
            metrics["acc"](preds, labels.int())

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_f1", metrics["f1"], prog_bar=True, on_epoch=True)
        if metrics["auc"] is not None:
            self.log(f"{stage}_auc", metrics["auc"], prog_bar=True, on_epoch=True)
        if metrics["acc"] is not None:
            self.log(f"{stage}_acc", metrics["acc"], prog_bar=False, on_epoch=True)
        
        # Log per-class metrics for validation to diagnose issues
        if stage == "val":
            self.val_f1_per_class(preds, labels.int())
            self.val_auc_per_class(preds, labels.int())
            f1_per_class = self.val_f1_per_class.compute()
            auc_per_class = self.val_auc_per_class.compute()
            for i, class_name in enumerate(["Nodule", "Fibrosis"]):
                self.log(f"val_f1_{class_name}", f1_per_class[i], prog_bar=False, on_epoch=True)
                self.log(f"val_auc_{class_name}", auc_per_class[i], prog_bar=False, on_epoch=True)
            # Reset for next epoch
            self.val_f1_per_class.reset()
            self.val_auc_per_class.reset()
        
        # Diagnostic: Check if model is predicting any positives (critical for rare diseases)
        if stage in ["val", "test"]:
            binary_preds = (preds > self.prediction_threshold).float()
            num_positives = binary_preds.sum().item()
            total_samples = binary_preds.numel()
            positive_rate = num_positives / total_samples if total_samples > 0 else 0.0
            self.log(f"{stage}_positive_rate", positive_rate, prog_bar=False, on_epoch=True)
            
            # Log actual vs predicted positives for each class
            for i, class_name in enumerate(["Nodule", "Fibrosis"]):
                class_preds = binary_preds[:, i]
                class_labels = labels[:, i]
                predicted_pos = class_preds.sum().item()
                actual_pos = class_labels.sum().item()
                self.log(f"{stage}_pred_{class_name}", predicted_pos, prog_bar=False, on_epoch=True)
                self.log(f"{stage}_actual_{class_name}", actual_pos, prog_bar=False, on_epoch=True)

        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")


