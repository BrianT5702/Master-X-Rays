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


def asymmetric_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_neg: float = 4.0,
    gamma_pos: float = 1.0,
    clip: float = 0.2,
    eps: float = 1e-8,
    reduction: str = "mean",
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Asymmetric Loss (ASL) for multi-label classification with extreme class imbalance.
    
    Reference: "Asymmetric Loss For Multi-Label Classification" by Ben-Baruch et al., ICCV 2021.
    
    ASL decouples the gradient contribution of positive and negative samples, applying
    hard thresholding to easy negatives to prevent gradient suppression from dominating
    the rare positive class learning signal.
    
    Args:
        logits: Raw model outputs (before sigmoid), shape (B, C)
        targets: Binary target labels, shape (B, C)
        gamma_neg: Focusing parameter for negative samples (default: 4.0)
        gamma_pos: Focusing parameter for positive samples (default: 1.0)
        clip: Probability margin for hard thresholding negatives (default: 0.2 for extreme imbalance)
        eps: Small epsilon for numerical stability
        reduction: 'mean' or 'none'
    
    Returns:
        Asymmetric loss tensor
    """
    probs = torch.sigmoid(logits)
    
    # ASL for multilabel classification (Nodule, Fibrosis)
    # For each class independently: handle positive and negative samples
    # Positive samples: when target=1, we want high probability
    # Negative samples: when target=0, we want low probability (with hard thresholding)
    
    # For positive samples: pt = probability of positive class
    # When target=1: pt_pos = probs; when target=0: pt_pos = 0 (masked out)
    pt_pos = probs * targets
    
    # For negative samples: ASL hard thresholding
    # According to ASL paper: for negatives, clamp p (probability of positive) to minimum of clip
    # This means: if model predicts p < clip for a negative, treat it as p = clip (easy negative)
    # If model predicts p >= clip, use actual p (hard negative - model is wrong, should be penalized)
    # Formula: p_m = max(p, m) for negatives, where m = clip
    probs_m = torch.clamp(probs, min=clip)  # Clamp p to minimum of clip for negatives
    pt_neg = (1 - probs_m) * (1 - targets)  # pt_neg = 1 - p_m for negatives
    
    # Compute asymmetric loss components
    # Positive loss: focal loss variant - focuses on hard positive examples
    loss_pos = -targets * torch.log(pt_pos + eps) * torch.pow(1 - pt_pos, gamma_pos)
    # Negative loss: with hard thresholding to prevent easy negatives from suppressing rare positives
    loss_neg = -(1 - targets) * torch.log(1 - pt_neg + eps) * torch.pow(pt_neg, gamma_neg)
    
    loss = loss_pos + loss_neg
    
    # Apply class weights if provided (critical for rare class prioritization)
    if class_weights is not None:
        class_weights = class_weights.to(loss.device)
        # Expand weights to match batch and class dimensions: [num_classes] -> [batch, num_classes]
        weight_tensor = class_weights.unsqueeze(0).expand_as(loss)
        loss = loss * weight_tensor
    
    if reduction == "mean":
        return loss.mean()
    return loss


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
        warmup_epochs: int = 0,
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
        self.warmup_epochs = warmup_epochs

        num_classes = getattr(model, "num_classes", None)
        if num_classes is None:
            raise ValueError("Model must expose `num_classes` attribute for metrics initialization.")

        self.train_f1 = MultilabelF1Score(num_labels=num_classes, average="macro", threshold=prediction_threshold)
        self.val_f1 = MultilabelF1Score(num_labels=num_classes, average="macro", threshold=prediction_threshold)
        self.test_f1 = MultilabelF1Score(num_labels=num_classes, average="macro", threshold=prediction_threshold)

        self.val_auc = MultilabelAUROC(num_labels=num_classes, average="macro")
        self.test_auc = MultilabelAUROC(num_labels=num_classes, average="macro")

        self.val_acc = MultilabelAccuracy(num_labels=num_classes, average="macro")
        self.test_acc = MultilabelAccuracy(num_labels=num_classes, average="macro")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        
        # Implement warmup + cosine annealing
        if self.warmup_epochs > 0:
            # Linear warmup followed by cosine annealing
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=self.warmup_epochs
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_epochs - self.warmup_epochs, eta_min=self.learning_rate * 0.01
            )
            # Chain warmup then cosine annealing
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_epochs],
            )
        else:
            # No warmup, just cosine annealing
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_epochs, eta_min=self.learning_rate * 0.01
            )
        
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        images = batch["image"]
        labels = batch["labels"]
        logits = self(images)

        # Select loss function based on config
        loss_name = self.loss_config.get("name", "asymmetric")
        
        if loss_name == "asymmetric" or loss_name == "asl":
            # Asymmetric Loss (ASL) - recommended for extreme class imbalance
            gamma_neg = self.loss_config.get("gamma_neg", 4.0)
            gamma_pos = self.loss_config.get("gamma_pos", 1.0)
            clip = self.loss_config.get("clip", 0.2)
            loss = asymmetric_loss(
                logits, labels, gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip, reduction="mean",
                class_weights=self.class_weights  # Apply class weights to ASL (critical fix)
            )
        elif loss_name == "focal":
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

        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")


