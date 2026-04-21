from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryRecall,
    MultilabelAUROC,
    MultilabelAccuracy,
    MultilabelF1Score,
)


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
    clip: Union[float, torch.Tensor, List[float], Tuple[float, ...]] = 0.2,
    eps: float = 1e-8,
    reduction: str = "mean",
    class_weights: Optional[torch.Tensor] = None,
    class_weight_mode: str = "positive_only",
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
        clip: Per-class minimum prob for negative ASL clamping (scalar or length-C list/tensor).
            Use a higher clip on rare classes (e.g. Fibrosis) to suppress easy negatives more aggressively.
        eps: Small epsilon for numerical stability
        reduction: 'mean' or 'none'
    
    Returns:
        Asymmetric loss tensor
    """
    probs = torch.sigmoid(logits)
    num_c = int(logits.shape[1])
    if isinstance(clip, torch.Tensor):
        clip_t = clip.to(device=logits.device, dtype=logits.dtype).view(1, -1)
    elif isinstance(clip, (list, tuple)):
        clip_t = torch.tensor(clip, device=logits.device, dtype=logits.dtype).view(1, -1)
    else:
        clip_t = torch.full(
            (1, num_c), float(clip), device=logits.device, dtype=logits.dtype
        )
    if clip_t.shape[1] == 1 and num_c > 1:
        clip_t = clip_t.expand(1, num_c)
    elif clip_t.shape[1] != num_c:
        raise ValueError(f"clip must have length 1 or {num_c}, got {clip_t.shape[1]}")
    
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
    probs_m = torch.max(probs, clip_t)  # per-class clip for negatives
    pt_neg = (1 - probs_m) * (1 - targets)  # pt_neg = 1 - p_m for negatives
    
    # Compute asymmetric loss components
    # Positive loss: focal loss variant - focuses on hard positive examples
    loss_pos = -targets * torch.log(pt_pos + eps) * torch.pow(1 - pt_pos, gamma_pos)
    # Negative loss: with hard thresholding to prevent easy negatives from suppressing rare positives
    loss_neg = -(1 - targets) * torch.log(1 - pt_neg + eps) * torch.pow(pt_neg, gamma_neg)
    
    loss = loss_pos + loss_neg
    
    # Apply class weights if provided.
    # positive_only: upweight only positive targets for rare diseases (recommended)
    # all: legacy behavior, scales both positive and negative terms per class
    if class_weights is not None:
        class_weights = class_weights.to(loss.device)
        if class_weight_mode == "positive_only":
            # [B, C]: positives get class weight, negatives stay weight=1
            weight_tensor = 1.0 + (class_weights.unsqueeze(0) - 1.0) * targets
        else:
            # [num_classes] -> [batch, num_classes]
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


def _adamw_param_groups(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    backbone_lr_mult: float,
) -> List[Dict[str, Any]]:
    """Split torchvision CNN/Swin blocks vs classifier head for transfer learning."""
    if backbone_lr_mult >= 0.999:
        return [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
        ]
    backbone_params: List[nn.Parameter] = []
    head_params: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        top = name.split(".", 1)[0]
        if top in ("classifier", "fc", "head", "fusion_gate"):
            head_params.append(param)
        else:
            backbone_params.append(param)
    if not backbone_params or not head_params:
        return [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
        ]
    return [
        {
            "params": backbone_params,
            "lr": learning_rate * backbone_lr_mult,
            "weight_decay": weight_decay,
        },
        {"params": head_params, "lr": learning_rate, "weight_decay": weight_decay},
    ]


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
        f1_metric_threshold: Optional[float] = None,
        per_class_thresholds: Optional[Union[List[float], Iterable[float]]] = None,
        backbone_lr_mult: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.backbone_lr_mult = backbone_lr_mult
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
        # Use a lower threshold for F1 metric so it responds during training (ASL keeps probs modest)
        self._f1_threshold = f1_metric_threshold if f1_metric_threshold is not None else prediction_threshold

        num_classes = getattr(model, "num_classes", None)
        if num_classes is None:
            raise ValueError("Model must expose `num_classes` attribute for metrics initialization.")

        thr_list: Optional[List[float]] = None
        if per_class_thresholds is not None:
            thr_list = [float(t) for t in list(per_class_thresholds)]
            if len(thr_list) != num_classes:
                raise ValueError(
                    f"per_class_thresholds must have length {num_classes}, got {len(thr_list)}"
                )
        self._use_per_class_thr = thr_list is not None

        if self._use_per_class_thr:
            # Separate threshold per label (Nodule, Fibrosis) — macro F1 = mean of per-class F1
            for stage in ("train", "val", "test"):
                f1_bins = nn.ModuleList(
                    [BinaryF1Score(threshold=thr_list[i]) for i in range(num_classes)]  # type: ignore[index]
                )
                acc_bins = nn.ModuleList(
                    [BinaryAccuracy(threshold=thr_list[i]) for i in range(num_classes)]  # type: ignore[index]
                )
                setattr(self, f"{stage}_f1_bins", f1_bins)
                setattr(self, f"{stage}_acc_bins", acc_bins)
            # Val-only macro sensitivity (recall at tuned thresholds) for clinical checkpointing
            self.val_recall_bins = nn.ModuleList(
                [BinaryRecall(threshold=thr_list[i]) for i in range(num_classes)]  # type: ignore[index]
            )
            # Placeholders so old code paths can reference (not updated when per-class mode)
            self.train_f1 = MultilabelF1Score(num_labels=num_classes, average="macro", threshold=self._f1_threshold)
            self.val_f1 = MultilabelF1Score(num_labels=num_classes, average="macro", threshold=self._f1_threshold)
            self.test_f1 = MultilabelF1Score(num_labels=num_classes, average="macro", threshold=self._f1_threshold)
            self.val_acc = MultilabelAccuracy(num_labels=num_classes, average="macro", threshold=self._f1_threshold)
            self.test_acc = MultilabelAccuracy(num_labels=num_classes, average="macro", threshold=self._f1_threshold)
        else:
            self.train_f1 = MultilabelF1Score(num_labels=num_classes, average="macro", threshold=self._f1_threshold)
            self.val_f1 = MultilabelF1Score(num_labels=num_classes, average="macro", threshold=self._f1_threshold)
            self.test_f1 = MultilabelF1Score(num_labels=num_classes, average="macro", threshold=self._f1_threshold)
            self.val_acc = MultilabelAccuracy(num_labels=num_classes, average="macro", threshold=self._f1_threshold)
            self.test_acc = MultilabelAccuracy(num_labels=num_classes, average="macro", threshold=self._f1_threshold)

        self.val_auc = MultilabelAUROC(num_labels=num_classes, average="macro")
        self.test_auc = MultilabelAUROC(num_labels=num_classes, average="macro")
        # Per-class AUROC (macro alone can hide a dead label)
        self.val_auc_nodule = BinaryAUROC()
        self.val_auc_fibrosis = BinaryAUROC()
        self.test_auc_nodule = BinaryAUROC()
        self.test_auc_fibrosis = BinaryAUROC()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def _forward_logits(self, images: torch.Tensor, stage: str) -> torch.Tensor:
        """Same forward path for train/val/test; Lightning Trainer applies mixed precision consistently."""
        return self.model(images)

    def configure_optimizers(self) -> Dict[str, Any]:
        param_groups = _adamw_param_groups(
            self.model, self.learning_rate, self.weight_decay, self.backbone_lr_mult
        )
        optimizer = torch.optim.AdamW(param_groups)

        # Implement warmup + cosine annealing
        cos_floor = max(1e-8, float(self.learning_rate) * 0.01)
        if self.warmup_epochs > 0:
            # Linear warmup followed by cosine annealing
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=self.warmup_epochs
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs - self.warmup_epochs,
                eta_min=cos_floor,
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
                optimizer, T_max=self.max_epochs, eta_min=cos_floor
            )
        
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        images = batch["image"]
        labels = batch["labels"]
        batch_size = int(images.shape[0])
        logits = self._forward_logits(images, stage)
        if stage in ("val", "test"):
            logits = torch.clamp(logits, -50.0, 50.0)

        # Select loss function based on config
        loss_name = self.loss_config.get("name", "asymmetric")
        
        if loss_name == "asymmetric" or loss_name == "asl":
            # Asymmetric Loss (ASL) - recommended for extreme class imbalance
            gamma_neg = self.loss_config.get("gamma_neg", 4.0)
            gamma_pos = self.loss_config.get("gamma_pos", 1.0)
            clip_cfg = self.loss_config.get("clip_per_class")
            if clip_cfg is not None:
                clip: Union[float, List[float]] = [float(x) for x in list(clip_cfg)]
            else:
                clip = float(self.loss_config.get("clip", 0.2))
            class_weight_mode = self.loss_config.get("class_weight_mode", "positive_only")
            loss = asymmetric_loss(
                logits,
                labels,
                gamma_neg=gamma_neg,
                gamma_pos=gamma_pos,
                clip=clip,
                reduction="mean",
                class_weights=self.class_weights,
                class_weight_mode=class_weight_mode,
            )
            # Auxiliary BCE keeps gradient signal when ASL collapses; label smoothing reduces overconfidence
            bce_aux_weight = self.loss_config.get("bce_aux_weight", 0.0)
            label_smoothing = self.loss_config.get("label_smoothing", 0.0)
            if bce_aux_weight > 0:
                targets_bce = labels
                if label_smoothing > 0:
                    # Soft labels: 0 -> eps, 1 -> 1-eps (reduces overconfident predictions, fewer FPs)
                    targets_bce = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
                bce_aux = F.binary_cross_entropy_with_logits(logits, targets_bce, reduction="mean")
                loss = loss + bce_aux_weight * bce_aux
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

        if stage in ("val", "test") and not torch.isfinite(loss):
            loss = torch.zeros((), device=loss.device, dtype=torch.float32)

        preds = torch.sigmoid(logits)
        metrics = {
            "f1": getattr(self, f"{stage}_f1"),
            "auc": getattr(self, f"{stage}_auc", None),
            "acc": getattr(self, f"{stage}_acc", None),
        }

        if self._use_per_class_thr:
            f1_bins = getattr(self, f"{stage}_f1_bins")
            acc_bins = getattr(self, f"{stage}_acc_bins")
            for i in range(len(f1_bins)):
                f1_bins[i](preds[:, i], labels[:, i].int())
                acc_bins[i](preds[:, i], labels[:, i].int())
            if stage == "val" and hasattr(self, "val_recall_bins"):
                for i in range(len(self.val_recall_bins)):
                    self.val_recall_bins[i](preds[:, i], labels[:, i].int())
        else:
            metrics["f1"](preds, labels.int())
            if metrics["acc"] is not None:
                metrics["acc"](preds, labels.int())

        if metrics["auc"] is not None:
            metrics["auc"](preds, labels.int())

        if stage == "val":
            self.val_auc_nodule(preds[:, 0], labels[:, 0].int())
            self.val_auc_fibrosis(preds[:, 1], labels[:, 1].int())
        elif stage == "test":
            self.test_auc_nodule(preds[:, 0], labels[:, 0].int())
            self.test_auc_fibrosis(preds[:, 1], labels[:, 1].int())

        # Log only epoch-level metrics to avoid thousands of per-step rows in metrics.csv
        log_kw = {"on_step": False, "on_epoch": True, "batch_size": batch_size}
        self.log(f"{stage}_loss", loss, prog_bar=True, **log_kw)
        if not self._use_per_class_thr:
            self.log(f"{stage}_f1", metrics["f1"], prog_bar=True, **log_kw)
            if metrics["acc"] is not None:
                self.log(f"{stage}_acc", metrics["acc"], prog_bar=False, **log_kw)
        if metrics["auc"] is not None:
            self.log(f"{stage}_auc", metrics["auc"], prog_bar=True, **log_kw)
        if stage == "val":
            self.log("val_auc_nodule", self.val_auc_nodule, prog_bar=False, **log_kw)
            self.log("val_auc_fibrosis", self.val_auc_fibrosis, prog_bar=False, **log_kw)
        elif stage == "test":
            self.log("test_auc_nodule", self.test_auc_nodule, prog_bar=False, **log_kw)
            self.log("test_auc_fibrosis", self.test_auc_fibrosis, prog_bar=False, **log_kw)
        finite = torch.isfinite(preds)
        mean_pred = preds[finite].mean() if finite.any() else preds.new_zeros(())
        if stage in ("val", "test"):
            self.log(f"{stage}_mean_pred", mean_pred, prog_bar=False, **log_kw)

        return loss

    def _log_per_class_epoch_metrics(self, stage: str) -> None:
        """Compute macro F1/acc from per-class Binary metrics and log (incl. val_f1 for checkpoints)."""
        f1_bins = getattr(self, f"{stage}_f1_bins")
        acc_bins = getattr(self, f"{stage}_acc_bins")
        f1s = [m.compute() for m in f1_bins]
        accs = [m.compute() for m in acc_bins]
        for m in f1_bins:
            m.reset()
        for m in acc_bins:
            m.reset()
        macro_f1 = torch.stack(
            [x.detach() if isinstance(x, torch.Tensor) else torch.tensor(float(x)) for x in f1s]
        ).mean()
        macro_acc = torch.stack(
            [x.detach() if isinstance(x, torch.Tensor) else torch.tensor(float(x)) for x in accs]
        ).mean()
        class_names = ("nodule", "fibrosis")
        end_kw = {"on_step": False, "on_epoch": True, "batch_size": 1}
        for i, name in enumerate(class_names):
            self.log(f"{stage}_f1_{name}", f1s[i], prog_bar=False, **end_kw)
            self.log(f"{stage}_acc_{name}", accs[i], prog_bar=False, **end_kw)
        self.log(f"{stage}_f1_macro_per_class", macro_f1, prog_bar=False, **end_kw)
        self.log(f"{stage}_acc_macro_per_class", macro_acc, prog_bar=False, **end_kw)
        if stage == "train":
            self.log("train_f1", macro_f1, prog_bar=True, **end_kw)
            self.log("train_acc", macro_acc, prog_bar=False, **end_kw)
        elif stage == "val":
            self.log("val_f1", macro_f1, prog_bar=True, **end_kw)
            self.log("val_acc", macro_acc, prog_bar=False, **end_kw)
            if hasattr(self, "val_recall_bins"):
                recalls = [m.compute() for m in self.val_recall_bins]
                for m in self.val_recall_bins:
                    m.reset()
                macro_sens = torch.stack(
                    [
                        x.detach() if isinstance(x, torch.Tensor) else torch.tensor(float(x))
                        for x in recalls
                    ]
                ).mean()
                self.log("val_macro_sensitivity", macro_sens, prog_bar=False, **end_kw)
        elif stage == "test":
            self.log("test_f1", macro_f1, prog_bar=False, **end_kw)
            self.log("test_acc", macro_acc, prog_bar=False, **end_kw)

    def on_train_epoch_end(self) -> None:
        if self._use_per_class_thr:
            self._log_per_class_epoch_metrics("train")

    def on_validation_epoch_end(self) -> None:
        if self._use_per_class_thr:
            self._log_per_class_epoch_metrics("val")

    def on_test_epoch_end(self) -> None:
        if self._use_per_class_thr:
            self._log_per_class_epoch_metrics("test")

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")


