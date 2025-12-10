"""Grad-CAM and Grad-CAM++ implementations for model interpretability."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class GradCAM:
    """Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """
        Args:
            model: Trained model
            target_layer: Target convolutional layer for activation maps
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output) -> None:
        """Save activation maps."""
        # Clone to avoid in-place operation conflicts (e.g., DenseNet uses in-place ReLU)
        self.activations = output.clone() if isinstance(output, torch.Tensor) else output

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        """Save gradients."""
        # Clone to avoid in-place operation conflicts
        if grad_output and len(grad_output) > 0:
            self.gradients = grad_output[0].clone() if isinstance(grad_output[0], torch.Tensor) else grad_output[0]
        else:
            self.gradients = None

    def generate(
        self, input_tensor: torch.Tensor, target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input image tensor (B, C, H, W)
            target_class: Target class index. If None, uses highest prediction.

        Returns:
            Heatmap tensor (B, H, W)
        """
        self.model.eval()
        input_tensor.requires_grad_(True)

        # Temporarily patch F.relu to disable inplace for DenseNet compatibility
        original_relu = F.relu
        def relu_no_inplace(input, inplace=False):
            return original_relu(input, inplace=False)
        
        # Monkey-patch F.relu temporarily
        F.relu = relu_no_inplace

        try:
            # Forward pass
            output = self.model(input_tensor)
        finally:
            # Restore original F.relu
            F.relu = original_relu

        # Determine target class
        if target_class is None:
            # For multi-label, use the class with highest probability
            probs = torch.sigmoid(output)
            target_class = probs.argmax(dim=1)

        # Backward pass
        self.model.zero_grad()
        if output.dim() > 1:
            # Multi-label case: select specific class output
            target = output[:, target_class] if isinstance(target_class, int) else output.gather(1, target_class.unsqueeze(1)).squeeze(1)
        else:
            target = output
        target.sum().backward()

        # Get gradients and activations
        gradients = self.gradients  # (B, C, H, W)
        activations = self.activations  # (B, C, H, W)

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (B, 1, H, W)
        cam = F.relu(cam)  # Apply ReLU

        # Normalize
        cam = cam.squeeze(1)  # (B, H, W)
        cam = (cam - cam.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]) / (
            cam.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8
        )

        return cam


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks."""

    def generate(
        self, input_tensor: torch.Tensor, target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate Grad-CAM++ heatmap (improved version with better localization).

        Args:
            input_tensor: Input image tensor (B, C, H, W)
            target_class: Target class index. If None, uses highest prediction.

        Returns:
            Heatmap tensor (B, H, W)
        """
        self.model.eval()
        input_tensor.requires_grad_(True)

        # Temporarily patch F.relu to disable inplace for DenseNet compatibility
        original_relu = F.relu
        def relu_no_inplace(input, inplace=False):
            return original_relu(input, inplace=False)
        
        # Monkey-patch F.relu temporarily
        F.relu = relu_no_inplace

        try:
            # Forward pass
            output = self.model(input_tensor)
        finally:
            # Restore original F.relu
            F.relu = original_relu

        # Determine target class
        if target_class is None:
            probs = torch.sigmoid(output)
            target_class = probs.argmax(dim=1)

        # Backward pass
        self.model.zero_grad()
        if output.dim() > 1:
            target = output[:, target_class] if isinstance(target_class, int) else output.gather(1, target_class.unsqueeze(1)).squeeze(1)
        else:
            target = output
        target.sum().backward()

        # Get gradients and activations
        gradients = self.gradients  # (B, C, H, W)
        activations = self.activations  # (B, C, H, W)

        # Grad-CAM++ specific calculations
        alpha = torch.sum(gradients, dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        alpha = F.relu(alpha) / (torch.sum(F.relu(gradients), dim=(2, 3), keepdim=True) + 1e-8)

        weights = alpha * F.relu(gradients)  # (B, C, H, W)
        weights = torch.sum(weights, dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        # Weighted combination
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (B, 1, H, W)
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze(1)  # (B, H, W)
        cam = (cam - cam.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]) / (
            cam.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8
        )

        return cam


def get_target_layer(model: nn.Module, backbone_name: str) -> nn.Module:
    """
    Get target layer for Grad-CAM based on backbone architecture.

    Args:
        model: Model instance
        backbone_name: Name of backbone ('densenet121', 'resnet50', 'swin_t')

    Returns:
        Target convolutional layer
    """
    if backbone_name == "densenet121":
        # DenseNet: use last dense block's last conv layer
        return model.features.norm5
    elif backbone_name == "resnet50":
        # ResNet: use last convolutional layer before avgpool
        return model.layer4[-1].conv3
    elif backbone_name == "swin_t":
        # Swin Transformer: use last stage's norm layer
        return model.features[-1][-1].norm2
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")


def generate_heatmap(
    model: nn.Module,
    input_tensor: torch.Tensor,
    backbone_name: str,
    target_class: Optional[int] = None,
    method: str = "gradcam",
) -> torch.Tensor:
    """
    Convenience function to generate heatmap.

    Args:
        model: Trained model
        input_tensor: Input image tensor (B, C, H, W)
        backbone_name: Backbone name
        target_class: Target class index
        method: 'gradcam' or 'gradcam++'

    Returns:
        Heatmap tensor (B, H, W)
    """
    target_layer = get_target_layer(model, backbone_name)

    if method.lower() == "gradcam++":
        gradcam = GradCAMPlusPlus(model, target_layer)
    else:
        gradcam = GradCAM(model, target_layer)

    return gradcam.generate(input_tensor, target_class)

