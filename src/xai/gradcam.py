"""Grad-CAM and Grad-CAM++ implementations for model interpretability."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def _signed_aware_cam_normalize(cam_linear: torch.Tensor) -> torch.Tensor:
    """Turn linear CAM combination into a [0,1] heatmap per batch item.

    Standard Grad-CAM uses ReLU on the weighted sum of feature maps. EfficientNet (and
    other BN+SiLU stacks) often produce signed activations, so sum(w * A) can be
    negative everywhere → ReLU → an all-zero map (solid blue in jet). If that happens,
    we fall back to absolute magnitude so saliency is still visible (diagnostic only).
    """
    cam_pos = F.relu(cam_linear)
    b = cam_linear.shape[0]
    rows: list[torch.Tensor] = []
    for i in range(b):
        c = cam_pos[i]
        if float(c.max()) < 1e-8:
            c = torch.abs(cam_linear[i])
        lo, hi = c.min(), c.max()
        if float(hi - lo) < 1e-8:
            rows.append(torch.zeros_like(cam_linear[i]))
        else:
            rows.append((c - lo) / (hi - lo + 1e-8))
    return torch.stack(rows, dim=0)


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
        self.activations = None
        self.gradients = None

        # Register hooks for both forward and backward passes
        self.forward_hook_handle = self.target_layer.register_forward_hook(self._save_activation)
        self.backward_hook_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output) -> None:
        """Save activation maps from forward pass."""
        # Store activations - DO NOT clone or detach! Keep them in computation graph
        # This is critical for gradients to flow through backward hook
        if isinstance(output, torch.Tensor):
            # Just store reference - don't clone (cloning breaks computation graph)
            # Inplace operations in DenseNet are handled by using train mode
            self.activations = output
        else:
            self.activations = output

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        """Save gradients from backward pass."""
        # Store gradients - grad_output is a tuple, take the first element
        # For ReLU layers, grad_output[0] contains the gradients
        if grad_output is not None and len(grad_output) > 0 and grad_output[0] is not None:
            self.gradients = grad_output[0].clone()

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
        # Set model to train mode and ensure all parameters require grad
        was_training = self.model.training
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Clear previous gradients
        self.gradients = None
        self.activations = None
        
        # Ensure input tensor requires gradients
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        # Forward pass with gradients enabled
        # Activations will be saved by forward hook
        # Gradients will be saved by backward hook during backward pass
        with torch.enable_grad():
            output = self.model(input_tensor)

        # Restore model mode
        self.model.train(was_training)

        # Check if activations were captured
        if self.activations is None:
            raise RuntimeError("Activations not captured. Forward hook may have failed.")

        # Determine target class (use detached output for decision)
        if target_class is None:
            probs = torch.sigmoid(output.detach())
            target_class = probs.argmax(dim=1)

        # Determine target output (keep connected to computation graph for backward)
        if output.dim() > 1:
            if isinstance(target_class, int):
                target = output[:, target_class]
            else:
                target = output.gather(1, target_class.unsqueeze(1)).squeeze(1)
        else:
            target = output

        # CRITICAL CHECK: target MUST require grad for backward to work
        if not target.requires_grad:
            raise RuntimeError(
                f"Target output does not require gradients. "
                f"Output requires_grad: {output.requires_grad}, "
                f"Model training: {self.model.training}, "
                f"Input requires_grad: {input_tensor.requires_grad}. "
                f"This usually means the model forward pass didn't create a computation graph."
            )

        # Backward pass - this will trigger the backward hook to save gradients
        self.model.zero_grad()
        target.sum().backward(retain_graph=False)

        # Get gradients from backward hook
        gradients = self.gradients
        if gradients is None:
            raise RuntimeError("Gradients not captured. Backward hook may have failed. Make sure the target layer supports gradient computation.")

        # Get activations (detach for final computation)
        activations = self.activations.detach()  # (B, C, H, W)

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        # Weighted combination of activation maps (may be signed for EfficientNet)
        cam_linear = torch.sum(weights * activations, dim=1)  # (B, H, W)
        return _signed_aware_cam_normalize(cam_linear)


class HiResCAM(GradCAM):
    """HiResCAM: uses (ReLU(grad) * activation) summed over channels for sharper maps.

    Draelos et al., "HiResCAM: High-Resolution Class Activation Mapping for Neural Networks"
    — often localizes better than global-average-pooled Grad-CAM on CNNs.
    """

    def generate(
        self, input_tensor: torch.Tensor, target_class: Optional[int] = None
    ) -> torch.Tensor:
        was_training = self.model.training
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

        self.gradients = None
        self.activations = None

        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        with torch.enable_grad():
            output = self.model(input_tensor)

        self.model.train(was_training)

        if self.activations is None:
            raise RuntimeError("Activations not captured. Forward hook may have failed.")

        if target_class is None:
            probs = torch.sigmoid(output.detach())
            target_class = probs.argmax(dim=1)

        if output.dim() > 1:
            if isinstance(target_class, int):
                target = output[:, target_class]
            else:
                target = output.gather(1, target_class.unsqueeze(1)).squeeze(1)
        else:
            target = output

        if not target.requires_grad:
            raise RuntimeError(
                "Target output does not require gradients for HiResCAM backward."
            )

        self.model.zero_grad()
        target.sum().backward(retain_graph=False)

        gradients = self.gradients
        if gradients is None:
            raise RuntimeError("Gradients not captured for HiResCAM.")

        activations = self.activations.detach()
        cam_linear = torch.sum(F.relu(gradients) * activations, dim=1)  # (B, H, W)
        return _signed_aware_cam_normalize(cam_linear)


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
        # Set model to eval mode (we'll handle gradients manually)
        self.model.eval()

        # Ensure input tensor requires gradients
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        # Forward pass - activations will be saved by the hook
        output = self.model(input_tensor)

        # Determine target class
        if target_class is None:
            probs = torch.sigmoid(output)
            target_class = probs.argmax(dim=1)

        # Determine target output
        if output.dim() > 1:
            if isinstance(target_class, int):
                target = output[:, target_class]
            else:
                target = output.gather(1, target_class.unsqueeze(1)).squeeze(1)
        else:
            target = output

        # Compute gradients directly using autograd.grad
        if self.activations is None:
            raise RuntimeError("Activations not captured. Forward hook may have failed.")

        # Ensure activations require grad
        if not self.activations.requires_grad:
            # Re-run forward pass with activations that require grad
            self.activations = None
            input_tensor = input_tensor.requires_grad_(True)
            output = self.model(input_tensor)
            if self.activations is None:
                raise RuntimeError("Failed to capture activations on second attempt.")
        
        # Compute gradients of target with respect to activations
        gradients = torch.autograd.grad(
            outputs=target.sum(),
            inputs=self.activations,
            retain_graph=False,
            create_graph=False,
        )[0]

        # Get activations from hook
        activations = self.activations  # (B, C, H, W)
        
        # Detach activations for final computation
        activations = activations.detach()

        # Grad-CAM++ specific calculations
        alpha = torch.sum(gradients, dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        alpha = F.relu(alpha) / (torch.sum(F.relu(gradients), dim=(2, 3), keepdim=True) + 1e-8)

        weights = alpha * F.relu(gradients)  # (B, C, H, W)
        weights = torch.sum(weights, dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        cam_linear = torch.sum(weights * activations, dim=1)  # (B, H, W)
        return _signed_aware_cam_normalize(cam_linear)


def get_target_layer(
    model: nn.Module,
    backbone_name: str,
    target_layer_mode: str = "default",
) -> nn.Module:
    """
    Get target layer for Grad-CAM based on backbone architecture.

    Args:
        model: Model instance
        backbone_name: Name of backbone ('densenet121', 'resnet50', 'swin_t', 'dense_swin_hybrid', ...)

    Returns:
        Target convolutional layer
    """
    if backbone_name == "densenet121":
        # DenseNet: use the last conv layer in the last dense block
        last_dense_block = model.features[-2]
        last_dense_layer = list(last_dense_block.children())[-1]
        return last_dense_layer.conv2
    elif backbone_name == "resnet50":
        return model.layer4[-1].conv3
    elif backbone_name == "swin_t":
        return model.features[-1][-1].norm2
    elif backbone_name == "dense_swin_hybrid":
        # Explain local path via DenseNet branch (same layer choice as densenet121).
        dn = model.densenet
        last_dense_block = dn.features[-2]
        last_dense_layer = list(last_dense_block.children())[-1]
        return last_dense_layer.conv2
    elif backbone_name in ("efficientnet_b0", "efficientnet_b2"):
        if target_layer_mode == "ultra_high_res":
            # Earlier stage than high_res → finer CAM grid (less blobby upsampling).
            return model.features[4][-1].block[1][0]
        if target_layer_mode == "high_res":
            # Higher-resolution CAM (roughly 20x20 at 320 input) for better localization
            # Pick a mid-late depthwise conv before the final stride-2 stage.
            return model.features[5][-1].block[1][0]
        if target_layer_mode == "mid_res":
            # Mid-resolution CAM with stronger semantics than high_res
            # Typically gives less tiny-point noise and more disease-region context.
            return model.features[6][-1].block[1][0]
        # Default: last conv before avgpool (roughly 10x10 at 320 input), semantically strong but coarse.
        last_conv = None
        for m in model.features.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise ValueError("No Conv2d found in EfficientNet features")
        return last_conv
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")


def generate_heatmap(
    model: nn.Module,
    input_tensor: torch.Tensor,
    backbone_name: str,
    target_class: Optional[int] = None,
    method: str = "gradcam++",
    target_layer_mode: str = "default",
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
    target_layer = get_target_layer(model, backbone_name, target_layer_mode=target_layer_mode)

    m = method.lower().replace("-", "_")
    if m == "gradcam++":
        cam_impl = GradCAMPlusPlus(model, target_layer)
    elif m in ("hires_cam", "hirescam"):
        cam_impl = HiResCAM(model, target_layer)
    else:
        cam_impl = GradCAM(model, target_layer)

    return cam_impl.generate(input_tensor, target_class)

