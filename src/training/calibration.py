"""Post-hoc temperature scaling for multi-label sigmoid heads (val-fit, test-apply)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

CalibrationMode = Literal["scalar", "per_class"]


class TemperatureScaleWrapper(nn.Module):
    """Wraps a backbone so forward logits are divided by temperature (scalar or per-class)."""

    def __init__(self, inner: nn.Module, temperatures: torch.Tensor) -> None:
        super().__init__()
        self.inner = inner
        t = temperatures.detach().float().view(-1)
        if (t <= 0).any():
            raise ValueError("temperatures must be positive")
        self.register_buffer("temperature", t)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        z = self.inner(*args, **kwargs)
        T = self.temperature
        if T.numel() == 1:
            return z / T
        if z.dim() >= 2 and z.shape[-1] == T.numel():
            return z / T.view(*((1,) * (z.dim() - 1)), -1)
        raise ValueError(
            f"Expected logits last dim {T.numel()} to match per-class temperatures, got shape {tuple(z.shape)}"
        )


def _bce_nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="mean")


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mode: CalibrationMode,
    device: torch.device,
    max_iter: int = 80,
) -> torch.Tensor:
    """Fit temperature(s) on validation logits to minimize multi-label BCE NLL. Returns T of shape (1,) or (C,)."""
    logits = logits.to(device)
    labels = labels.to(device).float()
    num_c = int(logits.shape[1])
    if mode == "scalar":
        log_t = torch.zeros((), device=device, requires_grad=True)

        def closure() -> torch.Tensor:
            opt.zero_grad(set_to_none=True)
            T = F.softplus(log_t) + 1e-4
            loss = _bce_nll(logits / T, labels)
            loss.backward()
            return loss

        opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")
        opt.step(closure)
        T = (F.softplus(log_t) + 1e-4).detach().view(1)
        return T.cpu()
    if mode == "per_class":
        log_t = torch.zeros(num_c, device=device, requires_grad=True)

        def closure_pc() -> torch.Tensor:
            opt.zero_grad(set_to_none=True)
            T = F.softplus(log_t) + 1e-4
            loss = _bce_nll(logits / T.view(1, -1), labels)
            loss.backward()
            return loss

        opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")
        opt.step(closure_pc)
        T = F.softplus(log_t) + 1e-4
        return T.detach().cpu()
    raise ValueError(f"Unknown mode {mode!r}")


def calibration_dict(mode: CalibrationMode, temperatures: torch.Tensor) -> Dict[str, Any]:
    t = temperatures.detach().float().view(-1).tolist()
    if mode == "scalar":
        return {"version": 1, "mode": "scalar", "temperature": float(t[0])}
    return {
        "version": 1,
        "mode": "per_class",
        "label_order": ["Nodule", "Fibrosis"],
        "temperatures": [float(x) for x in t],
    }


def save_calibration_json(path: Union[str, Path], payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def load_calibration_json(path: Union[str, Path]) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if int(data.get("version", 1)) != 1:
        raise ValueError(f"Unsupported calibration version: {data.get('version')}")
    mode = data.get("mode")
    if mode not in ("scalar", "per_class"):
        raise ValueError("calibration JSON must have mode 'scalar' or 'per_class'")
    if mode == "scalar" and "temperature" not in data:
        raise ValueError("scalar calibration requires 'temperature'")
    if mode == "per_class" and "temperatures" not in data:
        raise ValueError("per_class calibration requires 'temperatures'")
    return data


def temperatures_tensor_from_dict(data: Dict[str, Any]) -> torch.Tensor:
    if data["mode"] == "scalar":
        return torch.tensor([float(data["temperature"])], dtype=torch.float32)
    return torch.tensor([float(x) for x in data["temperatures"]], dtype=torch.float32)


def wrap_model_with_calibration(model: nn.Module, cal_path: Union[str, Path]) -> nn.Module:
    data = load_calibration_json(cal_path)
    T = temperatures_tensor_from_dict(data)
    return TemperatureScaleWrapper(model, T)
