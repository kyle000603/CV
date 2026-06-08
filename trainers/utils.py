from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import torch
import torch.nn as nn


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def scalar_dict_to_float(values: dict[str, torch.Tensor]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in values.items():
        if torch.is_tensor(value) and value.ndim == 0:
            output[key] = float(value.detach().cpu())
    return output


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """Set requires_grad for every parameter in a module."""
    for param in module.parameters():
        param.requires_grad = requires_grad


def count_trainable_parameters(module: nn.Module) -> int:
    """Count trainable parameters in a module."""
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def count_parameters(module: nn.Module) -> int:
    """Count all parameters in a module."""
    return sum(param.numel() for param in module.parameters())


def ensure_dir(path: str | Path) -> Path:
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def next_numbered_run_dir(
    root: str | Path,
    name: str,
    width: int = 3,
) -> Path:
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^(\d{{{width},}})_{re.escape(name)}$")
    max_index = 0
    for child in base.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match is not None:
            max_index = max(max_index, int(match.group(1)))
    next_index = max_index + 1
    return base / f"{next_index:0{width}d}_{name}"


def state_dict_for_save(models: dict[str, nn.Module]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, model in models.items():
        module = unwrap_model(model)
        if bool(getattr(module, "_cplightsit_skip_checkpoint", False)):
            continue
        output[name] = module.state_dict()
    return output
