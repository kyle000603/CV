from __future__ import annotations

import torch
import torch.nn.functional as F

from models.modules.light_utils import rgb_to_luminance


def compute_log_luminance_transfer(
    source_image: torch.Tensor,
    target_image: torch.Tensor,
    eps: float = 1e-4,
    q_clip: float | None = None,
) -> torch.Tensor:
    """Compute target-source log-luminance transfer for images in [-1, 1]."""
    source_lum = rgb_to_luminance(source_image).float().clamp_min(eps)
    target_lum = rgb_to_luminance(target_image).float().clamp_min(eps)
    transfer = torch.log(target_lum) - torch.log(source_lum)
    if q_clip is not None and q_clip > 0:
        transfer = transfer.clamp(min=-float(q_clip), max=float(q_clip))
    return torch.nan_to_num(transfer)


def compute_dense_transfer_loss(
    pred_delta_l: torch.Tensor,
    source_image: torch.Tensor,
    target_image: torch.Tensor,
    q_clip: float = 2.0,
    beta: float = 0.1,
    description: str = "Train/light_transfer",
) -> dict[str, torch.Tensor]:
    """Compute SmoothL1 loss for log-luminance transfer."""
    q_star = compute_log_luminance_transfer(source_image, target_image, q_clip=q_clip)
    loss = F.smooth_l1_loss(pred_delta_l.float(), q_star.float(), beta=beta)
    return {
        f"{description}/loss": torch.nan_to_num(loss),
        f"{description}/q_abs": torch.nan_to_num(q_star.float().abs().mean()),
        f"{description}/delta_l_abs": torch.nan_to_num(pred_delta_l.float().abs().mean()),
    }
