from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from rectified_flow.rectified_flow import RectifiedFlow


class TrajectoryFlow(RectifiedFlow):
    def forward(
        self,
        x: torch.Tensor,
        cond: Any,
        mask: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        x0_noise_scale: float = 0.0,
        timestep_shift: float = 0.1,
        bg_noise_ratio: float = 0.5,
        transition_width: float = 0.1,
        losses: Optional[dict[str, nn.Module]] = None,
        description: str = "",
        return_outputs: bool = False,
    ) -> dict[str, torch.Tensor]:
        _ = (mask, bg_noise_ratio, transition_width)
        return super().forward(
            x=x,
            cond=cond,
            x0=x0,
            x0_noise_scale=x0_noise_scale,
            timestep_shift=timestep_shift,
            losses=losses,
            description=description,
            return_outputs=return_outputs,
        )
