from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def call_model_with_cond(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    cond: Any,
) -> torch.Tensor:
    if isinstance(cond, dict):
        y = cond.get("y")
        if y is None:
            y = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return model(
            x,
            t,
            y,
            light_cond=cond.get("light_cond"),
            dense_cond=cond.get("dense_cond"),
            source_tokens=cond.get("source_tokens"),
        )
    return model(x, t, cond)


class RectifiedFlow(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def _shift_timestep(self, t: torch.Tensor, timestep_shift: float) -> torch.Tensor:
        if timestep_shift == 0.0:
            return t
        return (t + timestep_shift).clamp(0.0, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        cond: Any,
        timestep_shift: float = 0.1,
        losses: Optional[dict[str, nn.Module]] = None,
        description: str = "",
        return_outputs: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch = x.shape[0]
        t = torch.rand(batch, dtype=x.dtype, device=x.device)
        t = self._shift_timestep(t, timestep_shift)
        view_shape = [batch] + [1] * (x.ndim - 1)
        z0 = torch.randn_like(x)
        z1 = x
        zt = (1.0 - t.view(*view_shape)) * z0 + t.view(*view_shape) * z1
        ut = z1 - z0
        model_output = call_model_with_cond(self.model, zt, t, cond)
        if losses is not None and "mse" in losses:
            output = losses["mse"](model_output, ut, description=description)
        else:
            mse = F.mse_loss(model_output.float(), ut.float())
            prefix = description if description else "Flow"
            output = {f"{prefix}/loss": torch.nan_to_num(mse), f"{prefix}/mse": torch.nan_to_num(mse)}
        if return_outputs:
            output.update({"model_output": model_output, "zt": zt, "t": t, "ut": ut})
        return output

    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        cond: Any,
        null_cond: Optional[Any] = None,
        sample_steps: int = 50,
        cfg: float = 1.0,
        progress: bool = False,
        mode: str = "euler",
        timestep_shift: float = 1.0,
        cfg_mode: str = "constant",
    ) -> torch.Tensor:
        if mode != "euler":
            raise ValueError(f"Only Euler sampling is supported, got mode='{mode}'.")
        x = z
        steps = max(int(sample_steps), 1)
        iterator = range(steps)
        if progress:
            iterator = tqdm(iterator, desc="Sampling", dynamic_ncols=True)
        dt = 1.0 / steps
        for step in iterator:
            t_value = step / steps
            t = torch.full((x.shape[0],), t_value, dtype=x.dtype, device=x.device)
            t = self._shift_timestep(t, timestep_shift)
            velocity = call_model_with_cond(self.model, x, t, cond)
            if null_cond is not None and cfg != 1.0:
                null_velocity = call_model_with_cond(self.model, x, t, null_cond)
                velocity = null_velocity + cfg * (velocity - null_velocity)
            x = x + dt * velocity
        return torch.nan_to_num(x)
