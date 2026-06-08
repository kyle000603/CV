from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.light_utils import light_to_ray


class LightEncoder(nn.Module):
    def __init__(
        self,
        light_dim: int = 3,
        hidden_dim: int = 256,
        extended: bool = False,
    ) -> None:
        super().__init__()
        self.light_dim = light_dim
        self.extended = extended
        channels = [3, 32, 64, 128, hidden_dim]
        blocks: list[nn.Module] = []
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            blocks.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                ]
            )
        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max(light_dim, 3) + 1),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.pool(self.encoder(image)).flatten(1)
        raw = self.head(features)
        direction = F.normalize(raw[:, 0:2].float(), dim=1, eps=1e-6).to(dtype=image.dtype)
        temperature = torch.tanh(raw[:, 2:3]).to(dtype=image.dtype)
        confidence = torch.sigmoid(raw[:, -1:]).to(dtype=image.dtype)
        ray = light_to_ray(torch.cat([direction, temperature], dim=1))

        if self.extended and self.light_dim >= 6:
            rho = torch.sigmoid(raw[:, 3:4]).to(dtype=image.dtype)
            intensity = F.softplus(raw[:, 4:5]).to(dtype=image.dtype) + 1e-4
            ambient = torch.sigmoid(raw[:, 5:6]).to(dtype=image.dtype)
            light = torch.cat([rho, direction, intensity, ambient, temperature], dim=1)
        else:
            light = torch.cat([direction, temperature], dim=1)

        if light.shape[1] < self.light_dim:
            pad = torch.zeros(light.shape[0], self.light_dim - light.shape[1], dtype=light.dtype, device=light.device)
            light = torch.cat([light, pad], dim=1)
        light = light[:, : self.light_dim]
        return {"light": light, "ray": ray, "direction": direction, "temperature": temperature, "confidence": confidence}


def light_encoder_loss(
    pred: dict[str, torch.Tensor],
    target_light: torch.Tensor,
    lambda_dir: float = 1.0,
    lambda_temp: float = 1.0,
) -> dict[str, torch.Tensor]:
    pred_dir = F.normalize(pred["direction"].float(), dim=1, eps=1e-6)
    gt_dir = F.normalize(target_light[:, :2].float(), dim=1, eps=1e-6)
    dir_loss = (1.0 - (pred_dir * gt_dir).sum(dim=1)).mean()
    temp_loss = F.l1_loss(pred["temperature"].float(), target_light[:, 2:3].float())
    total = lambda_dir * dir_loss + lambda_temp * temp_loss
    return {
        "light_encoder/loss": torch.nan_to_num(total),
        "light_encoder/dir": torch.nan_to_num(dir_loss),
        "light_encoder/temp": torch.nan_to_num(temp_loss),
    }
