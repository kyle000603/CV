from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.light_utils import light_to_ray, ray_to_light


class RayEncoder(nn.Module):
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
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.pool(self.encoder(image)).flatten(1)
        raw = self.head(features)
        ray = F.normalize(raw[:, 0:3].float(), dim=1, eps=1e-6).to(dtype=image.dtype)
        direction = F.normalize(ray[:, 0:2].float(), dim=1, eps=1e-6).to(dtype=image.dtype)
        temperature = torch.tanh(raw[:, 3:4]).to(dtype=image.dtype)
        confidence = torch.sigmoid(raw[:, 4:5]).to(dtype=image.dtype)
        light = ray_to_light(ray, temperature)

        if self.extended and self.light_dim >= 6:
            rho = torch.zeros(light.shape[0], 1, dtype=light.dtype, device=light.device)
            intensity = torch.ones_like(rho)
            ambient = torch.full_like(rho, 0.1)
            light = torch.cat([rho, direction, intensity, ambient, temperature], dim=1)
        if light.shape[1] < self.light_dim:
            pad = torch.zeros(light.shape[0], self.light_dim - light.shape[1], dtype=light.dtype, device=light.device)
            light = torch.cat([light, pad], dim=1)
        light = light[:, : self.light_dim]
        return {
            "light": light,
            "ray": ray,
            "direction": direction,
            "temperature": temperature,
            "confidence": confidence,
        }


def ray_encoder_loss(
    pred: dict[str, torch.Tensor],
    target_light: torch.Tensor,
    target_ray: torch.Tensor | None = None,
    lambda_ray: float = 1.0,
    lambda_dir: float = 1.0,
    lambda_temp: float = 1.0,
) -> dict[str, torch.Tensor]:
    gt_ray = target_ray.float() if target_ray is not None else light_to_ray(target_light).float()
    pred_ray = F.normalize(pred["ray"].float(), dim=1, eps=1e-6)
    gt_ray = F.normalize(gt_ray, dim=1, eps=1e-6)
    ray_loss = (1.0 - (pred_ray * gt_ray).sum(dim=1)).mean()

    pred_dir = F.normalize(pred["direction"].float(), dim=1, eps=1e-6)
    gt_dir = F.normalize(target_light[:, :2].float(), dim=1, eps=1e-6)
    dir_loss = (1.0 - (pred_dir * gt_dir).sum(dim=1)).mean()
    temp_loss = F.l1_loss(pred["temperature"].float(), target_light[:, 2:3].float())
    total = lambda_ray * ray_loss + lambda_dir * dir_loss + lambda_temp * temp_loss
    total = torch.nan_to_num(total)
    return {
        "ray_encoder/loss": total,
        "ray_encoder/ray": torch.nan_to_num(ray_loss),
        "ray_encoder/dir": torch.nan_to_num(dir_loss),
        "ray_encoder/temp": torch.nan_to_num(temp_loss),
        "light_encoder/loss": total,
    }

