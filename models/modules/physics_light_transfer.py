from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.light_utils import depth_to_normals, light_to_ray, normalize_depth


class PhysicsLightTransfer(nn.Module):
    def __init__(
        self,
        tau_remove: float = 0.15,
        tau_create: float = 0.15,
        prior_temperature: float = 0.05,
        use_visibility: bool = False,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.tau_remove = tau_remove
        self.tau_create = tau_create
        self.prior_temperature = max(prior_temperature, eps)
        self.use_visibility = use_visibility
        self.eps = eps

    def _direction_3d(self, light: torch.Tensor, ray: Optional[torch.Tensor] = None) -> torch.Tensor:
        if ray is not None:
            return F.normalize(ray.to(device=light.device).float(), dim=1, eps=self.eps).to(dtype=light.dtype)
        return light_to_ray(light, eps=self.eps)

    def _intensity_ambient(self, light: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = light.shape[0]
        if light.shape[1] >= 6:
            intensity = light[:, 3:4].abs().clamp_min(self.eps)
            ambient = light[:, 4:5].abs().clamp_min(self.eps)
        else:
            intensity = torch.ones(batch, 1, dtype=light.dtype, device=light.device)
            ambient = torch.full((batch, 1), 0.1, dtype=light.dtype, device=light.device)
        return intensity, ambient

    def _shading(self, normal: torch.Tensor, light: torch.Tensor, ray: Optional[torch.Tensor] = None) -> torch.Tensor:
        direction = self._direction_3d(light, ray=ray).view(light.shape[0], 3, 1, 1)
        intensity, ambient = self._intensity_ambient(light)
        visibility = torch.ones_like(normal[:, :1])
        lambert = (normal * direction).sum(dim=1, keepdim=True).clamp_min(0.0)
        shading = ambient.view(-1, 1, 1, 1) + intensity.view(-1, 1, 1, 1) * visibility * lambert
        return shading.clamp_min(self.eps)

    def forward(
        self,
        source_image: torch.Tensor,
        source_light: torch.Tensor,
        target_light: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
        source_ray: Optional[torch.Tensor] = None,
        target_ray: Optional[torch.Tensor] = None,
        depth_valid: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        batch, _, height, width = source_image.shape
        if depth is None:
            depth_map = torch.zeros(batch, 1, height, width, dtype=source_image.dtype, device=source_image.device)
            confidence = torch.full_like(depth_map, 0.5)
        else:
            depth_map = normalize_depth(depth.to(device=source_image.device, dtype=source_image.dtype))
            if depth_map.shape[-2:] != (height, width):
                depth_map = F.interpolate(depth_map, size=(height, width), mode="bilinear", align_corners=False)
            confidence = torch.ones_like(depth_map)
            if depth_valid is not None:
                valid = depth_valid.to(device=source_image.device, dtype=source_image.dtype)
                if valid.ndim == 1:
                    valid = valid.view(batch, 1, 1, 1)
                while valid.ndim < depth_map.ndim:
                    valid = valid.unsqueeze(-1)
                confidence = confidence * valid + 0.5 * (1.0 - valid)

        normal = depth_to_normals(depth_map)
        h_s = self._shading(normal, source_light.to(source_image.device), ray=source_ray)
        h_t = self._shading(normal, target_light.to(source_image.device), ray=target_ray)
        delta_l_phys = torch.log(h_t.clamp_min(self.eps)) - torch.log(h_s.clamp_min(self.eps))
        remove_prior = torch.sigmoid((delta_l_phys - self.tau_remove) / self.prior_temperature)
        create_prior = torch.sigmoid((-delta_l_phys - self.tau_create) / self.prior_temperature)

        return {
            "H_s": torch.nan_to_num(h_s),
            "H_t": torch.nan_to_num(h_t),
            "delta_l_phys": torch.nan_to_num(delta_l_phys),
            "remove_prior": torch.nan_to_num(remove_prior),
            "create_prior": torch.nan_to_num(create_prior),
            "depth": torch.nan_to_num(depth_map),
            "normal": torch.nan_to_num(normal),
            "confidence": torch.nan_to_num(confidence),
        }
