from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


DIRECTION_TO_ANGLE: dict[str, float] = {
    "N": 0.0,
    "NE": 45.0,
    "E": 90.0,
    "SE": 135.0,
    "S": 180.0,
    "SW": 225.0,
    "W": 270.0,
    "NW": 315.0,
}


def direction_name_to_angle(name: str) -> float:
    key = name.strip().upper()
    if key not in DIRECTION_TO_ANGLE:
        valid = ", ".join(sorted(DIRECTION_TO_ANGLE))
        raise ValueError(f"Unknown light direction '{name}'. Expected one of: {valid}.")
    return DIRECTION_TO_ANGLE[key]


def angle_to_direction_vector(angle_deg: float, device: Optional[torch.device] = None) -> torch.Tensor:
    angle = math.radians(angle_deg)
    return torch.tensor([math.cos(angle), math.sin(angle)], dtype=torch.float32, device=device)


def angle_to_ray_vector(
    angle_deg: float,
    elevation_deg: float = 45.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    angle = math.radians(angle_deg)
    elevation = math.radians(elevation_deg)
    xy_scale = math.cos(elevation)
    z = math.sin(elevation)
    ray = torch.tensor(
        [math.cos(angle) * xy_scale, math.sin(angle) * xy_scale, z],
        dtype=torch.float32,
        device=device,
    )
    return F.normalize(ray, dim=0, eps=1e-6)


def angle_delta_degrees(source_angle: float, target_angle: float) -> float:
    return (target_angle - source_angle + 180.0) % 360.0 - 180.0


def encode_light(direction_name: str, temperature: float, extended: bool = False) -> torch.Tensor:
    angle = direction_name_to_angle(direction_name)
    direction = angle_to_direction_vector(angle)
    temp_norm = torch.tensor([(float(temperature) - 5500.0) / 2500.0], dtype=torch.float32)
    if extended:
        rho = torch.tensor([0.0], dtype=torch.float32)
        intensity = torch.tensor([1.0], dtype=torch.float32)
        ambient = torch.tensor([0.1], dtype=torch.float32)
        return torch.cat([rho, direction, intensity, ambient, temp_norm], dim=0)
    return torch.cat([direction, temp_norm], dim=0)


def light_to_ray(light: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if light.ndim != 2:
        raise ValueError(f"Expected light tensor with shape [B, C], got {tuple(light.shape)}.")
    xy = light[:, 1:3] if light.shape[1] >= 6 else light[:, 0:2]
    z = torch.ones(light.shape[0], 1, dtype=light.dtype, device=light.device)
    ray = torch.cat([xy, z], dim=1)
    return F.normalize(ray.float(), dim=1, eps=eps).to(dtype=light.dtype)


def ray_to_light(ray: torch.Tensor, temperature: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if ray.ndim != 2 or ray.shape[1] != 3:
        raise ValueError(f"Expected ray tensor with shape [B, 3], got {tuple(ray.shape)}.")
    if temperature.ndim == 1:
        temperature = temperature.unsqueeze(1)
    direction = F.normalize(ray[:, :2].float(), dim=1, eps=eps).to(dtype=ray.dtype)
    return torch.cat([direction, temperature.to(device=ray.device, dtype=ray.dtype)], dim=1)


def rotate_ray_z(ray: torch.Tensor, delta_angle_deg: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if ray.ndim != 2 or ray.shape[1] != 3:
        raise ValueError(f"Expected ray tensor with shape [B, 3], got {tuple(ray.shape)}.")
    if delta_angle_deg.ndim == 0:
        delta_angle_deg = delta_angle_deg.expand(ray.shape[0])
    delta = delta_angle_deg.to(device=ray.device, dtype=ray.dtype).view(ray.shape[0])
    radians = torch.deg2rad(delta.float()).to(dtype=ray.dtype)
    cos_delta = torch.cos(radians)
    sin_delta = torch.sin(radians)
    x = ray[:, 0] * cos_delta - ray[:, 1] * sin_delta
    y = ray[:, 0] * sin_delta + ray[:, 1] * cos_delta
    rotated = torch.stack([x, y, ray[:, 2]], dim=1)
    return F.normalize(rotated.float(), dim=1, eps=eps).to(dtype=ray.dtype)


def image_to_unit_range(x: torch.Tensor) -> torch.Tensor:
    return ((x.float() + 1.0) * 0.5).clamp(0.0, 1.0).to(dtype=x.dtype)


def unit_to_model_range(x: torch.Tensor) -> torch.Tensor:
    return (x.float().clamp(0.0, 1.0) * 2.0 - 1.0).to(dtype=x.dtype)


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    y = x.float().clamp(0.0, 1.0)
    out = torch.where(y <= 0.04045, y / 12.92, torch.pow((y + 0.055) / 1.055, 2.4))
    return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).to(dtype=x.dtype)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    y = x.float().clamp(0.0, 1.0)
    out = torch.where(y <= 0.0031308, y * 12.92, 1.055 * torch.pow(y.clamp_min(1e-8), 1.0 / 2.4) - 0.055)
    return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0).to(dtype=x.dtype)


def rgb_to_luminance(x: torch.Tensor) -> torch.Tensor:
    unit = image_to_unit_range(x)
    linear = srgb_to_linear(unit)
    weights = torch.tensor([0.2126, 0.7152, 0.0722], dtype=linear.dtype, device=linear.device)
    if linear.ndim >= 3 and linear.shape[-3] == 3:
        shape = [1] * linear.ndim
        shape[-3] = 3
        return (linear * weights.view(*shape)).sum(dim=-3, keepdim=True).clamp(0.0, 1.0)
    raise ValueError(f"Expected an RGB tensor with channel dimension at -3, got shape {tuple(x.shape)}.")


def safe_log_luminance(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    lum = rgb_to_luminance(x).float().clamp_min(eps)
    return torch.log(lum).to(dtype=x.dtype)


def gradient_x(x: torch.Tensor) -> torch.Tensor:
    grad = x[..., :, 1:] - x[..., :, :-1]
    return F.pad(grad, (0, 1, 0, 0), mode="replicate")


def gradient_y(x: torch.Tensor) -> torch.Tensor:
    grad = x[..., 1:, :] - x[..., :-1, :]
    return F.pad(grad, (0, 0, 0, 1), mode="replicate")


def normalize_depth(depth: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    d = depth.float()
    reduce_dims = tuple(range(2, d.ndim))
    d_min = d.amin(dim=reduce_dims, keepdim=True)
    d_max = d.amax(dim=reduce_dims, keepdim=True)
    norm = (d - d_min) / (d_max - d_min).clamp_min(eps)
    return torch.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0).to(dtype=depth.dtype)


def depth_to_normals(depth: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    d = normalize_depth(depth, eps=eps).float()
    dx = gradient_x(d)
    dy = gradient_y(d)
    ones = torch.ones_like(d)
    normals = torch.cat([-dx, -dy, ones], dim=1)
    normals = F.normalize(normals, dim=1, eps=eps)
    return torch.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=depth.dtype)
