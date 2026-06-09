from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.light_utils import ray_to_light


class RayEncoder(nn.Module):
    def __init__(
        self,
        light_dim: int = 3,
        hidden_dim: int = 256,
        extended: bool = False,
        image_size: int = 256,
        patch_size: int = 16,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        direction_count: int = 8,
    ) -> None:
        super().__init__()
        self.light_dim = light_dim
        self.extended = extended
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        self.direction_count = int(direction_count)
        grid_size = self.image_size // self.patch_size
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        self.patch_embed = nn.Conv2d(3, self.hidden_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, grid_size * grid_size + 1, self.hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(self.hidden_dim * float(mlp_ratio)),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(depth))
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.direction_head = nn.Linear(self.hidden_dim, self.direction_count)
        angles = torch.linspace(0.0, 2.0 * math.pi, steps=self.direction_count + 1, dtype=torch.float32)[:-1]
        direction_xy = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        self.register_buffer("direction_xy", direction_xy, persistent=False)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        nn.init.zeros_(self.direction_head.bias)

    def _pos_embed(self, height: int, width: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        patch_h = height // self.patch_size
        patch_w = width // self.patch_size
        token_count = patch_h * patch_w
        if token_count + 1 == self.pos_embed.shape[1]:
            return self.pos_embed.to(device=device, dtype=dtype)
        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        grid = int(math.sqrt(patch_pos.shape[1]))
        patch_pos = patch_pos.transpose(1, 2).reshape(1, self.hidden_dim, grid, grid)
        patch_pos = F.interpolate(patch_pos.float(), size=(patch_h, patch_w), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.reshape(1, self.hidden_dim, token_count).transpose(1, 2)
        return torch.cat([cls_pos.float(), patch_pos], dim=1).to(device=device, dtype=dtype)

    def _direction_to_ray(self, direction_logits: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probabilities = torch.softmax(direction_logits.float(), dim=1)
        direction_xy = self.direction_xy.to(device=direction_logits.device, dtype=torch.float32)
        direction = F.normalize(probabilities @ direction_xy, dim=1, eps=1e-6)
        z = torch.ones(direction.shape[0], 1, dtype=direction.dtype, device=direction.device)
        ray = F.normalize(torch.cat([direction, z], dim=1), dim=1, eps=1e-6).to(dtype=dtype)
        confidence = probabilities.max(dim=1, keepdim=True).values.to(dtype=dtype)
        return ray, direction.to(dtype=dtype), confidence

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        if image.shape[-1] % self.patch_size != 0 or image.shape[-2] % self.patch_size != 0:
            raise ValueError(f"Image size {tuple(image.shape[-2:])} must be divisible by patch_size={self.patch_size}.")
        patches = self.patch_embed(image).flatten(2).transpose(1, 2)
        cls = self.cls_token.to(device=image.device, dtype=patches.dtype).expand(image.shape[0], -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        tokens = tokens + self._pos_embed(image.shape[-2], image.shape[-1], patches.dtype, image.device)
        encoded = self.encoder(tokens)
        cls_features = self.norm(encoded[:, 0].float()).to(dtype=patches.dtype)
        direction_logits = self.direction_head(cls_features.float())
        ray, direction, confidence = self._direction_to_ray(direction_logits, image.dtype)
        temperature = torch.zeros(image.shape[0], 1, dtype=image.dtype, device=image.device)
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
            "direction_logits": direction_logits,
            "temperature": temperature,
            "confidence": confidence,
        }
