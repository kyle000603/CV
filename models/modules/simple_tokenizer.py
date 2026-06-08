from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleImageTokenizer(nn.Module):
    def __init__(
        self,
        image_size: int = 256,
        token_grid_size: int = 16,
        feature_dim: int = 392,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.token_grid_size = token_grid_size
        self.feature_dim = feature_dim
        patch_size = image_size // token_grid_size
        if image_size % token_grid_size != 0:
            raise ValueError("image_size must be divisible by token_grid_size.")
        self.patch_size = patch_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, feature_dim, kernel_size=patch_size, stride=patch_size),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 64, kernel_size=patch_size, stride=patch_size),
            nn.SiLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
        )

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(image)
        return tokens.flatten(2).transpose(1, 2)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, token_count, feature_dim = tokens.shape
        if feature_dim != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {feature_dim}.")
        grid = int(math.sqrt(token_count))
        if grid * grid != token_count:
            raise ValueError(f"Token count must be square, got {token_count}.")
        maps = tokens.transpose(1, 2).reshape(batch, feature_dim, grid, grid)
        image = self.decoder(maps)
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(image, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return torch.tanh(image)

