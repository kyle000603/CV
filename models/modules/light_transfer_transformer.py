from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightTransferTransformer(nn.Module):
    def __init__(
        self,
        in_image_channels: int = 3,
        light_dim: int = 3,
        hidden_size: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        patch_size: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.light_dim = light_dim
        input_channels = in_image_channels + 10
        self.patch_embed = nn.Conv2d(input_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.light_mlp = nn.Sequential(
            nn.Linear(light_dim * 3, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_size, 4, kernel_size=3, padding=1),
        )

    def forward(
        self,
        source_image: torch.Tensor,
        source_light: torch.Tensor,
        target_light: torch.Tensor,
        physics: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        maps = [
            source_image,
            physics["H_s"],
            physics["H_t"],
            physics["delta_l_phys"],
            physics["remove_prior"],
            physics["create_prior"],
            physics["depth"],
            physics["normal"],
            physics["confidence"],
        ]
        features = torch.cat(maps, dim=1)
        batch, _, height, width = features.shape
        tokens = self.patch_embed(features)
        grid_h, grid_w = tokens.shape[-2:]
        tokens = tokens.flatten(2).transpose(1, 2)
        light_pair = torch.cat([source_light, target_light, target_light - source_light], dim=1)
        tokens = tokens + self.light_mlp(light_pair.float()).to(dtype=tokens.dtype).unsqueeze(1)
        tokens = self.transformer(tokens)
        dense = tokens.transpose(1, 2).reshape(batch, -1, grid_h, grid_w)
        dense = F.interpolate(dense, size=(height, width), mode="bilinear", align_corners=False)
        decoded = self.decoder(dense)
        residual_delta_l = decoded[:, 0:1].clamp(-5.0, 5.0)
        remove_logits = decoded[:, 1:2].clamp(-20.0, 20.0)
        create_logits = decoded[:, 2:3].clamp(-20.0, 20.0)
        confidence = torch.sigmoid(decoded[:, 3:4])
        delta_l = physics["delta_l_phys"] + residual_delta_l
        dense_cond = torch.cat(
            [
                delta_l,
                torch.sigmoid(remove_logits),
                torch.sigmoid(create_logits),
                confidence,
                physics["H_s"],
                physics["H_t"],
                physics["depth"],
                physics["normal"],
            ],
            dim=1,
        )
        return {
            "delta_l": torch.nan_to_num(delta_l),
            "remove_logits": torch.nan_to_num(remove_logits),
            "create_logits": torch.nan_to_num(create_logits),
            "confidence": torch.nan_to_num(confidence),
            "dense_cond": torch.nan_to_num(dense_cond),
        }

