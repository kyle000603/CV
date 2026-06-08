from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion.DiT import DiT


class CPLightSiT(DiT):
    def __init__(
        self,
        *args: object,
        light_dim: int = 9,
        dense_cond_channels: int = 18,
        use_dense_condition: bool = True,
        use_source_tokens: bool = True,
        light_drop_prob: float = 0.0,
        dense_drop_prob: float = 0.0,
        source_drop_prob: float = 0.0,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.light_dim = light_dim
        self.dense_cond_channels = dense_cond_channels
        self.use_dense_condition = use_dense_condition
        self.use_source_tokens = use_source_tokens
        self.light_drop_prob = light_drop_prob
        self.dense_drop_prob = dense_drop_prob
        self.source_drop_prob = source_drop_prob
        self.light_mlp = nn.Sequential(nn.Linear(light_dim, self.hidden_size), nn.SiLU(), nn.Linear(self.hidden_size, self.hidden_size))
        self.dense_proj = nn.Linear(dense_cond_channels, self.hidden_size)
        self.source_proj = nn.Linear(self.in_channels, self.hidden_size)
        self.initialize_condition_adapters()

    def initialize_condition_adapters(self) -> None:
        """Initialize new condition adapters for stable frozen-backbone finetuning."""
        nn.init.zeros_(self.dense_proj.weight)
        nn.init.zeros_(self.dense_proj.bias)
        nn.init.zeros_(self.source_proj.weight)
        nn.init.zeros_(self.source_proj.bias)
        for module in self.light_mlp:
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)
        last_linear = next((module for module in reversed(self.light_mlp) if isinstance(module, nn.Linear)), None)
        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)

    def _drop_condition(self, tensor: torch.Tensor, probability: float) -> torch.Tensor:
        if not self.training or probability <= 0.0:
            return tensor
        keep = (torch.rand(tensor.shape[0], device=tensor.device) >= probability).to(dtype=tensor.dtype)
        shape = [tensor.shape[0]] + [1] * (tensor.ndim - 1)
        return tensor * keep.view(*shape)

    def _dense_to_tokens(self, dense_cond: torch.Tensor, token_count: int, dtype: torch.dtype) -> torch.Tensor:
        grid = int(math.ceil(math.sqrt(token_count)))
        dense = F.interpolate(dense_cond.float(), size=(grid, grid), mode="bilinear", align_corners=False)
        tokens = dense.flatten(2).transpose(1, 2)[:, :token_count]
        return tokens.to(dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        skip: list[int] = [],
        light_cond: Optional[torch.Tensor] = None,
        dense_cond: Optional[torch.Tensor] = None,
        source_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if light_cond is None and dense_cond is None and source_tokens is None:
            return super().forward(x, t, y, skip=skip)

        token_count = x.shape[1]
        h = self.x_embedder(x) + self._pos_embed(token_count, x.device, x.dtype)
        c = self._condition(t, y)

        if light_cond is not None:
            light = self._drop_condition(light_cond.to(device=x.device, dtype=x.dtype), self.light_drop_prob)
            c = c + self.light_mlp(light.float()).to(dtype=c.dtype)

        if self.use_dense_condition and dense_cond is not None:
            dense = self._drop_condition(dense_cond.to(device=x.device, dtype=x.dtype), self.dense_drop_prob)
            dense_tokens = self._dense_to_tokens(dense, token_count, x.dtype)
            h = h + self.dense_proj(dense_tokens.float()).to(dtype=h.dtype)

        if self.use_source_tokens and source_tokens is not None:
            source = self._drop_condition(source_tokens.to(device=x.device, dtype=x.dtype), self.source_drop_prob)
            h = h + self.source_proj(source.float()).to(dtype=h.dtype)

        return self._forward_from_embeddings(h, c, skip)

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float = 1.0,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(x, t, y, **kwargs)
