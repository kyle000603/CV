from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion.DiT import DiT


class ConditionCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.query_norm(x.float()).to(dtype=x.dtype)
        kv = self.context_norm(context.float()).to(dtype=x.dtype)
        attended, _ = self.attn(q, kv, kv, need_weights=False)
        x = x + attended
        return x + self.ffn(self.ffn_norm(x.float())).to(dtype=x.dtype)


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
        use_cross_attention_condition: bool = True,
        cross_attention_layers: int = 2,
        cross_attention_heads: Optional[int] = None,
        cross_attention_dropout: float = 0.0,
        cross_attention_mlp_ratio: float = 2.0,
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
        self.use_cross_attention_condition = use_cross_attention_condition
        self.cross_attention_layers = max(int(cross_attention_layers), 0)
        self.light_mlp = nn.Sequential(nn.Linear(light_dim, self.hidden_size), nn.SiLU(), nn.Linear(self.hidden_size, self.hidden_size))
        self.dense_proj = nn.Linear(dense_cond_channels, self.hidden_size)
        self.source_proj = nn.Linear(self.in_channels, self.hidden_size)
        self.cross_light_proj = nn.Linear(light_dim, self.hidden_size)
        self.cross_dense_proj = nn.Linear(dense_cond_channels, self.hidden_size)
        self.cross_source_proj = nn.Linear(self.in_channels, self.hidden_size)
        self.cross_type_embed = nn.Embedding(3, self.hidden_size)
        heads = int(cross_attention_heads or self.blocks[0].attn.num_heads)
        self.cross_attn_blocks = nn.ModuleList(
            [
                ConditionCrossAttentionBlock(
                    hidden_size=self.hidden_size,
                    num_heads=heads,
                    dropout=cross_attention_dropout,
                    mlp_ratio=cross_attention_mlp_ratio,
                )
                for _ in range(self.cross_attention_layers)
            ]
        )
        self.cross_attn_insert_indices = self._make_cross_attention_insert_indices(self.cross_attention_layers)
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

    def _make_cross_attention_insert_indices(self, count: int) -> list[int]:
        if count <= 0:
            return []
        depth = len(self.blocks)
        if count == 1:
            return [0]
        return [min(depth - 1, round(i * depth / count)) for i in range(count)]

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

    def _build_cross_attention_context(
        self,
        token_count: int,
        dtype: torch.dtype,
        device: torch.device,
        light_cond: Optional[torch.Tensor],
        dense_cond: Optional[torch.Tensor],
        source_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor | None:
        context_tokens: list[torch.Tensor] = []
        type_embeddings = self.cross_type_embed.weight.to(device=device, dtype=dtype)
        if light_cond is not None:
            light = self._drop_condition(light_cond.to(device=device, dtype=dtype), self.light_drop_prob)
            light_token = self.cross_light_proj(light.float()).to(dtype=dtype).unsqueeze(1)
            context_tokens.append(light_token + type_embeddings[0].view(1, 1, -1))
        if self.use_dense_condition and dense_cond is not None:
            dense = self._drop_condition(dense_cond.to(device=device, dtype=dtype), self.dense_drop_prob)
            dense_tokens = self._dense_to_tokens(dense, token_count, dtype)
            context_tokens.append(self.cross_dense_proj(dense_tokens.float()).to(dtype=dtype) + type_embeddings[1].view(1, 1, -1))
        if self.use_source_tokens and source_tokens is not None:
            source = self._drop_condition(source_tokens.to(device=device, dtype=dtype), self.source_drop_prob)
            context_tokens.append(self.cross_source_proj(source.float()).to(dtype=dtype) + type_embeddings[2].view(1, 1, -1))
        if not context_tokens:
            return None
        return torch.cat(context_tokens, dim=1)

    def _forward_with_cross_attention(
        self,
        h: torch.Tensor,
        c: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        cross_index = 0
        insert_indices = self.cross_attn_insert_indices
        for block_index, block in enumerate(self.blocks):
            while cross_index < len(insert_indices) and insert_indices[cross_index] == block_index:
                h = self.cross_attn_blocks[cross_index](h, context)
                cross_index += 1
            h = block(h, c)
        return self.final_layer(h, c)

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

        if self.use_cross_attention_condition and len(self.cross_attn_blocks) > 0:
            context = self._build_cross_attention_context(token_count, h.dtype, h.device, light_cond, dense_cond, source_tokens)
            if context is not None:
                return self._forward_with_cross_attention(h, c, context)

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
