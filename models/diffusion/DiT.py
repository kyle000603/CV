from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / max(half, 1))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, device: torch.device) -> torch.Tensor:
    grid_h = torch.arange(grid_size, dtype=torch.float32, device=device)
    grid_w = torch.arange(grid_size, dtype=torch.float32, device=device)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid_tensor = torch.stack(grid, dim=0).reshape(2, 1, grid_size, grid_size)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_tensor[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_tensor[1])
    return torch.cat([emb_h, emb_w], dim=1).unsqueeze(0)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega = 1.0 / (10000 ** (omega / max(embed_dim / 2, 1)))
    out = pos.reshape(-1).unsqueeze(1) * omega.unsqueeze(0)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    if emb.shape[1] < embed_dim:
        emb = F.pad(emb, (0, embed_dim - emb.shape[1]))
    return emb


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float = 0.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.use_cfg_embedding = dropout_prob > 0.0
        embedding_count = num_classes + int(self.use_cfg_embedding)
        self.embedding_table = nn.Embedding(embedding_count, hidden_size)

    def token_drop(self, labels: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout_prob <= 0.0:
            return labels
        drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        labels = labels.clone()
        labels[drop_ids] = self.num_classes
        return labels

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.long().clamp_min(0)
        if self.use_cfg_embedding:
            labels = self.token_drop(labels).clamp_max(self.num_classes)
        else:
            labels = labels.clamp_max(self.num_classes - 1)
        return self.embedding_table(labels)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, qk_norm: bool = False) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, hidden_size))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        attn_input = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_output
        mlp_input = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiT(nn.Module):
    def __init__(
        self,
        input_size: int = 16,
        num_classes: int = 1,
        patch_size: int = 1,
        depth: int = 12,
        hidden_size: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        use_swiglu: bool = False,
        in_channels: int = 392,
        learn_sigma: bool = False,
        qk_norm: bool = True,
        class_dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.hidden_size = hidden_size
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio, qk_norm=qk_norm) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        pos_embed = get_2d_sincos_pos_embed(hidden_size, input_size, device=torch.device("cpu"))
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def _pos_embed(self, token_count: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if token_count == self.input_size * self.input_size:
            return self.pos_embed.to(device=device, dtype=dtype)
        grid = int(math.ceil(math.sqrt(token_count)))
        pos = get_2d_sincos_pos_embed(self.hidden_size, grid, device=device)[:, :token_count]
        return pos.to(dtype=dtype)

    def _condition(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.t_embedder(t) + self.y_embedder(y)

    def _forward_from_embeddings(self, x: torch.Tensor, c: torch.Tensor, skip: list[int]) -> torch.Tensor:
        for index, block in enumerate(self.blocks):
            x = block(x, c)
        return self.final_layer(x, c)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, skip: list[int] = []) -> torch.Tensor:
        token_count = x.shape[1]
        h = self.x_embedder(x) + self._pos_embed(token_count, x.device, x.dtype)
        c = self._condition(t, y)
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

