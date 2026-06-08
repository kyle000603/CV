from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusersAutoencoderTokenizer(nn.Module):
    """Frozen pretrained AutoencoderKL tokenizer for CP-LightSiT latent training."""

    def __init__(
        self,
        pretrained_model_name_or_path: str = "pretrained/vae/sd-vae-ft-mse",
        image_size: int = 256,
        latent_channels: int = 4,
        downsample_factor: int = 8,
        patch_size: int = 2,
        scaling_factor: float = 0.18215,
        cache_dir: str | None = None,
        local_files_only: bool = True,
        sample_posterior: bool = False,
        torch_dtype: str = "bfloat16",
    ) -> None:
        super().__init__()
        self.pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        self.image_size = int(image_size)
        self.latent_channels = int(latent_channels)
        self.downsample_factor = int(downsample_factor)
        self.patch_size = int(patch_size)
        self.scaling_factor = float(scaling_factor)
        self.cache_dir = None if cache_dir is None else str(cache_dir)
        self.local_files_only = bool(local_files_only)
        self.sample_posterior = bool(sample_posterior)
        self.torch_dtype = str(torch_dtype)
        self.latent_grid_size = self.image_size // self.downsample_factor
        if self.image_size % self.downsample_factor != 0:
            raise ValueError("image_size must be divisible by downsample_factor.")
        if self.latent_grid_size % self.patch_size != 0:
            raise ValueError("VAE latent grid size must be divisible by patch_size.")
        self.token_grid_size = self.latent_grid_size // self.patch_size
        self.feature_dim = self.latent_channels * self.patch_size * self.patch_size
        self._vae: nn.Module | None = None
        self._cplightsit_skip_checkpoint = True

    def _resolve_dtype(self, device: torch.device, fallback: torch.dtype) -> torch.dtype:
        value = self.torch_dtype.lower()
        if device.type != "cuda":
            return torch.float32
        if value in {"auto", "bf16", "bfloat16"}:
            return torch.bfloat16
        if value in {"fp16", "float16", "half"}:
            return torch.float16
        if value in {"fp32", "float32"}:
            return torch.float32
        return fallback

    def _load_vae(self, device: torch.device, dtype: torch.dtype) -> nn.Module:
        if self._vae is not None:
            return self._vae
        try:
            from diffusers import AutoencoderKL
        except ImportError as exc:
            raise RuntimeError(
                "CP-LightSiT now uses a pretrained VAE tokenizer. Install the required packages in the CV env:\n"
                "python -m pip install -U diffusers transformers accelerate safetensors"
            ) from exc

        path_or_repo = self.pretrained_model_name_or_path
        if self.local_files_only and not Path(path_or_repo).exists():
            raise FileNotFoundError(
                f"Pretrained VAE was not found at {path_or_repo}. Run training once with assets.vae_pretrained.enabled=true, "
                "or set tokenizer.local_files_only=false to allow direct Hugging Face loading."
            )
        vae = AutoencoderKL.from_pretrained(
            path_or_repo,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )
        vae.requires_grad_(False)
        vae.eval()
        vae.to(device=device, dtype=self._resolve_dtype(device, dtype))
        self._vae = vae
        return vae

    def _patchify(self, latent: torch.Tensor) -> torch.Tensor:
        patch = self.patch_size
        batch, channels, height, width = latent.shape
        if channels != self.latent_channels:
            raise ValueError(f"Expected {self.latent_channels} latent channels, got {channels}.")
        if height != self.latent_grid_size or width != self.latent_grid_size:
            latent = F.interpolate(latent.float(), size=(self.latent_grid_size, self.latent_grid_size), mode="bilinear", align_corners=False).to(
                dtype=latent.dtype
            )
        patches = latent.reshape(batch, channels, height // patch, patch, width // patch, patch)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(batch, -1, channels * patch * patch)
        return patches

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, token_count, feature_dim = tokens.shape
        if feature_dim != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {feature_dim}.")
        grid = int(math.sqrt(token_count))
        if grid * grid != token_count:
            raise ValueError(f"Token count must be square, got {token_count}.")
        patch = self.patch_size
        latent = tokens.reshape(batch, grid, grid, self.latent_channels, patch, patch)
        latent = latent.permute(0, 3, 1, 4, 2, 5).reshape(batch, self.latent_channels, grid * patch, grid * patch)
        if latent.shape[-1] != self.latent_grid_size:
            latent = F.interpolate(latent.float(), size=(self.latent_grid_size, self.latent_grid_size), mode="bilinear", align_corners=False).to(
                dtype=tokens.dtype
            )
        return latent

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        vae = self._load_vae(image.device, image.dtype)
        vae_dtype = next(vae.parameters()).dtype
        encoded: Any = vae.encode(image.to(device=image.device, dtype=vae_dtype))
        posterior = encoded.latent_dist
        latent = posterior.sample() if self.sample_posterior else posterior.mode()
        latent = latent.to(device=image.device, dtype=vae_dtype) * self.scaling_factor
        return self._patchify(latent)

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        vae = self._load_vae(tokens.device, tokens.dtype)
        vae_dtype = next(vae.parameters()).dtype
        latent = self._unpatchify(tokens).to(dtype=vae_dtype) / self.scaling_factor
        decoded: Any = vae.decode(latent)
        image = decoded.sample.to(device=tokens.device, dtype=tokens.dtype)
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(image.float(), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False).to(dtype=tokens.dtype)
        return image.clamp(-1.0, 1.0)
