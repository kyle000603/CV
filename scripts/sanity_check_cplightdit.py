from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainers.utils import count_trainable_parameters, unwrap_model


def _write_demo_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0, 255, 96, dtype=np.uint8)
    x = np.tile(grid[None, :], (96, 1))
    y = np.tile(grid[:, None], (1, 96))
    image = np.stack(
        [
            ((x.astype(np.int16) + color[0]) % 256).astype(np.uint8),
            ((y.astype(np.int16) + color[1]) % 256).astype(np.uint8),
            np.full_like(x, color[2]),
        ],
        axis=-1,
    )
    Image.fromarray(image, mode="RGB").save(path)
    Image.fromarray(np.tile(grid[:, None], (1, 96)), mode="L").save(path.with_name(f"{path.stem}_depth.png"))


def _create_demo_vidit(root: Path) -> None:
    for split in ["train", "val", "test"]:
        split_root = root / split
        _write_demo_image(split_root / "scene001_N_5500.png", (20, 40, 80))
        _write_demo_image(split_root / "scene001_E_5500.png", (80, 20, 100))
        _write_demo_image(split_root / "scene001_S_5500.png", (120, 60, 40))


def _prepare_sanity_config(cfg: DictConfig) -> None:
    temp_root = Path(tempfile.mkdtemp(prefix="cplightdit_vidit_"))
    _create_demo_vidit(temp_root)
    cfg.dataset.train.root = str(temp_root)
    cfg.dataset.val.root = str(temp_root)
    cfg.dataset.test.root = str(temp_root)
    cfg.result_root = tempfile.mkdtemp(prefix="cplightdit_sanity_ckpt_")
    cfg.result_name = "Sanity"
    cfg.wandb.mode = "disabled"
    cfg.assets.hf_vidit.enabled = False
    cfg.assets.vidit.enabled = False
    cfg.assets.sit_pretrained.enabled = False
    cfg.allow_freeze_without_pretrain = True
    cfg.stage = "cplightdit_finetune"
    cfg.loss_mode = "minimal"
    cfg.enable_image_space_losses = False
    cfg.decode_loss_every = 0
    cfg.debug_one_batch = True
    cfg.epochs = 1

    cfg.image_size = 64
    cfg.mask_size = 8
    cfg.batch_size = 2
    cfg.num_workers = 0
    cfg.dataloader.global_batch_size = 2
    cfg.dataloader.num_workers = 0
    cfg.dataloader.drop_last = False
    cfg.dataloader.pin_memory = False
    cfg.dataloader.persistent_workers = False
    cfg.dataloader.prefetch_factor = 2
    cfg.dataset.train.image_size = 64
    cfg.dataset.train.mask_size = 8
    cfg.feature_dim = 64
    cfg.token_grid_size = 8
    cfg.token_count = 64
    cfg.model.input_size = 8
    cfg.model.in_channels = 64
    cfg.model.depth = 2
    cfg.model.hidden_size = 128
    cfg.model.num_heads = 4
    cfg.model.mlp_ratio = 2.0
    cfg.light_encoder.hidden_dim = 64
    cfg.light_transfer_transformer.hidden_size = 64
    cfg.light_transfer_transformer.depth = 1
    cfg.light_transfer_transformer.patch_size = 8


def _assert_finite(name: str, value: Any) -> None:
    if torch.is_tensor(value):
        assert torch.isfinite(value).all(), f"{name} contains non-finite values"
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(f"{name}.{key}", item)


@hydra.main(version_base=None, config_path="../configs", config_name="TrainCPLightDiT_Minimal")
def main(cfg: DictConfig) -> None:
    _prepare_sanity_config(cfg)
    trainer_factory = instantiate(cfg.trainer)
    model = instantiate(cfg.model)
    trainer = trainer_factory(config=cfg, model=model)
    batch = next(iter(trainer.train_dataloader))
    losses = trainer.train_single_step(batch)
    _assert_finite("losses", losses)
    expected_total = float(cfg.lambda_flow) * losses["Train/flow/loss"] + float(cfg.lambda_transfer) * losses["Train/transfer/loss"]
    assert torch.allclose(losses["Train/total"], expected_total, atol=1e-5), "Minimal total loss formula mismatch."

    print(f"CP-LightDiT trainable parameters: {count_trainable_parameters(unwrap_model(trainer.models['model'])):,}")
    print(f"RayEncoder trainable parameters: {count_trainable_parameters(unwrap_model(trainer.models['light_encoder'])):,}")
    print(
        "LightTransferTransformer trainable parameters: "
        f"{count_trainable_parameters(unwrap_model(trainer.models['light_transfer_transformer'])):,}"
    )
    print(f"Tokenizer trainable parameters: {count_trainable_parameters(unwrap_model(trainer.models['tokenizer'])):,}")
    print("CP-LightDiT minimal sanity check passed.")
    trainer.cleanup()


if __name__ == "__main__":
    main()
