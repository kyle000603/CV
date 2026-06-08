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
    temp_root = Path(tempfile.mkdtemp(prefix="cplightsit_vidit_"))
    _create_demo_vidit(temp_root)
    cfg.dataset.train.root = str(temp_root)
    cfg.dataset.val.root = str(temp_root)
    cfg.dataset.test.root = str(temp_root)
    cfg.result_root = tempfile.mkdtemp(prefix="cplightsit_sanity_ckpt_")
    cfg.result_name = "Sanity"
    cfg.wandb.mode = "disabled"
    cfg.assets.hf_vidit.enabled = False
    cfg.assets.vidit.enabled = False
    cfg.assets.sit_pretrained.enabled = False
    cfg.assets.vae_pretrained.enabled = False
    cfg.allow_freeze_without_pretrain = True
    cfg.stage = "cplightsit_finetune"
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
    cfg.tokenizer = {
        "_target_": "models.modules.simple_tokenizer.SimpleImageTokenizer",
        "image_size": 64,
        "token_grid_size": 8,
        "feature_dim": 64,
    }
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


def _assert_dense_transfer_contract(trainer: Any, cfg: DictConfig, batch: dict[str, torch.Tensor]) -> None:
    moved = trainer._move_tensor_batch(batch)
    source_image = moved["source_image"]
    source_light = moved["source_light"]
    target_light = moved["target_light"]
    source_ray = moved.get("source_ray")
    target_ray = moved.get("target_ray")
    depth = moved.get("depth")
    depth_valid = moved.get("depth_valid")
    assert torch.is_tensor(source_image)
    assert torch.is_tensor(source_light)
    assert torch.is_tensor(target_light)
    physics = trainer.physics_light_transfer(
        source_image,
        source_light,
        target_light,
        depth if torch.is_tensor(depth) else None,
        source_ray=source_ray if torch.is_tensor(source_ray) else None,
        target_ray=target_ray if torch.is_tensor(target_ray) else None,
        depth_valid=depth_valid if torch.is_tensor(depth_valid) else None,
    )
    if int(cfg.light_transfer_transformer.output_dense_channels) == 18:
        for key in ["source_ray_map", "target_ray_map", "source_response", "target_response"]:
            assert key in physics, f"PhysicsLightTransfer missing {key}"
        assert int(cfg.model.dense_cond_channels) == 18, "CP-LightSiT dense_cond_channels should be 18."
    transfer = trainer.light_transfer_transformer(source_image, source_light, target_light, physics)
    expected_channels = int(cfg.light_transfer_transformer.output_dense_channels)
    assert transfer["dense_cond"].shape[1] == expected_channels, "Dense condition channel mismatch."


def _assert_ray_encoder_checkpoint_loader(trainer: Any, cfg: DictConfig) -> None:
    checkpoint_path = Path(str(cfg.result_root)) / "dummy_ray_encoder.pth"
    torch.save({"light_encoder": unwrap_model(trainer.models["light_encoder"]).state_dict()}, checkpoint_path)
    cfg.ray_encoder_checkpoint = str(checkpoint_path)
    trainer.load_ray_encoder_checkpoint_if_needed()
    assert count_trainable_parameters(unwrap_model(trainer.models["light_encoder"])) == 0, "RayEncoder should remain frozen."


@hydra.main(version_base=None, config_path="../configs", config_name="TrainCPLightSiT_Minimal")
def main(cfg: DictConfig) -> None:
    _prepare_sanity_config(cfg)
    trainer_factory = instantiate(cfg.trainer)
    model = instantiate(cfg.model)
    trainer = trainer_factory(config=cfg, model=model)
    batch = next(iter(trainer.train_dataloader))
    _assert_dense_transfer_contract(trainer, cfg, batch)
    _assert_ray_encoder_checkpoint_loader(trainer, cfg)
    losses = trainer.train_single_step(batch)
    _assert_finite("losses", losses)
    expected_total = float(cfg.lambda_flow) * losses["Train/flow/loss"] + float(cfg.lambda_transfer) * losses["Train/transfer/loss"]
    assert torch.allclose(losses["Train/total"], expected_total, atol=1e-5), "Minimal total loss formula mismatch."

    print(f"CP-LightSiT trainable parameters: {count_trainable_parameters(unwrap_model(trainer.models['model'])):,}")
    print(f"RayEncoder trainable parameters: {count_trainable_parameters(unwrap_model(trainer.models['light_encoder'])):,}")
    print(
        "LightTransferTransformer trainable parameters: "
        f"{count_trainable_parameters(unwrap_model(trainer.models['light_transfer_transformer'])):,}"
    )
    print(f"Tokenizer trainable parameters: {count_trainable_parameters(unwrap_model(trainer.models['tokenizer'])):,}")
    print("CP-LightSiT minimal sanity check passed.")
    trainer.cleanup()


if __name__ == "__main__":
    main()
