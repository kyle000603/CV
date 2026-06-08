from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.modules.light_utils import encode_light, image_to_unit_range, light_to_ray, ray_to_light, rotate_ray_z
from rectified_flow.trajectory_flow import TrajectoryFlow


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample CP-LightSiT relighting.")
    parser.add_argument("--config-name", default="TrainCPLightSiT")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--target-direction", required=True)
    parser.add_argument("--target-temperature", type=float, required=True)
    parser.add_argument("--target-rotation-deg", type=float, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-debug-maps", action="store_true")
    return parser.parse_args()


def _load_config(config_name: str) -> DictConfig:
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return hydra.compose(config_name=config_name)


def _load_checkpoint_blob(path: Path, device: torch.device) -> dict[str, Any]:
    blob = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(blob, dict):
        raise ValueError(f"Checkpoint must contain a dictionary, got {type(blob).__name__}.")
    return blob


def _config_from_checkpoint_or_file(checkpoint: dict[str, Any], config_name: str) -> DictConfig:
    if "config" in checkpoint:
        return OmegaConf.create(checkpoint["config"])
    return _load_config(config_name)


def _load_image(path: Path, image_size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ]
    )
    with Image.open(path) as image:
        return transform(image.convert("RGB")).unsqueeze(0)


def _save_model_image(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = torch.nan_to_num(tensor.detach().cpu(), nan=0.0, posinf=1.0, neginf=-1.0)
    image = image_to_unit_range(safe.squeeze(0)).permute(1, 2, 0).numpy()
    Image.fromarray((image.clip(0.0, 1.0) * 255.0).astype("uint8"), mode="RGB").save(path)


def _save_map(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = torch.nan_to_num(tensor.detach().float().cpu().squeeze(), nan=0.0, posinf=0.0, neginf=0.0)
    value = (value - value.min()) / (value.max() - value.min()).clamp_min(1e-6)
    Image.fromarray((value.numpy() * 255.0).astype("uint8"), mode="L").save(path)


def _load_checkpoint(checkpoint: dict[str, Any], modules: dict[str, torch.nn.Module]) -> None:
    states = checkpoint.get("models", checkpoint)
    for name, module in modules.items():
        if name in states:
            current = module.state_dict()
            filtered = {
                key: value
                for key, value in states[name].items()
                if key in current and tuple(current[key].shape) == tuple(value.shape)
            }
            module.load_state_dict(filtered, strict=False)


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = _load_checkpoint_blob(Path(args.checkpoint), device)
    cfg = _config_from_checkpoint_or_file(checkpoint, args.config_name)

    model = instantiate(cfg.model).to(device).eval()
    light_encoder = instantiate(cfg.light_encoder).to(device).eval()
    physics_light_transfer = instantiate(cfg.physics_light_transfer).to(device).eval()
    light_transfer_transformer = instantiate(cfg.light_transfer_transformer).to(device).eval()
    tokenizer = instantiate(
        {
            "_target_": "models.modules.simple_tokenizer.SimpleImageTokenizer",
            "image_size": cfg.image_size,
            "token_grid_size": cfg.token_grid_size,
            "feature_dim": cfg.feature_dim,
        }
    ).to(device).eval()
    _load_checkpoint(
        checkpoint,
        {
            "model": model,
            "light_encoder": light_encoder,
            "light_transfer_transformer": light_transfer_transformer,
            "tokenizer": tokenizer,
        },
    )

    source_image = _load_image(Path(args.source_image), int(cfg.image_size)).to(device)
    with torch.no_grad():
        source_pred = light_encoder(source_image)
        source_light = source_pred["light"]
        source_ray = source_pred["ray"]
        if args.target_rotation_deg is None:
            target_light = encode_light(args.target_direction, args.target_temperature, extended=False).to(device).unsqueeze(0)
            target_ray = light_to_ray(target_light)
        else:
            delta_angle = torch.tensor([args.target_rotation_deg], dtype=source_ray.dtype, device=device)
            target_ray = rotate_ray_z(source_ray, delta_angle)
            target_temp = torch.tensor(
                [[(args.target_temperature - 5500.0) / 2500.0]],
                dtype=source_ray.dtype,
                device=device,
            )
            target_light = ray_to_light(target_ray, target_temp)
        physics = physics_light_transfer(
            source_image,
            source_light,
            target_light,
            depth=None,
            source_ray=source_ray,
            target_ray=target_ray,
        )
        transfer = light_transfer_transformer(source_image, source_light, target_light, physics)
        source_tokens = tokenizer.encode(source_image)
        z = torch.randn_like(source_tokens)
        light_cond = torch.cat([source_light, target_light, target_light - source_light], dim=1)
        cond = {
            "y": torch.zeros(1, dtype=torch.long, device=device),
            "light_cond": light_cond,
            "dense_cond": transfer["dense_cond"],
            "source_tokens": source_tokens,
        }
        flow = TrajectoryFlow(model)
        sample_steps = args.num_steps or int(cfg.diffusion.inference.sample_steps)
        sampled_tokens = flow.sample(
            z,
            cond=cond,
            sample_steps=sample_steps,
            cfg=float(cfg.diffusion.inference.cfg),
            mode=str(cfg.diffusion.inference.mode),
            timestep_shift=float(cfg.diffusion.inference.timestep_shift),
            cfg_mode=str(cfg.diffusion.inference.cfg_mode),
            progress=True,
        )
        relit = tokenizer.decode(sampled_tokens)

    output_path = Path(args.output)
    _save_model_image(relit, output_path)
    if args.save_debug_maps:
        debug_dir = output_path.parent / f"{output_path.stem}_debug"
        _save_map(physics["H_s"], debug_dir / "H_s.png")
        _save_map(physics["H_t"], debug_dir / "H_t.png")
        _save_map(physics["delta_l_phys"], debug_dir / "delta_l_phys.png")
        _save_map(transfer["delta_l"], debug_dir / "delta_l_refined.png")
        _save_map(torch.sigmoid(transfer["remove_logits"]), debug_dir / "remove_map.png")
        _save_map(torch.sigmoid(transfer["create_logits"]), debug_dir / "create_map.png")


if __name__ == "__main__":
    main()
