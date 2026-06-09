from __future__ import annotations

import argparse
import math
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_ = JpegImagePlugin

from losses.light_transfer import compute_log_luminance_transfer
from models.modules.light_utils import image_to_unit_range, ray_to_light, rotate_ray_z, rgb_to_luminance
from rectified_flow.trajectory_flow import TrajectoryFlow


@dataclass
class EvalResult:
    index: int
    source_meta: dict[str, Any]
    target_meta: dict[str, Any]
    metrics: dict[str, float]
    panels: list[tuple[str, Image.Image]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CP-LightSiT and write a PDF report.")
    parser.add_argument("--checkpoint", default=None, help="Path to CP-LightSiT checkpoint. Defaults to latest run best.pth.")
    parser.add_argument("--config-name", default="TrainCPLightSiT_Minimal")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output", default="outputs/cplightsit_inference_report.pdf")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--good-count", type=int, default=5)
    parser.add_argument("--bad-count", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-gt-source-light", action="store_true")
    parser.add_argument("--sample-progress", action="store_true")
    parser.add_argument("--flow-start-mode", choices=["source", "noise"], default=None)
    parser.add_argument("--edit-noise-scale", type=float, default=None)
    return parser.parse_args()


def resolve_checkpoint(path_value: str | None) -> Path:
    if path_value:
        path = Path(path_value)
        if path.exists():
            return path
        candidates = [
            path.parent / "checkpoint" / path.name,
            path.parent / "checkpoint" / "best.pth",
        ]
        replacement = next((candidate for candidate in candidates if candidate.exists()), None)
        if replacement is not None:
            return replacement
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    pointer = ROOT / "checkpoint" / "latest_CPLightSiT.txt"
    if pointer.exists():
        run_dir = Path(pointer.read_text(encoding="utf-8").strip())
        candidates = [run_dir / "checkpoint" / "best.pth", run_dir / "best.pth", run_dir / "latest.pth"]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    fallback = ROOT / "checkpoint" / "001_CPLightSiT" / "checkpoint" / "best.pth"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Could not find a CP-LightSiT checkpoint. Pass --checkpoint explicitly.")


def load_config(config_name: str) -> DictConfig:
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return hydra.compose(config_name=config_name)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected checkpoint dictionary, got {type(checkpoint).__name__}.")
    return checkpoint


def config_from_checkpoint(checkpoint: dict[str, Any], config_name: str) -> DictConfig:
    if "config" in checkpoint:
        return OmegaConf.create(checkpoint["config"])
    return load_config(config_name)


def load_modules(checkpoint: dict[str, Any], cfg: DictConfig, device: torch.device) -> dict[str, torch.nn.Module]:
    if "tokenizer" in cfg:
        tokenizer = instantiate(cfg.tokenizer)
    else:
        tokenizer = instantiate(
            {
                "_target_": "models.modules.simple_tokenizer.SimpleImageTokenizer",
                "image_size": cfg.image_size,
                "token_grid_size": cfg.token_grid_size,
                "feature_dim": cfg.feature_dim,
            }
        )
    modules: dict[str, torch.nn.Module] = {
        "model": instantiate(cfg.model).to(device).eval(),
        "light_encoder": instantiate(cfg.light_encoder).to(device).eval(),
        "physics_light_transfer": instantiate(cfg.physics_light_transfer).to(device).eval(),
        "light_transfer_transformer": instantiate(cfg.light_transfer_transformer).to(device).eval(),
        "tokenizer": tokenizer.to(device).eval(),
    }
    states = checkpoint.get("models", checkpoint)
    load_report: list[str] = []
    for name, module in modules.items():
        if name not in states:
            continue
        target = module.state_dict()
        source = states[name]
        filtered = {
            key: value
            for key, value in source.items()
            if key in target and torch.is_tensor(value) and tuple(value.shape) == tuple(target[key].shape)
        }
        missing, unexpected = module.load_state_dict(filtered, strict=False)
        load_report.append(f"{name}: loaded={len(filtered)} missing={len(missing)} unexpected={len(unexpected)}")
    print("Checkpoint load report:")
    for line in load_report:
        print(f"  {line}")
    return modules


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_amp_dtype(cfg: DictConfig) -> torch.dtype:
    dtype_name = str(cfg.get("amp_dtype", "bfloat16")).lower()
    if dtype_name in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.bfloat16


def autocast_context(cfg: DictConfig, device: torch.device) -> Any:
    if device.type != "cuda" or not bool(cfg.get("amp_enabled", True)):
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=resolve_amp_dtype(cfg))


def resolve_flow_start_mode(cfg: DictConfig, arg_value: str | None) -> str:
    mode = str(arg_value or cfg.get("flow_start_mode", "source")).lower()
    if mode in {"source", "edit", "editing"}:
        return "source"
    if mode in {"noise", "random", "generation"}:
        return "noise"
    raise ValueError(f"Unsupported flow_start_mode='{mode}'. Expected 'source' or 'noise'.")


def resolve_edit_noise_scale(cfg: DictConfig, arg_value: float | None) -> float:
    value = float(cfg.get("edit_noise_scale", 0.0) if arg_value is None else arg_value)
    return max(value, 0.0)


def make_initial_tokens(
    source_tokens: torch.Tensor,
    flow_start_mode: str,
    edit_noise_scale: float,
) -> torch.Tensor:
    if flow_start_mode == "noise":
        return torch.randn_like(source_tokens)
    z = source_tokens
    if edit_noise_scale > 0.0:
        z = z + edit_noise_scale * torch.randn_like(z)
    return z


def move_tensor_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in batch.items():
        output[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return output


def per_sample_mean(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(1).mean(dim=1)


def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))


def ssim_index(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.float().clamp(0.0, 1.0)
    y = y.float().clamp(0.0, 1.0)
    kernel = 11
    padding = kernel // 2
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(x, kernel, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, kernel, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(x * x, kernel, stride=1, padding=padding) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, kernel, stride=1, padding=padding) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, kernel, stride=1, padding=padding) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ).clamp_min(1e-8)
    return score.flatten(1).mean(dim=1).clamp(-1.0, 1.0)


def cosine_per_sample(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a.float(), dim=1, eps=1e-6)
    b = F.normalize(b.float(), dim=1, eps=1e-6)
    return torch.nan_to_num((a * b).sum(dim=1))


def meta_at(batch: dict[str, Any], key: str, offset: int) -> dict[str, Any]:
    value = batch.get(key, {})
    if not isinstance(value, dict):
        return {}
    output: dict[str, Any] = {}
    for meta_key, meta_value in value.items():
        if isinstance(meta_value, list):
            output[meta_key] = meta_value[offset]
        elif torch.is_tensor(meta_value):
            item = meta_value[offset]
            output[meta_key] = item.item() if item.ndim == 0 else item.detach().cpu().tolist()
        else:
            output[meta_key] = meta_value
    return output


def tensor_to_rgb_image(tensor: torch.Tensor, size: tuple[int, int] = (300, 300)) -> Image.Image:
    unit = image_to_unit_range(tensor.detach().float().cpu()).clamp(0.0, 1.0)
    array = (unit.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array, mode="RGB").resize(size, Image.Resampling.BICUBIC)


def tensor_to_map_image(tensor: torch.Tensor, size: tuple[int, int] = (300, 300), symmetric: bool = False) -> Image.Image:
    value = tensor.detach().float().cpu().squeeze()
    if value.ndim == 3:
        value = value.mean(dim=0)
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    if symmetric:
        scale = value.abs().amax().clamp_min(1e-6)
        norm = (value / scale * 0.5 + 0.5).clamp(0.0, 1.0)
    else:
        v_min = value.amin()
        v_max = value.amax()
        norm = ((value - v_min) / (v_max - v_min).clamp_min(1e-6)).clamp(0.0, 1.0)
    array = (norm.numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array, mode="L").convert("RGB").resize(size, Image.Resampling.BICUBIC)


def build_panels(
    source: torch.Tensor,
    target: torch.Tensor,
    pred: torch.Tensor,
    q_target: torch.Tensor,
    q_model: torch.Tensor,
) -> list[tuple[str, Image.Image]]:
    error = (image_to_unit_range(pred) - image_to_unit_range(target)).abs().mean(dim=0, keepdim=True)
    return [
        ("source", tensor_to_rgb_image(source)),
        ("target", tensor_to_rgb_image(target)),
        ("prediction", tensor_to_rgb_image(pred)),
        ("abs error", tensor_to_map_image(error)),
        ("gt log transfer", tensor_to_map_image(q_target, symmetric=True)),
        ("model transfer", tensor_to_map_image(q_model, symmetric=True)),
    ]


def make_dataset(cfg: DictConfig, split: str, dataset_root: str | None) -> torch.utils.data.Dataset[Any]:
    ds_cfg = cfg.dataset[split]
    if dataset_root is not None:
        with open_dict(ds_cfg):
            ds_cfg.root = dataset_root
    return instantiate(ds_cfg)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> tuple[dict[str, Any], list[EvalResult]]:
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    device = select_device(args.device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    cfg = config_from_checkpoint(checkpoint, args.config_name)
    if args.dataset_root is not None:
        with open_dict(cfg):
            cfg.dataset[args.split].root = args.dataset_root
    modules = load_modules(checkpoint, cfg, device)
    flow = TrajectoryFlow(modules["model"])
    flow_start_mode = resolve_flow_start_mode(cfg, args.flow_start_mode)
    edit_noise_scale = resolve_edit_noise_scale(cfg, args.edit_noise_scale)

    dataset = make_dataset(cfg, args.split, args.dataset_root)
    sample_count = min(int(args.max_samples), len(dataset))
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    indices = indices[:sample_count]
    subset = Subset(dataset, indices)
    dataloader = DataLoader(
        subset,
        batch_size=max(int(args.eval_batch_size), 1),
        shuffle=False,
        num_workers=max(int(args.num_workers), 0),
        pin_memory=torch.cuda.is_available() and device.type == "cuda",
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    results: list[EvalResult] = []
    iterator = tqdm(dataloader, desc="Evaluating", disable=False, dynamic_ncols=True)
    for batch_id, raw_batch in enumerate(iterator):
        batch = move_tensor_batch(raw_batch, device)
        source_image = batch["source_image"]
        target_image = batch["target_image"]
        source_light = batch["source_light"]
        target_light = batch["target_light"]
        source_ray_gt = batch["source_ray"]
        target_ray_gt = batch["target_ray"]
        delta_angle = batch["delta_angle"]
        depth = batch.get("depth")
        depth_valid = batch.get("depth_valid")
        y = batch["y"]

        with autocast_context(cfg, device):
            source_pred = modules["light_encoder"](source_image)
            if args.use_gt_source_light:
                source_ray_used = source_ray_gt
                source_light_used = source_light
            else:
                source_ray_used = source_pred["ray"]
                source_light_used = source_pred["light"]
            target_ray_rotated = rotate_ray_z(source_ray_used, delta_angle)
            target_light_rotated = ray_to_light(target_ray_rotated, target_light[:, 2:3])

            physics = modules["physics_light_transfer"](
                source_image,
                source_light_used,
                target_light_rotated,
                depth,
                source_ray=source_ray_used,
                target_ray=target_ray_rotated,
                depth_valid=depth_valid,
            )
            transfer = modules["light_transfer_transformer"](source_image, source_light_used, target_light_rotated, physics)
            source_tokens = modules["tokenizer"].encode(source_image)
            z = make_initial_tokens(source_tokens, flow_start_mode, edit_noise_scale)
            model_dtype = next(modules["model"].parameters()).dtype
            z = z.to(dtype=model_dtype)
            source_tokens = source_tokens.to(dtype=model_dtype)
            light_cond = torch.cat([source_light_used, target_light_rotated, target_light_rotated - source_light_used], dim=1)
            cond = {"y": y, "light_cond": light_cond, "dense_cond": transfer["dense_cond"], "source_tokens": source_tokens}
            sampled_tokens = flow.sample(
                z,
                cond=cond,
                sample_steps=args.num_steps or int(cfg.diffusion.inference.sample_steps),
                cfg=float(cfg.diffusion.inference.cfg),
                mode=str(cfg.diffusion.inference.mode),
                timestep_shift=float(cfg.diffusion.inference.timestep_shift),
                cfg_mode=str(cfg.diffusion.inference.cfg_mode),
                progress=args.sample_progress,
            )
            pred_image = modules["tokenizer"].decode(sampled_tokens)

        pred_unit = image_to_unit_range(pred_image)
        target_unit = image_to_unit_range(target_image)
        source_unit = image_to_unit_range(source_image)
        abs_error = (pred_unit - target_unit).abs()
        sq_error = (pred_unit - target_unit).square()
        mae = per_sample_mean(abs_error)
        mse = per_sample_mean(sq_error)
        rmse = mse.sqrt()
        psnr = psnr_from_mse(mse)
        ssim = ssim_index(pred_unit, target_unit)
        luma_mae = per_sample_mean((rgb_to_luminance(pred_image) - rgb_to_luminance(target_image)).abs())
        q_target = compute_log_luminance_transfer(source_image, target_image, q_clip=float(cfg.get("q_clip", 2.0)))
        q_pred_image = compute_log_luminance_transfer(source_image, pred_image, q_clip=float(cfg.get("q_clip", 2.0)))
        transfer_l1 = per_sample_mean((q_pred_image - q_target).abs())
        transfer_model_l1 = per_sample_mean((transfer["delta_l"] - q_target).abs())
        source_ray_cos = cosine_per_sample(source_pred["ray"], source_ray_gt)
        target_ray_cos = cosine_per_sample(target_ray_rotated, target_ray_gt)
        source_recon_mae = per_sample_mean((source_unit - pred_unit).abs())
        dense_abs = per_sample_mean(transfer["dense_cond"].abs())
        remove_mean = per_sample_mean(torch.sigmoid(transfer["remove_logits"]))
        create_mean = per_sample_mean(torch.sigmoid(transfer["create_logits"]))
        score = mae + (1.0 - ssim)

        batch_size = source_image.shape[0]
        for offset in range(batch_size):
            global_index = indices[batch_id * int(args.eval_batch_size) + offset]
            metrics = {
                "score": float(score[offset].detach().cpu()),
                "mae": float(mae[offset].detach().cpu()),
                "mse": float(mse[offset].detach().cpu()),
                "rmse": float(rmse[offset].detach().cpu()),
                "psnr": float(psnr[offset].detach().cpu()),
                "ssim": float(ssim[offset].detach().cpu()),
                "luma_mae": float(luma_mae[offset].detach().cpu()),
                "transfer_l1": float(transfer_l1[offset].detach().cpu()),
                "transfer_model_l1": float(transfer_model_l1[offset].detach().cpu()),
                "source_ray_cos": float(source_ray_cos[offset].detach().cpu()),
                "target_ray_cos": float(target_ray_cos[offset].detach().cpu()),
                "source_recon_mae": float(source_recon_mae[offset].detach().cpu()),
                "dense_abs": float(dense_abs[offset].detach().cpu()),
                "remove_mean": float(remove_mean[offset].detach().cpu()),
                "create_mean": float(create_mean[offset].detach().cpu()),
            }
            panels = build_panels(
                source_image[offset],
                target_image[offset],
                pred_image[offset],
                q_target[offset],
                transfer["delta_l"][offset],
            )
            results.append(
                EvalResult(
                    index=global_index,
                    source_meta=meta_at(raw_batch, "source_meta", offset),
                    target_meta=meta_at(raw_batch, "target_meta", offset),
                    metrics=metrics,
                    panels=panels,
                )
            )

    summary = {
        "checkpoint": str(checkpoint_path),
        "config": args.config_name,
        "split": args.split,
        "dataset_root": str(cfg.dataset[args.split].root),
        "sample_count": len(results),
        "sample_steps": args.num_steps or int(cfg.diffusion.inference.sample_steps),
        "use_gt_source_light": bool(args.use_gt_source_light),
        "flow_start_mode": flow_start_mode,
        "edit_noise_scale": edit_noise_scale,
    }
    return summary, results


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_lines(draw: ImageDraw.ImageDraw, lines: list[str], x: int, y: int, font: ImageFont.ImageFont, fill: str = "black", spacing: int = 8) -> int:
    cursor = y
    for line in lines:
        draw.text((x, cursor), line, font=font, fill=fill)
        bbox = draw.textbbox((x, cursor), line, font=font)
        cursor += bbox[3] - bbox[1] + spacing
    return cursor


def metric_stats(results: list[EvalResult]) -> dict[str, dict[str, float]]:
    keys = sorted(results[0].metrics) if results else []
    stats: dict[str, dict[str, float]] = {}
    for key in keys:
        values = torch.tensor([item.metrics[key] for item in results], dtype=torch.float32)
        stats[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(unbiased=False)),
            "median": float(values.median()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return stats


def make_summary_pages(summary: dict[str, Any], results: list[EvalResult]) -> list[Image.Image]:
    width, height = 1800, 1200
    title_font = load_font(42, bold=True)
    header_font = load_font(26, bold=True)
    font = load_font(22)
    small = load_font(18)
    pages: list[Image.Image] = []

    page = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(page)
    draw.text((70, 55), "CP-LightSiT Inference Report", font=title_font, fill="black")
    y = 130
    y = draw_lines(
        draw,
        [
            f"checkpoint: {summary['checkpoint']}",
            f"dataset: {summary['dataset_root']} split={summary['split']}",
            f"samples: {summary['sample_count']}   sample_steps: {summary['sample_steps']}   use_gt_source_light: {summary['use_gt_source_light']}",
            f"flow_start_mode: {summary['flow_start_mode']}   edit_noise_scale: {summary['edit_noise_scale']}",
        ],
        70,
        y,
        font,
    )
    draw.text((70, y + 20), "Metric Summary", font=header_font, fill="black")
    y += 70
    stats = metric_stats(results)
    rows = ["metric                         mean        std      median        min        max"]
    preferred = [
        "score",
        "mae",
        "rmse",
        "psnr",
        "ssim",
        "luma_mae",
        "transfer_l1",
        "transfer_model_l1",
        "source_ray_cos",
        "target_ray_cos",
        "dense_abs",
        "remove_mean",
        "create_mean",
    ]
    for key in preferred:
        if key not in stats:
            continue
        item = stats[key]
        rows.append(
            f"{key:<28} {item['mean']:>8.4f} {item['std']:>8.4f} {item['median']:>8.4f} {item['min']:>8.4f} {item['max']:>8.4f}"
        )
    draw_lines(draw, rows, 70, y, small, spacing=6)
    pages.append(page)
    return pages


def make_sample_page(result: EvalResult, title: str) -> Image.Image:
    width, height = 1800, 1200
    page = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(page)
    title_font = load_font(34, bold=True)
    font = load_font(20)
    small = load_font(17)
    draw.text((60, 40), title, font=title_font, fill="black")
    metric_line = (
        f"idx={result.index}  score={result.metrics['score']:.4f}  MAE={result.metrics['mae']:.4f}  "
        f"PSNR={result.metrics['psnr']:.2f}  SSIM={result.metrics['ssim']:.4f}  "
        f"source_ray_cos={result.metrics['source_ray_cos']:.4f}  target_ray_cos={result.metrics['target_ray_cos']:.4f}"
    )
    draw.text((60, 90), metric_line, font=font, fill="black")
    source_text = f"source: {result.source_meta.get('scene', '?')} {result.source_meta.get('direction', '?')} T={result.source_meta.get('temperature', '?')}"
    target_text = f"target: {result.target_meta.get('scene', '?')} {result.target_meta.get('direction', '?')} T={result.target_meta.get('temperature', '?')}"
    draw.text((60, 125), source_text, font=small, fill="black")
    draw.text((60, 150), target_text, font=small, fill="black")

    start_x, start_y = 60, 205
    gap_x, gap_y = 35, 58
    panel_w, panel_h = 300, 300
    for i, (label, image) in enumerate(result.panels):
        col = i % 3
        row = i // 3
        x = start_x + col * (panel_w + gap_x)
        y = start_y + row * (panel_h + gap_y)
        draw.text((x, y - 28), label, font=font, fill="black")
        page.paste(image, (x, y))
        draw.rectangle((x, y, x + panel_w, y + panel_h), outline="black", width=2)

    detail_x = 1120
    draw.text((detail_x, 205), "All Metrics", font=font, fill="black")
    metric_rows = [f"{key}: {value:.6f}" for key, value in sorted(result.metrics.items())]
    draw_lines(draw, metric_rows, detail_x, 245, small, spacing=5)
    return page


def write_pdf(summary: dict[str, Any], results: list[EvalResult], output: Path, good_count: int, bad_count: int) -> None:
    if not results:
        raise ValueError("No evaluation results were produced.")
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(results, key=lambda item: item.metrics["score"])
    good = ordered[: max(good_count, 0)]
    good_ids = {id(item) for item in good}
    bad_candidates = [item for item in reversed(ordered) if id(item) not in good_ids]
    bad = bad_candidates[: max(bad_count, 0)]
    pages = make_summary_pages(summary, results)
    for rank, item in enumerate(good, start=1):
        pages.append(make_sample_page(item, f"Good Sample {rank}"))
    for rank, item in enumerate(bad, start=1):
        pages.append(make_sample_page(item, f"Bad Sample {rank}"))
    pages[0].save(output, save_all=True, append_images=pages[1:], resolution=150.0)
    print(f"Wrote PDF report: {output}")


def main() -> None:
    args = parse_args()
    summary, results = evaluate(args)
    write_pdf(summary, results, Path(args.output), good_count=args.good_count, bad_count=args.bad_count)


if __name__ == "__main__":
    main()
