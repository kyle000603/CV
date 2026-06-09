from __future__ import annotations

import argparse
import json
import math
import random
import sys
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

from models.modules.light_utils import DIRECTION_TO_ANGLE, direction_name_to_angle, image_to_unit_range

DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


@dataclass
class RayEvalResult:
    index: int
    image_path: str
    scene: str
    gt_direction: str
    pred_direction: str
    gt_angle: float
    pred_angle: float
    angle_error: float
    ray_cosine: float
    confidence: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RayEncoder and write a PDF report.")
    parser.add_argument("--checkpoint", default=None, help="Path to ray_encoder_best.pth. Defaults to latest RayEncoder run.")
    parser.add_argument("--config-name", default="TrainRayEncoder")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output", default="ray_encoder_report.pdf")
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means evaluate the whole split.")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--good-count", type=int, default=5)
    parser.add_argument("--bad-count", type=int, default=5)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_checkpoint(path_value: str | None) -> Path:
    if path_value:
        path = Path(path_value)
        if path.exists():
            return path
        candidates = [
            path.parent / "checkpoint" / path.name,
            path.parent / "checkpoint" / "ray_encoder_best.pth",
        ]
        replacement = next((candidate for candidate in candidates if candidate.exists()), None)
        if replacement is not None:
            return replacement
        raise FileNotFoundError(f"RayEncoder checkpoint does not exist: {path}")

    pointer = ROOT / "checkpoint" / "latest_RayEncoder.txt"
    if pointer.exists():
        run_dir = Path(pointer.read_text(encoding="utf-8").strip())
        candidates = [
            run_dir / "checkpoint" / "ray_encoder_best.pth",
            run_dir / "ray_encoder_best.pth",
            run_dir / "ray_encoder_latest.pth",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    fallback = ROOT / "checkpoint" / "001_RayEncoder" / "checkpoint" / "ray_encoder_best.pth"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Could not find a RayEncoder checkpoint. Pass --checkpoint explicitly.")


def load_config(config_name: str) -> DictConfig:
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return hydra.compose(config_name=config_name)


def config_from_checkpoint(checkpoint: dict[str, Any], config_name: str) -> DictConfig:
    if "config" in checkpoint:
        return OmegaConf.create(checkpoint["config"])
    return load_config(config_name)


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def nearest_direction(angle_deg: torch.Tensor) -> torch.Tensor:
    direction_angles = torch.tensor([DIRECTION_TO_ANGLE[name] for name in DIRECTIONS], dtype=torch.float32, device=angle_deg.device)
    delta = (angle_deg[:, None] - direction_angles[None, :] + 180.0) % 360.0 - 180.0
    return delta.abs().argmin(dim=1)


def circular_abs_error(pred_angle: torch.Tensor, gt_angle: torch.Tensor) -> torch.Tensor:
    return ((pred_angle - gt_angle + 180.0) % 360.0 - 180.0).abs()


def meta_at(batch: dict[str, Any], offset: int) -> dict[str, Any]:
    value = batch.get("source_meta", {})
    if not isinstance(value, dict):
        return {}
    output: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, list):
            output[key] = item[offset]
        elif torch.is_tensor(item):
            tensor = item[offset]
            output[key] = tensor.item() if tensor.ndim == 0 else tensor.detach().cpu().tolist()
        else:
            output[key] = item
    return output


def tensor_to_rgb_image(tensor: torch.Tensor, size: tuple[int, int] = (320, 320)) -> Image.Image:
    unit = image_to_unit_range(tensor.detach().float().cpu()).clamp(0.0, 1.0)
    array = (unit.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array, mode="RGB").resize(size, Image.Resampling.BICUBIC)


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


def draw_lines(draw: ImageDraw.ImageDraw, lines: list[str], x: int, y: int, font: ImageFont.ImageFont, fill: str = "black") -> int:
    cursor = y
    for line in lines:
        draw.text((x, cursor), line, font=font, fill=fill)
        cursor += int(font.size * 1.35) if hasattr(font, "size") else 24
    return cursor


def summarize(values: torch.Tensor) -> dict[str, float]:
    values = values.float()
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p10": float(values.quantile(0.1)),
        "p90": float(values.quantile(0.9)),
    }


def make_dataset(cfg: DictConfig, split: str, dataset_root: str | None) -> torch.utils.data.Dataset[Any]:
    ds_cfg = cfg.dataset[split]
    with open_dict(ds_cfg):
        ds_cfg.preload_images = False
        ds_cfg.repeat_factor = 1
        if dataset_root is not None:
            ds_cfg.root = dataset_root
    return instantiate(ds_cfg)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> tuple[dict[str, Any], list[RayEvalResult]]:
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected checkpoint dictionary, got {type(checkpoint).__name__}.")
    cfg = config_from_checkpoint(checkpoint, args.config_name)
    device = select_device(args.device)
    model = instantiate(cfg.light_encoder).to(device).eval()
    state = checkpoint.get("light_encoder")
    if state is None and isinstance(checkpoint.get("models"), dict):
        state = checkpoint["models"].get("light_encoder")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint does not contain RayEncoder weights: {checkpoint_path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"RayEncoder checkpoint: {checkpoint_path}")
    print(f"Loaded RayEncoder weights: missing={len(missing)} unexpected={len(unexpected)}")

    dataset = make_dataset(cfg, args.split, args.dataset_root)
    sample_count = len(dataset) if int(args.max_samples) <= 0 else min(int(args.max_samples), len(dataset))
    indices = list(range(len(dataset)))
    if sample_count < len(indices):
        random.Random(args.seed).shuffle(indices)
        indices = indices[:sample_count]
    subset = Subset(dataset, indices)
    dataloader = DataLoader(
        subset,
        batch_size=max(int(args.eval_batch_size), 1),
        shuffle=False,
        num_workers=max(int(args.num_workers), 0),
        pin_memory=device.type == "cuda",
    )

    results: list[RayEvalResult] = []
    confusion = torch.zeros(len(DIRECTIONS), len(DIRECTIONS), dtype=torch.int64)
    ray_cosines: list[torch.Tensor] = []
    angle_errors: list[torch.Tensor] = []
    confidences: list[torch.Tensor] = []
    correct: list[torch.Tensor] = []

    for batch_id, raw_batch in enumerate(tqdm(dataloader, desc="Evaluating RayEncoder", dynamic_ncols=True)):
        image = raw_batch["source_image"].to(device, non_blocking=True)
        gt_ray = raw_batch["source_ray"].to(device, non_blocking=True).float()
        gt_angle = raw_batch["source_angle"].to(device, non_blocking=True).float()
        pred = model(image)
        pred_ray = F.normalize(pred["ray"].float(), dim=1, eps=1e-6)
        gt_ray = F.normalize(gt_ray, dim=1, eps=1e-6)
        ray_cos = (pred_ray * gt_ray).sum(dim=1).clamp(-1.0, 1.0)
        pred_angle = torch.rad2deg(torch.atan2(pred_ray[:, 1], pred_ray[:, 0])) % 360.0
        angle_error = circular_abs_error(pred_angle, gt_angle)
        pred_idx = nearest_direction(pred_angle)
        gt_idx = nearest_direction(gt_angle)
        is_correct = pred_idx == gt_idx
        confidence = pred["confidence"].float().flatten()

        ray_cosines.append(ray_cos.detach().cpu())
        angle_errors.append(angle_error.detach().cpu())
        confidences.append(confidence.detach().cpu())
        correct.append(is_correct.float().detach().cpu())

        for gt_item, pred_item in zip(gt_idx.detach().cpu(), pred_idx.detach().cpu()):
            confusion[int(gt_item), int(pred_item)] += 1

        batch_size = image.shape[0]
        for offset in range(batch_size):
            meta = meta_at(raw_batch, offset)
            global_index = indices[batch_id * max(int(args.eval_batch_size), 1) + offset]
            gt_name = str(meta.get("direction", DIRECTIONS[int(gt_idx[offset])]))
            results.append(
                RayEvalResult(
                    index=global_index,
                    image_path=str(meta.get("image_path", "")),
                    scene=str(meta.get("scene", "")),
                    gt_direction=gt_name,
                    pred_direction=DIRECTIONS[int(pred_idx[offset])],
                    gt_angle=float(gt_angle[offset].detach().cpu()),
                    pred_angle=float(pred_angle[offset].detach().cpu()),
                    angle_error=float(angle_error[offset].detach().cpu()),
                    ray_cosine=float(ray_cos[offset].detach().cpu()),
                    confidence=float(confidence[offset].detach().cpu()),
                )
            )

    ray_tensor = torch.cat(ray_cosines)
    angle_tensor = torch.cat(angle_errors)
    confidence_tensor = torch.cat(confidences)
    correct_tensor = torch.cat(correct)
    summary = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "dataset_root": str(cfg.dataset[args.split].root),
        "sample_count": len(results),
        "accuracy_8way": float(correct_tensor.mean()),
        "ray_cosine": summarize(ray_tensor),
        "angle_error_deg": summarize(angle_tensor),
        "confidence": summarize(confidence_tensor),
        "confusion": confusion.tolist(),
    }
    return summary, results


def draw_confusion_matrix(draw: ImageDraw.ImageDraw, confusion: list[list[int]], x: int, y: int) -> None:
    font = load_font(22)
    bold = load_font(22, bold=True)
    cell = 74
    draw.text((x + cell, y - 38), "Predicted direction", font=bold, fill="black")
    for col, name in enumerate(DIRECTIONS):
        draw.text((x + (col + 1) * cell + 16, y), name, font=bold, fill="black")
    for row, name in enumerate(DIRECTIONS):
        draw.text((x, y + (row + 1) * cell + 20), name, font=bold, fill="black")

    max_value = max(max(row) for row in confusion) if confusion else 1
    for row, values in enumerate(confusion):
        row_sum = max(sum(values), 1)
        for col, value in enumerate(values):
            intensity = int(255 - 170 * (value / max(max_value, 1)))
            color = (255, intensity, intensity) if row != col else (intensity, 255, intensity)
            x0 = x + (col + 1) * cell
            y0 = y + (row + 1) * cell
            draw.rectangle((x0, y0, x0 + cell - 4, y0 + cell - 4), fill=color, outline="gray")
            percent = value / row_sum * 100.0
            draw.text((x0 + 8, y0 + 14), str(value), font=font, fill="black")
            draw.text((x0 + 8, y0 + 40), f"{percent:.0f}%", font=load_font(16), fill="black")


def load_result_image(result: RayEvalResult) -> Image.Image:
    path = Path(result.image_path)
    if path.exists():
        with Image.open(path) as image:
            return image.convert("RGB").resize((320, 320), Image.Resampling.BICUBIC)
    placeholder = Image.new("RGB", (320, 320), "lightgray")
    draw = ImageDraw.Draw(placeholder)
    draw.text((20, 145), "image missing", font=load_font(24), fill="black")
    return placeholder


def make_summary_page(summary: dict[str, Any]) -> Image.Image:
    page = Image.new("RGB", (1800, 1200), "white")
    draw = ImageDraw.Draw(page)
    title_font = load_font(52, bold=True)
    font = load_font(28)
    small = load_font(22)
    draw.text((70, 55), "RayEncoder Evaluation Report", font=title_font, fill="black")
    ray = summary["ray_cosine"]
    angle = summary["angle_error_deg"]
    conf = summary["confidence"]
    lines = [
        f"Checkpoint: {summary['checkpoint']}",
        f"Split: {summary['split']}  |  Samples: {summary['sample_count']}",
        f"8-way direction accuracy: {summary['accuracy_8way'] * 100:.2f}%",
        f"Ray cosine: mean {ray['mean']:.4f}, median {ray['median']:.4f}, p10 {ray['p10']:.4f}, p90 {ray['p90']:.4f}",
        f"Angular error: mean {angle['mean']:.2f} deg, median {angle['median']:.2f} deg, p90 {angle['p90']:.2f} deg",
        f"Confidence: mean {conf['mean']:.4f}, median {conf['median']:.4f}",
    ]
    draw_lines(draw, lines, 70, 145, font)
    draw.text((70, 420), "Confusion Matrix", font=load_font(34, bold=True), fill="black")
    draw.text((70, 462), "Rows are ground truth, columns are predicted. Diagonal cells are correct.", font=small, fill="dimgray")
    draw_confusion_matrix(draw, summary["confusion"], 110, 545)
    return page


def make_sample_page(title: str, samples: list[RayEvalResult]) -> Image.Image:
    page = Image.new("RGB", (1800, 1200), "white")
    draw = ImageDraw.Draw(page)
    draw.text((70, 55), title, font=load_font(46, bold=True), fill="black")
    font = load_font(23)
    small = load_font(19)
    for i, result in enumerate(samples):
        row = i // 2
        col = i % 2
        x = 70 + col * 875
        y = 145 + row * 350
        image = load_result_image(result)
        page.paste(image, (x, y))
        lines = [
            f"scene: {result.scene}",
            f"GT: {result.gt_direction} ({result.gt_angle:.0f} deg)",
            f"Pred: {result.pred_direction} ({result.pred_angle:.1f} deg)",
            f"angle error: {result.angle_error:.1f} deg",
            f"ray cosine: {result.ray_cosine:.4f}",
            f"confidence: {result.confidence:.3f}",
        ]
        draw_lines(draw, lines, x + 340, y + 8, font)
        path = Path(result.image_path).name
        draw.text((x, y + 325), path[:64], font=small, fill="dimgray")
    return page


def write_outputs(summary: dict[str, Any], results: list[RayEvalResult], args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best = sorted(results, key=lambda item: item.angle_error)[: max(int(args.good_count), 0)]
    worst = sorted(results, key=lambda item: item.angle_error, reverse=True)[: max(int(args.bad_count), 0)]
    pages = [make_summary_page(summary)]
    if best:
        pages.append(make_sample_page(f"Best {len(best)} Samples by Angular Error", best))
    if worst:
        pages.append(make_sample_page(f"Worst {len(worst)} Samples by Angular Error", worst))
    pages[0].save(output_path, save_all=True, append_images=pages[1:])
    print(f"Wrote RayEncoder PDF report: {output_path}")

    metrics_path = Path(args.metrics_output) if args.metrics_output else output_path.with_suffix(".json")
    payload = {
        "summary": summary,
        "best": [result.__dict__ for result in best],
        "worst": [result.__dict__ for result in worst],
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote RayEncoder metrics: {metrics_path}")


def main() -> None:
    args = parse_args()
    summary, results = evaluate(args)
    write_outputs(summary, results, args)


if __name__ == "__main__":
    main()
