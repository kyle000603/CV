from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from models.modules.light_utils import (
    angle_delta_degrees,
    angle_to_ray_vector,
    direction_name_to_angle,
    encode_light,
)


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
VALID_DIRECTIONS = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}


@dataclass(frozen=True)
class VIDITRecord:
    scene: str
    direction: str
    temperature: float
    image_path: Path
    depth_path: Optional[Path] = None
    split: Optional[str] = None
    role: Optional[str] = None


def _is_depth_like(path: Path) -> bool:
    name = path.stem.lower()
    return any(token in name for token in ["depth", "disp", "normal", "mask"])


def _example_error(root: Path, examples: list[Path]) -> ValueError:
    shown = [str(path.relative_to(root)) if path.is_absolute() and path.exists() else str(path) for path in examples[:5]]
    return ValueError(
        "Could not parse VIDIT image filenames. Expected patterns like "
        "'scene001_N_5500.png', 'scene001_dir_N_temp_5500.png', "
        "'0001_N_4500_rgb.png', or 'scene-0001_light-NE_temp-6500.png'. "
        "The NTIRE Track1 challenge layout is also supported when matching source/target folders "
        "exist, for example 'train/input/Image001.png' with 'train/gt/Image001.png' or "
        "'validation/Image301.png' with 'validation_gt/Image301.png'. "
        f"Examples found under root: {shown}."
    )


def _parse_filename(path: Path) -> Optional[tuple[str, str, float]]:
    stem = path.stem
    patterns = [
        r"^(?P<scene>.+?)_dir_(?P<direction>N|NE|E|SE|S|SW|W|NW)_temp_(?P<temp>\d{3,5})(?:_rgb)?$",
        r"^(?P<scene>.+?)_(?P<direction>N|NE|E|SE|S|SW|W|NW)_(?P<temp>\d{3,5})(?:_rgb)?$",
        r"^(?P<scene>.+?)_(?P<temp>\d{3,5})_(?P<direction>N|NE|E|SE|S|SW|W|NW)(?:_rgb)?$",
        r"^(?P<scene>.+?)_light-(?P<direction>N|NE|E|SE|S|SW|W|NW)_temp-(?P<temp>\d{3,5})(?:_rgb)?$",
        r"^(?P<scene>.+?)-light-(?P<direction>N|NE|E|SE|S|SW|W|NW)-temp-(?P<temp>\d{3,5})(?:-rgb)?$",
    ]
    for pattern in patterns:
        match = re.match(pattern, stem, flags=re.IGNORECASE)
        if match is not None:
            direction = match.group("direction").upper()
            return match.group("scene"), direction, float(match.group("temp"))
    tokens = [token for token in re.split(r"[_\-\s]+", stem, flags=re.IGNORECASE) if token]
    direction_index = next((idx for idx, token in enumerate(tokens) if token.upper() in VALID_DIRECTIONS), None)
    temp_index = next((idx for idx, token in enumerate(tokens) if token.isdigit() and 2000 <= int(token) <= 8000), None)
    if direction_index is not None and temp_index is not None:
        scene_tokens = [token for idx, token in enumerate(tokens) if idx not in {direction_index, temp_index}]
        if scene_tokens:
            return "_".join(scene_tokens), tokens[direction_index].upper(), float(tokens[temp_index])
    return None


def _metadata_value(record: dict[str, Any], keys: list[str]) -> Optional[Any]:
    lower_to_key = {key.lower(): key for key in record}
    for key in keys:
        if key in record:
            return record[key]
        lower = key.lower()
        if lower in lower_to_key:
            return record[lower_to_key[lower]]
    return None


def _records_from_json(path: Path, split: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        if split in data and isinstance(data[split], list):
            return [item for item in data[split] if isinstance(item, dict)]
        for key in ["records", "images", "items", "data"]:
            if key in data and isinstance(data[key], list):
                return [item for item in data[key] if isinstance(item, dict)]
    return []


def _records_from_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_record(raw: dict[str, Any], root: Path, split: str, examples: list[Path]) -> Optional[VIDITRecord]:
    raw_split = _metadata_value(raw, ["split", "subset"])
    if raw_split is not None and str(raw_split).lower() != split.lower():
        return None

    image_value = _metadata_value(raw, ["image_path", "path", "file", "filename", "rgb", "image"])
    if image_value is None:
        return None
    image_path = Path(str(image_value))
    if not image_path.is_absolute():
        image_path = root / image_path

    parsed = _parse_filename(image_path)
    scene = _metadata_value(raw, ["scene", "scene_id", "sceneid"]) or (parsed[0] if parsed else None)
    direction = _metadata_value(raw, ["direction", "direction_name", "dir", "light", "light_direction"]) or (
        parsed[1] if parsed else None
    )
    temperature = _metadata_value(raw, ["temperature", "temp", "color_temperature", "cct"]) or (
        parsed[2] if parsed else None
    )
    if scene is None or direction is None or temperature is None:
        raise _example_error(root, examples)

    depth_value = _metadata_value(raw, ["depth_path", "depth", "depth_file"])
    depth_path = None
    if depth_value:
        depth_path = Path(str(depth_value))
        if not depth_path.is_absolute():
            depth_path = root / depth_path

    return VIDITRecord(
        scene=str(scene),
        direction=str(direction).upper(),
        temperature=float(temperature),
        image_path=image_path,
        depth_path=depth_path,
        split=str(raw_split) if raw_split is not None else None,
    )


def _metadata_files(root: Path, split: str) -> list[Path]:
    names = ["metadata.json", "metadata.csv", "train.json", "val.json", "test.json", "index.json"]
    candidates = [root / name for name in names]
    candidates.extend(root / split / name for name in names)
    return [path for path in candidates if path.exists()]


def _load_metadata_records(root: Path, split: str, examples: list[Path]) -> list[VIDITRecord]:
    records: list[VIDITRecord] = []
    for path in _metadata_files(root, split):
        raw_records = _records_from_csv(path) if path.suffix.lower() == ".csv" else _records_from_json(path, split)
        for raw in raw_records:
            record = _to_record(raw, root, split, examples)
            if record is not None:
                records.append(record)
        if records:
            return records
    return records


def _scan_records(root: Path, split: str, examples: list[Path]) -> list[VIDITRecord]:
    all_images = [path for path in root.rglob("*") if path.suffix.lower() in SUPPORTED_EXTENSIONS]
    split_images = [path for path in all_images if split.lower() in {part.lower() for part in path.parts}]
    image_paths = split_images if split_images else all_images
    records: list[VIDITRecord] = []
    for path in image_paths:
        if _is_depth_like(path):
            continue
        parsed = _parse_filename(path)
        if parsed is None:
            continue
        scene, direction, temperature = parsed
        records.append(
            VIDITRecord(
                scene=scene,
                direction=direction,
                temperature=temperature,
                image_path=path,
                depth_path=_find_depth_path(path, root),
                split=split,
            )
        )
    if not records and image_paths:
        raise _example_error(root, examples or image_paths)
    return records


def _find_depth_path(image_path: Path, root: Path) -> Optional[Path]:
    scene = _parse_filename(image_path)
    candidates = [
        image_path.with_suffix(".npy"),
        image_path.with_name(f"{image_path.stem}_depth{image_path.suffix}"),
        image_path.with_name(f"{image_path.stem.replace('_rgb', '')}_depth{image_path.suffix}"),
    ]
    if scene is not None:
        scene_id = scene[0]
        candidates.extend(root.rglob(f"{scene_id}*depth*.png"))
        candidates.extend(root.rglob(f"{scene_id}.npy"))
    for candidate in candidates:
        if candidate.exists() and candidate.suffix.lower() in {*SUPPORTED_EXTENSIONS, ".npy"}:
            return candidate
    return None


def _split_aliases(split: str) -> list[str]:
    lower = split.lower()
    if lower in {"val", "valid", "validation"}:
        return ["validation", "val", "valid"]
    if lower in {"train", "training"}:
        return ["train", "training"]
    if lower == "test":
        return ["test"]
    return [lower]


def _existing_dirs(candidates: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir() and candidate not in seen:
            result.append(candidate)
            seen.add(candidate)
    return result


def _track1_source_dirs(root: Path, split: str) -> list[Path]:
    candidates: list[Path] = []
    for alias in _split_aliases(split):
        base = root / alias
        candidates.extend(
            [
                base / "input",
                base / "inputs",
                base / "source",
                base / "src",
                base,
            ]
        )
    return _existing_dirs(candidates)


def _track1_target_dirs(root: Path, split: str) -> list[Path]:
    candidates: list[Path] = []
    for alias in _split_aliases(split):
        base = root / alias
        candidates.extend(
            [
                base / "gt",
                base / "target",
                base / "targets",
                base / "output",
                root / f"{alias}_gt",
                root / f"{alias}_target",
                root / f"{alias}_targets",
            ]
        )
    return _existing_dirs(candidates)


def _track1_image_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.glob("*")
        if path.suffix.lower() in SUPPORTED_EXTENSIONS and not _is_depth_like(path)
    )


def _load_track1_records(
    root: Path,
    split: str,
    source_direction: str,
    target_direction: str,
    temperature: float,
) -> list[VIDITRecord]:
    source_direction = source_direction.upper()
    target_direction = target_direction.upper()
    if source_direction == target_direction:
        raise ValueError("Track1 pseudo source and target directions must be different.")

    target_dirs = _track1_target_dirs(root, split)
    if not target_dirs:
        return []

    records: list[VIDITRecord] = []
    for source_dir in _track1_source_dirs(root, split):
        source_images = _track1_image_paths(source_dir)
        if not source_images:
            continue
        for target_dir in target_dirs:
            target_by_stem = {path.stem: path for path in _track1_image_paths(target_dir)}
            matched_sources = [path for path in source_images if path.stem in target_by_stem]
            if not matched_sources:
                continue
            for source_path in matched_sources:
                target_path = target_by_stem[source_path.stem]
                scene = f"track1_{source_path.stem}"
                records.append(
                    VIDITRecord(
                        scene=scene,
                        direction=source_direction,
                        temperature=temperature,
                        image_path=source_path,
                        depth_path=_find_depth_path(source_path, root),
                        split=split,
                        role="source",
                    )
                )
                records.append(
                    VIDITRecord(
                        scene=scene,
                        direction=target_direction,
                        temperature=temperature,
                        image_path=target_path,
                        depth_path=None,
                        split=split,
                        role="target",
                    )
                )
            return records
    return records


def _track1_missing_target_hint(root: Path, split: str) -> str:
    source_dirs = _track1_source_dirs(root, split)
    if not source_dirs:
        return ""
    source_count = sum(len(_track1_image_paths(directory)) for directory in source_dirs)
    target_dirs = _track1_target_dirs(root, split)
    target_count = sum(len(_track1_image_paths(directory)) for directory in target_dirs)
    if source_count > 0 and target_count == 0:
        names = ", ".join(str(path.relative_to(root)) for path in source_dirs[:3])
        return (
            f" Found Track1 source images under {names}, but no matching target/gt folder for "
            f"split='{split}'. Training needs paired targets such as 'train/gt/Image001.png' "
            "or a metadata file with target images."
        )
    return ""


def _coerce_depth_array(array: Any) -> np.ndarray:
    if isinstance(array, np.ndarray) and array.dtype != object:
        return array
    item = array.item() if isinstance(array, np.ndarray) and array.shape == () else array
    if isinstance(item, dict):
        for key in ["normalized_depth", "depth", "depth_map", "disparity"]:
            value = item.get(key)
            if value is not None:
                return np.asarray(value)
        for value in item.values():
            value_array = np.asarray(value)
            if value_array.ndim >= 2 and value_array.dtype != object:
                return value_array
    return np.asarray(item)


def _load_records(
    root: Path,
    split: str,
    track1_source_direction: str,
    track1_target_direction: str,
    track1_temperature: float,
) -> list[VIDITRecord]:
    if not root.exists():
        raise FileNotFoundError(f"VIDIT root does not exist: {root}")
    examples = [path for path in root.rglob("*") if path.suffix.lower() in SUPPORTED_EXTENSIONS][:5]
    records = _load_metadata_records(root, split, examples)
    if records:
        return records
    try:
        records = _scan_records(root, split, examples)
    except ValueError as exc:
        track1_records = _load_track1_records(
            root=root,
            split=split,
            source_direction=track1_source_direction,
            target_direction=track1_target_direction,
            temperature=track1_temperature,
        )
        if track1_records:
            return track1_records
        if _track1_missing_target_hint(root, split):
            return []
        raise exc
    if records:
        return records
    return _load_track1_records(
        root=root,
        split=split,
        source_direction=track1_source_direction,
        target_direction=track1_target_direction,
        temperature=track1_temperature,
    )


class VIDITRelightingDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 256,
        mask_size: int = 16,
        fixed_color_temperature: bool = True,
        heldout_directions: Optional[list[str]] = None,
        use_depth: bool = False,
        extended_light: bool = False,
        max_pairs_per_scene: Optional[int] = None,
        track1_source_direction: str = "N",
        track1_target_direction: str = "E",
        track1_temperature: float = 5500.0,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.mask_size = mask_size
        self.fixed_color_temperature = fixed_color_temperature
        self.heldout_directions = {item.upper() for item in heldout_directions or []}
        self.use_depth = use_depth
        self.extended_light = extended_light
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 2.0 - 1.0),
            ]
        )
        self.depth_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )
        records = _load_records(
            self.root,
            split,
            track1_source_direction=track1_source_direction,
            track1_target_direction=track1_target_direction,
            track1_temperature=track1_temperature,
        )
        self.records = self._filter_records(records)
        self.pairs = self._build_pairs(max_pairs_per_scene=max_pairs_per_scene)
        if not self.pairs:
            hint = _track1_missing_target_hint(self.root, split)
            raise ValueError(
                f"No same-scene source-target VIDIT pairs found in {self.root} for split='{split}'. "
                "Need at least two different light directions for the same scene and color temperature."
                f"{hint}"
            )

    def _filter_records(self, records: list[VIDITRecord]) -> list[VIDITRecord]:
        filtered: list[VIDITRecord] = []
        for record in records:
            if record.direction not in VALID_DIRECTIONS:
                continue
            if self.heldout_directions:
                is_heldout = record.direction in self.heldout_directions
                if self.split.lower() == "train" and is_heldout:
                    continue
                if self.split.lower() != "train" and not is_heldout:
                    continue
            filtered.append(record)
        return filtered

    def _build_pairs(self, max_pairs_per_scene: Optional[int]) -> list[tuple[VIDITRecord, VIDITRecord]]:
        groups: dict[tuple[str, Optional[float]], list[VIDITRecord]] = {}
        for record in self.records:
            key = (record.scene, record.temperature if self.fixed_color_temperature else None)
            groups.setdefault(key, []).append(record)

        pairs: list[tuple[VIDITRecord, VIDITRecord]] = []
        for group in groups.values():
            has_roles = any(record.role is not None for record in group)
            if has_roles:
                sources = [record for record in group if record.role == "source"]
                targets = [record for record in group if record.role == "target"]
                group_pairs = [
                    (source, target)
                    for source in sources
                    for target in targets
                    if source.image_path != target.image_path and source.direction != target.direction
                ]
            else:
                group_pairs = [
                    (source, target)
                    for source in group
                    for target in group
                    if source.image_path != target.image_path and source.direction != target.direction
                ]
            group_pairs.sort(key=lambda pair: (pair[0].image_path.as_posix(), pair[1].image_path.as_posix()))
            if max_pairs_per_scene is not None:
                random.Random(0).shuffle(group_pairs)
                group_pairs = group_pairs[:max_pairs_per_scene]
            pairs.extend(group_pairs)
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))

    def _load_depth(self, record: VIDITRecord) -> Optional[torch.Tensor]:
        if not self.use_depth:
            return None
        if record.depth_path is not None and record.depth_path.exists():
            if record.depth_path.suffix.lower() == ".npy":
                try:
                    array = np.load(record.depth_path)
                except ValueError:
                    array = np.load(record.depth_path, allow_pickle=True)
                array = _coerce_depth_array(array)
                tensor = torch.as_tensor(array).float()
                if tensor.ndim == 3 and tensor.shape[-1] <= 4:
                    tensor = tensor.permute(2, 0, 1)
                elif tensor.ndim == 2:
                    tensor = tensor.unsqueeze(0)
                elif tensor.ndim == 3 and tensor.shape[0] > 4:
                    tensor = tensor[:1]
                if tensor.ndim != 3:
                    return torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)
                if tensor.shape[0] > 1:
                    tensor = tensor[:1]
                finite = torch.isfinite(tensor)
                if finite.any():
                    finite_values = tensor[finite]
                    minimum = finite_values.min()
                    maximum = finite_values.max()
                    tensor = torch.where(finite, tensor, minimum)
                    if (maximum - minimum).abs() > 1e-6:
                        tensor = (tensor - minimum) / (maximum - minimum)
                    else:
                        tensor = torch.zeros_like(tensor)
                else:
                    tensor = torch.zeros_like(tensor)
                return F.interpolate(
                    tensor.unsqueeze(0),
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            with Image.open(record.depth_path) as image:
                return self.depth_transform(image.convert("L"))
        return torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)

    def _depth_valid(self, record: VIDITRecord) -> torch.Tensor:
        is_valid = self.use_depth and record.depth_path is not None and record.depth_path.exists()
        return torch.tensor(1.0 if is_valid else 0.0, dtype=torch.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source, target = self.pairs[index]
        source_image = self._load_image(source.image_path)
        target_image = self._load_image(target.image_path)
        source_light = encode_light(source.direction, source.temperature, extended=self.extended_light)
        target_light = encode_light(target.direction, target.temperature, extended=self.extended_light)
        source_angle = direction_name_to_angle(source.direction)
        target_angle = direction_name_to_angle(target.direction)
        delta_angle = angle_delta_degrees(source_angle, target_angle)
        source_ray = angle_to_ray_vector(source_angle)
        target_ray = angle_to_ray_vector(target_angle)
        light_cond = torch.cat([source_light, target_light, target_light - source_light], dim=0)
        depth = self._load_depth(source)
        sample: dict[str, Any] = {
            "source_image": source_image,
            "target_image": target_image,
            "image": target_image,
            "source_light": source_light,
            "target_light": target_light,
            "source_ray": source_ray,
            "target_ray": target_ray,
            "source_angle": torch.tensor(source_angle, dtype=torch.float32),
            "target_angle": torch.tensor(target_angle, dtype=torch.float32),
            "delta_angle": torch.tensor(delta_angle, dtype=torch.float32),
            "light_cond": light_cond,
            "y": torch.tensor(0, dtype=torch.long),
            "mask": torch.ones(self.mask_size * self.mask_size, dtype=torch.float32),
            "depth": depth,
            "depth_valid": self._depth_valid(source),
            "source_meta": {
                "scene": source.scene,
                "direction": source.direction,
                "temperature": source.temperature,
                "image_path": str(source.image_path),
            },
            "target_meta": {
                "scene": target.scene,
                "direction": target.direction,
                "temperature": target.temperature,
                "image_path": str(target.image_path),
            },
        }
        return sample
