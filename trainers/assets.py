from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
HF_VIDIT_MARKER_VERSION = 1


def count_images(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    return sum(1 for item in root.rglob("*") if item.suffix.lower() in IMAGE_EXTENSIONS)


def directory_has_images(path: str | Path) -> bool:
    return count_images(path) > 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_url(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    print(f"Downloading {url} -> {output_path}")
    with urllib.request.urlopen(url) as response, temp_path.open("wb") as handle:
        total = response.length or 0
        seen = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            seen += len(chunk)
            if total:
                percent = 100.0 * seen / total
                print(f"\r  {seen / (1024 ** 2):.1f} / {total / (1024 ** 2):.1f} MiB ({percent:.1f}%)", end="")
    if total:
        print()
    temp_path.replace(output_path)


def ensure_file(url: str, output_path: str | Path, sha256: str | None = None) -> Path:
    path = Path(output_path)
    if path.exists():
        if sha256 is None or _sha256(path) == sha256:
            print(f"Using existing file: {path}")
            return path
        print(f"Checksum mismatch for {path}; re-downloading.")
        path.unlink()
    _download_url(url, path)
    if sha256 is not None:
        actual = _sha256(path)
        if actual != sha256:
            path.unlink(missing_ok=True)
            raise ValueError(f"Checksum mismatch for {path}: expected {sha256}, got {actual}.")
    return path


def ensure_hf_file(
    repo_id: str,
    filename: str,
    output_path: str | Path,
    repo_type: str = "model",
    sha256: str | None = None,
) -> Path:
    path = Path(output_path)
    if path.exists():
        if sha256 is None or _sha256(path) == sha256:
            print(f"Using existing file: {path}")
            return path
        print(f"Checksum mismatch for {path}; re-downloading from Hugging Face.")
        path.unlink()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face Hub support is required for this pretrained checkpoint source. "
            "Install it with: /home/jovyan/irrlab/anaconda3/envs/CV/bin/python -m pip install -U huggingface_hub"
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Hugging Face file {repo_id}/{filename} -> {path}")
    cached_path = Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type))
    temp_path = path.with_suffix(path.suffix + ".part")
    shutil.copyfile(cached_path, temp_path)
    temp_path.replace(path)
    if sha256 is not None:
        actual = _sha256(path)
        if actual != sha256:
            path.unlink(missing_ok=True)
            raise ValueError(f"Checksum mismatch for {path}: expected {sha256}, got {actual}.")
    return path


def _archive_marker_path(archive_path: Path, output_dir: Path) -> Path:
    safe_name = archive_path.name.replace("/", "_")
    return output_dir / ".cplightdit_assets" / f"{safe_name}.json"


def _archive_signature(archive_path: Path) -> dict[str, Any]:
    stat = archive_path.stat()
    return {
        "archive": archive_path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _archive_was_extracted(archive_path: Path, output_dir: Path) -> bool:
    marker = _archive_marker_path(archive_path, output_dir)
    if not marker.exists():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return data == _archive_signature(archive_path)


def _write_archive_marker(archive_path: Path, output_dir: Path) -> None:
    marker = _archive_marker_path(archive_path, output_dir)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(_archive_signature(archive_path), indent=2), encoding="utf-8")


def _archive_basename(url: str, configured_path: Path) -> str:
    name = Path(url.split("?", 1)[0]).name
    return name or configured_path.name


def _resolve_local_archive(url: str, configured_path: Path, root: Path) -> Path | None:
    basename = _archive_basename(url, configured_path)
    candidates = [
        configured_path,
        root / basename,
        root / "archives" / basename,
        Path("data") / "archives" / basename,
        Path("data") / "VIDIT" / basename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _validate_archive(archive: Path) -> None:
    if archive.suffix.lower() == ".zip":
        if not zipfile.is_zipfile(archive):
            raise ValueError(
                f"VIDIT archive is not a valid complete zip file: {archive}. "
                "The file is likely incomplete or corrupted. Re-download it, or verify it with "
                f"'zip -T {archive}' before re-running training."
            )
        return
    if tarfile.is_tarfile(archive):
        return
    raise ValueError(f"Unsupported archive format: {archive}")


def extract_archive(archive_path: str | Path, output_dir: str | Path) -> None:
    archive = Path(archive_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if _archive_was_extracted(archive, output):
        print(f"Using previously extracted archive: {archive}")
        return
    _validate_archive(archive)
    print(f"Extracting {archive} -> {output}")
    if zipfile.is_zipfile(archive):
        try:
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(output)
        except zipfile.BadZipFile as exc:
            raise ValueError(
                f"VIDIT archive extraction failed because the zip is incomplete or corrupted: {archive}. "
                "Re-download this archive and re-run training."
            ) from exc
        _write_archive_marker(archive, output)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            handle.extractall(output)
        _write_archive_marker(archive, output)
        return
    raise ValueError(f"Unsupported archive format: {archive}")


def _hf_vidit_marker_path(root: Path) -> Path:
    return root / ".cplightdit_assets" / "hf_vidit.json"


def _hf_rgb_image_count(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(
        1
        for item in root.rglob("*.png")
        if not item.stem.lower().endswith("_depth") and "_depth" not in item.stem.lower()
    )


def _hf_marker_matches(root: Path, expected: dict[str, Any], min_images: int) -> bool:
    marker = _hf_vidit_marker_path(root)
    if not marker.exists() or _hf_rgb_image_count(root) < min_images:
        return False
    try:
        actual = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return all(actual.get(key) == value for key, value in expected.items())


def _write_hf_marker(root: Path, marker: dict[str, Any]) -> None:
    marker_path = _hf_vidit_marker_path(root)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True), encoding="utf-8")


def _safe_token(value: Any) -> str:
    token = str(value).strip()
    token = token.replace("/", "_").replace("\\", "_")
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", token)
    return token.strip("_") or "unknown"


def _row_value(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    raise KeyError(f"None of the expected keys were found in Hugging Face row: {keys}")


def _save_pil_like_image(value: Any, path: Path, mode: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    image = value
    if isinstance(value, dict) and "bytes" in value:
        from io import BytesIO
        from PIL import Image

        image = Image.open(BytesIO(value["bytes"]))
    elif isinstance(value, dict) and "path" in value:
        from PIL import Image

        image = Image.open(value["path"])
    if mode is not None and hasattr(image, "convert"):
        image = image.convert(mode)
    if not hasattr(image, "save"):
        raise TypeError(f"Hugging Face image value cannot be saved as a PIL image: {type(value).__name__}")
    image.save(path)


def _save_hf_vidit_row(row: dict[str, Any], split_dir: Path) -> tuple[Path, Path]:
    scene = _safe_token(_row_value(row, ["scene", "scene_id", "name"]))
    direction = _safe_token(_row_value(row, ["direction", "dir", "light_direction"])).upper()
    temperature = int(_row_value(row, ["temprature", "temperature", "temp"]))
    stem = f"{scene}_{direction}_{temperature}"
    image_path = split_dir / f"{stem}.png"
    depth_path = split_dir / f"{stem}_depth.png"
    _save_pil_like_image(_row_value(row, ["image", "rgb"]), image_path, mode="RGB")
    _save_pil_like_image(_row_value(row, ["depth_map", "depth", "depth_image"]), depth_path, mode="L")
    return image_path, depth_path


def _split_hf_scenes(scenes: list[str], val_fraction: float, seed: int) -> set[str]:
    if val_fraction <= 0.0:
        return set()
    unique_scenes = sorted(set(scenes))
    rng = random.Random(seed)
    rng.shuffle(unique_scenes)
    val_count = max(1, int(round(len(unique_scenes) * val_fraction)))
    val_count = min(val_count, max(0, len(unique_scenes) - 1))
    return set(unique_scenes[:val_count])


def ensure_hf_vidit_assets(cfg: DictConfig) -> None:
    hf_cfg = cfg.get("assets", {}).get("hf_vidit") if "assets" in cfg else None
    if hf_cfg is None or not bool(hf_cfg.get("enabled", False)):
        return

    root = Path(str(hf_cfg.get("root", cfg.dataset.train.root)))
    dataset_name = str(hf_cfg.get("dataset_name", "Nahrawy/VIDIT-Depth-ControlNet"))
    dataset_split = str(hf_cfg.get("split", "train"))
    cache_dir_value = hf_cfg.get("cache_dir", None)
    cache_dir = None if cache_dir_value is None else str(cache_dir_value)
    val_fraction = float(hf_cfg.get("val_fraction", 0.1))
    seed = int(hf_cfg.get("seed", cfg.get("global_seed", 0)))
    min_images = int(hf_cfg.get("min_images", 100))
    max_items_value = hf_cfg.get("max_items", None)
    max_items = None if max_items_value is None else int(max_items_value)
    force = bool(hf_cfg.get("force", False))
    disable_xet = bool(hf_cfg.get("disable_xet", True))
    if disable_xet:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    marker = {
        "version": HF_VIDIT_MARKER_VERSION,
        "dataset_name": dataset_name,
        "split": dataset_split,
        "val_fraction": val_fraction,
        "seed": seed,
        "max_items": max_items,
        "disable_xet": disable_xet,
    }
    if not force and _hf_marker_matches(root, marker, min_images):
        print(f"Using existing Hugging Face VIDIT conversion under {root} ({_hf_rgb_image_count(root)} RGB images)")
        return

    if force and root.exists():
        print(f"Removing existing Hugging Face VIDIT conversion under {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "assets.hf_vidit.enabled=true requires Hugging Face datasets support. "
            "Install it in the CV environment with: "
            "/home/jovyan/irrlab/anaconda3/envs/CV/bin/python -m pip install -U datasets huggingface_hub pyarrow"
        ) from exc

    print(f"Loading Hugging Face VIDIT dataset: {dataset_name} split={dataset_split}")
    dataset = load_dataset(dataset_name, split=dataset_split, cache_dir=cache_dir)
    if max_items is not None:
        dataset = dataset.select(range(min(max_items, len(dataset))))

    scenes = [str(scene) for scene in dataset["scene"]]
    val_scenes = _split_hf_scenes(scenes, val_fraction=val_fraction, seed=seed)
    converted = 0
    train_count = 0
    val_count = 0
    for row in dataset:
        scene = str(_row_value(row, ["scene", "scene_id", "name"]))
        split_name = "val" if scene in val_scenes else "train"
        _save_hf_vidit_row(row, root / split_name)
        converted += 1
        if split_name == "val":
            val_count += 1
        else:
            train_count += 1
        if converted % 500 == 0:
            print(f"Converted {converted} Hugging Face VIDIT rows...")

    marker.update({"converted": converted, "train_rows": train_count, "val_rows": val_count})
    _write_hf_marker(root, marker)
    print(
        "Prepared Hugging Face VIDIT data: "
        f"{train_count} train rows, {val_count} val rows, root={root}"
    )


def ensure_vidit_assets(cfg: DictConfig) -> None:
    vidit_cfg = cfg.get("assets", {}).get("vidit") if "assets" in cfg else None
    if vidit_cfg is None or not bool(vidit_cfg.get("enabled", False)):
        return

    root = Path(str(vidit_cfg.get("root", cfg.dataset.train.root)))
    min_images = int(vidit_cfg.get("min_images", 1))
    image_count = count_images(root)
    local_archives_exist = any(root.glob("*.zip")) or any((root / "archives").glob("*.zip"))
    if image_count >= min_images and not local_archives_exist:
        print(f"Using existing VIDIT data under {root} ({image_count} images)")
        return

    archives = vidit_cfg.get("archives", [])
    if len(archives) == 0:
        raise ValueError("assets.vidit.enabled=true but assets.vidit.archives is empty.")

    for archive_cfg in archives:
        url = str(archive_cfg["url"])
        path = Path(str(archive_cfg.get("path", root / "archives" / Path(url).name)))
        extract_to = Path(str(archive_cfg.get("extract_to", root)))
        checksum = archive_cfg.get("sha256")
        local_archive = _resolve_local_archive(url, path, root)
        try:
            if local_archive is not None:
                if checksum is not None and _sha256(local_archive) != str(checksum):
                    raise ValueError(f"Checksum mismatch for existing archive: {local_archive}")
                print(f"Using existing VIDIT archive: {local_archive}")
                archive_path = local_archive
            else:
                archive_path = ensure_file(url, path, sha256=str(checksum) if checksum else None)
        except (OSError, urllib.error.URLError) as exc:
            raise RuntimeError(
                "VIDIT automatic download failed. This usually means the current machine or container "
                "cannot reach the configured VIDIT server. Manually place the extracted VIDIT data under "
                f"'{root}', or place the archive at '{path}' or '{root / _archive_basename(url, path)}' "
                "and re-run, or disable automatic VIDIT download with 'assets.vidit.enabled=false' after "
                "setting dataset.train.root and dataset.val.root to your local VIDIT directory. "
                f"Original error: {exc}"
            ) from exc
        extract_archive(archive_path, extract_to)

    image_count = count_images(root)
    if image_count < min_images:
        raise FileNotFoundError(
            f"VIDIT download/extraction finished, but only {image_count} images were found under {root}; "
            f"expected at least {min_images}. "
            "Set assets.vidit.root or dataset.*.root to the extracted VIDIT image directory."
        )


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ["model", "ema", "state_dict", "module"]:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return {str(k).removeprefix("module."): v for k, v in value.items() if torch.is_tensor(v)}
        return {str(k).removeprefix("module."): v for k, v in checkpoint.items() if torch.is_tensor(v)}
    raise ValueError(f"Unsupported checkpoint object type: {type(checkpoint).__name__}")


def _compatible_state_dict(
    target: nn.Module,
    source_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    target_state = target.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in source_state.items():
        if key in target_state and tuple(target_state[key].shape) == tuple(value.shape):
            compatible[key] = value
        else:
            skipped.append(key)
    return compatible, skipped


def _pretrained_cfg(cfg: DictConfig) -> Any:
    if "assets" not in cfg:
        return None
    if "sit_pretrained" in cfg.assets:
        return cfg.assets.sit_pretrained
    if "dit_pretrained" in cfg.assets:
        return cfg.assets.dit_pretrained
    return None


def ensure_pretrained_checkpoint(cfg: DictConfig) -> Path | None:
    dit_cfg = _pretrained_cfg(cfg)
    if dit_cfg is None or not bool(dit_cfg.get("enabled", False)):
        return None
    path = Path(str(dit_cfg["path"]))
    checksum = dit_cfg.get("sha256")
    checksum_value = str(checksum) if checksum else None
    if path.exists():
        if checksum_value is None or _sha256(path) == checksum_value:
            print(f"Using existing file: {path}")
            return path
        print(f"Checksum mismatch for {path}; re-downloading.")
        path.unlink()

    errors: list[str] = []
    hf_repo_id = dit_cfg.get("hf_repo_id", None)
    hf_filename = dit_cfg.get("hf_filename", None)
    if hf_repo_id is not None and hf_filename is not None:
        try:
            return ensure_hf_file(
                repo_id=str(hf_repo_id),
                filename=str(hf_filename),
                output_path=path,
                repo_type=str(dit_cfg.get("hf_repo_type", "model")),
                sha256=checksum_value,
            )
        except Exception as exc:
            errors.append(f"Hugging Face source failed: {exc}")

    url = dit_cfg.get("url", None)
    if url is not None:
        try:
            return ensure_file(str(url), path, sha256=checksum_value)
        except Exception as exc:
            errors.append(f"URL source failed: {exc}")

    if bool(dit_cfg.get("optional", False)):
        print(
            "Pretrained checkpoint could not be prepared; continuing without it. "
            + " ".join(errors)
        )
        return None

    detail = " ".join(errors) if errors else "No pretrained checkpoint source is configured."
    raise RuntimeError(
        f"Failed to prepare pretrained checkpoint at {path}. {detail} "
        "You can manually place the file there, set assets.sit_pretrained.optional=true to train from scratch, "
        "or disable it with assets.sit_pretrained.enabled=false."
    )


def load_pretrained_backbone(cfg: DictConfig, model: nn.Module, device: torch.device) -> bool:
    dit_cfg = _pretrained_cfg(cfg)
    if dit_cfg is None or not bool(dit_cfg.get("enabled", False)):
        return False
    checkpoint_path = ensure_pretrained_checkpoint(cfg)
    if checkpoint_path is None:
        return False
    if not bool(dit_cfg.get("load", True)):
        return False

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source_state = _extract_state_dict(checkpoint)
    compatible, skipped = _compatible_state_dict(model, source_state)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(
        "Loaded pretrained backbone weights: "
        f"{len(compatible)} tensors matched, {len(skipped)} skipped, "
        f"{len(missing)} missing, {len(unexpected)} unexpected."
    )
    if len(compatible) == 0 and bool(dit_cfg.get("require_match", False)):
        raise RuntimeError(
            "No tensors from the downloaded DiT checkpoint matched this CP-LightDiT model. "
            "Use a token-space CP-LightDiT checkpoint or set assets.sit_pretrained.require_match=false."
        )
    return len(compatible) > 0


def cfg_to_container(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
