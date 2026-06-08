from __future__ import annotations

import hashlib
import json
import time
import traceback
from pathlib import Path
from typing import Any

import os

import hydra
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from trainers.assets import ensure_hf_vidit_assets, ensure_pretrained_checkpoint, ensure_vidit_assets
from trainers.base import Trainer
from trainers.file_logging import setup_output_log
from trainers.utils import ensure_dir, next_numbered_run_dir

os.environ["HYDRA_FULL_ERROR"] = "1"


def _launcher_rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def _launcher_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _asset_preflight_root(cfg: DictConfig) -> Path:
    assets = cfg.get("assets", {})
    hf_cfg = assets.get("hf_vidit") if "hf_vidit" in assets else None
    if hf_cfg is not None and bool(hf_cfg.get("enabled", False)):
        return Path(str(hf_cfg.get("root", cfg.dataset.train.root)))
    vidit_cfg = assets.get("vidit") if "vidit" in assets else None
    if vidit_cfg is not None and bool(vidit_cfg.get("enabled", False)):
        return Path(str(vidit_cfg.get("root", cfg.dataset.train.root)))
    return Path("data")


def _asset_preflight_paths(cfg: DictConfig) -> tuple[Path, Path]:
    assets_payload = OmegaConf.to_container(cfg.get("assets", {}), resolve=True)
    assets_hash = hashlib.sha256(json.dumps(assets_payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    launcher_hash = _launcher_hash()
    marker_dir = _asset_preflight_root(cfg) / ".cplightsit_assets" / "preflight"
    stem = f"{launcher_hash}_{assets_hash}"
    return marker_dir / f"{stem}.ready", marker_dir / f"{stem}.error"


def _launcher_hash() -> str:
    launcher_payload = "|".join(
        [
            os.environ.get("TORCHELASTIC_RUN_ID", ""),
            os.environ.get("MASTER_ADDR", ""),
            os.environ.get("MASTER_PORT", ""),
            os.environ.get("WORLD_SIZE", "1"),
        ]
    )
    return hashlib.sha256(launcher_payload.encode("utf-8")).hexdigest()[:12]


def _write_marker(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_result_dir(cfg: DictConfig) -> Path:
    checkpoint_path = cfg.get("checkpoint")
    if checkpoint_path is not None:
        return ensure_dir(Path(str(checkpoint_path)).parent)

    if bool(cfg.get("use_numbered_result_dir", False)):
        root = cfg.get("result_root", "checkpoint")
        name = str(cfg.get("result_name", cfg.project))
        width = int(cfg.get("result_index_width", 3))
        result_dir = ensure_dir(next_numbered_run_dir(root=root, name=name, width=width))
        pointer = ensure_dir(root) / f"latest_{name}.txt"
        pointer.write_text(str(result_dir), encoding="utf-8")
        return result_dir

    return ensure_dir(str(cfg.result_dir))


def _prepare_result_dir_once_before_ddp(cfg: DictConfig) -> Path:
    rank = _launcher_rank()
    world_size = _launcher_world_size()
    if world_size > 1 and rank != 0:
        return Path(str(cfg.result_dir))

    result_dir = _resolve_result_dir(cfg)
    with open_dict(cfg):
        cfg.result_dir = str(result_dir)
        cfg.result_dir_prepared_before_ddp = True
    return result_dir


def _setup_training_output_log(cfg: DictConfig) -> None:
    setup_output_log(
        result_dir=cfg.result_dir,
        rank=_launcher_rank(),
        enabled=bool(cfg.get("log_to_file", True)),
        rank_zero_only=bool(cfg.get("log_rank_zero_only", True)),
    )


def _prepare_assets_once_before_ddp(cfg: DictConfig) -> None:
    ready_path, error_path = _asset_preflight_paths(cfg)
    rank = _launcher_rank()
    world_size = _launcher_world_size()
    timeout_seconds = float(cfg.get("asset_preflight_timeout_seconds", 0.0))

    if world_size <= 1 or rank == 0:
        ready_path.unlink(missing_ok=True)
        error_path.unlink(missing_ok=True)
        try:
            ensure_hf_vidit_assets(cfg)
            ensure_vidit_assets(cfg)
            if str(cfg.get("stage", "")) != "ray_pretrain":
                checkpoint_path = ensure_pretrained_checkpoint(cfg)
                if checkpoint_path is not None:
                    print(f"Prepared pretrained checkpoint: {checkpoint_path}")
            _write_marker(ready_path, {"status": "ready", "time": time.time()})
        except Exception as exc:
            _write_marker(
                error_path,
                {
                    "status": "error",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "time": time.time(),
                },
            )
            raise
        return

    start_time = time.time()
    last_notice = 0.0
    while True:
        if ready_path.exists():
            return
        if error_path.exists():
            try:
                error_payload = json.loads(error_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                error_payload = {"error": f"Could not read {error_path}"}
            raise RuntimeError(
                "Rank 0 failed while preparing CP-LightSiT assets before DDP setup: "
                f"{error_payload.get('error', 'unknown error')}"
            )
        elapsed = time.time() - start_time
        if timeout_seconds > 0 and elapsed > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for asset preflight marker: {ready_path}")
        if not bool(cfg.get("silent_nonzero_rank_wait", True)) and elapsed - last_notice >= 30.0:
            print(f"Waiting for rank 0 to prepare CP-LightSiT assets before DDP setup... ({elapsed:.0f}s)")
            last_notice = elapsed
        time.sleep(5.0)


@hydra.main(version_base=None, config_path="configs", config_name="TrainCPLightSiT")
def main(cfg: DictConfig) -> None:
    _unused: Any = None
    _prepare_result_dir_once_before_ddp(cfg)
    _setup_training_output_log(cfg)
    _prepare_assets_once_before_ddp(cfg)
    with open_dict(cfg):
        cfg.assets_prepared_before_ddp = True
    trainer_factory = instantiate(cfg.trainer)
    model: nn.Module = instantiate(cfg.model)
    trainer: Trainer = trainer_factory(config=cfg, model=model)
    trainer.train()


if __name__ == "__main__":
    main()
