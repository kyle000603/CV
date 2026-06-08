from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel

from trainers.assets import ensure_hf_vidit_assets, ensure_vidit_assets, load_pretrained_backbone
from trainers.file_logging import setup_output_log
from trainers.utils import ensure_dir, next_numbered_run_dir, scalar_dict_to_float, state_dict_for_save, unwrap_model


class Trainer:
    def __init__(self, config: DictConfig, models: dict[str, nn.Module]) -> None:
        self.config = config
        self.models = models
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wandb_run: Any = None
        self.pretrained_loaded = False
        self.setup_ddp()
        try:
            self.setup_result_dir()
            self.setup_file_logging()
            self.setup_assets()
            self.setup_datasets()
            self.setup_dataloaders()
            self.setup_losses()
            self.setup_models()
            self.setup_pretrained_models()
            self.wrap_ddp_models()
            self.setup_optimizer()
            self.setup_wandb()
            self.load_checkpoint_if_needed()
        except Exception:
            self.cleanup()
            raise

    def setup_ddp(self) -> None:
        if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device("cuda", self.local_rank)
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()

    def barrier(self) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        if torch.cuda.is_available() and dist.get_backend() == "nccl":
            dist.barrier(device_ids=[self.local_rank])
        else:
            dist.barrier()

    def setup_result_dir(self) -> None:
        if bool(self.config.get("result_dir_prepared_before_ddp", False)):
            self.result_dir = ensure_dir(str(self.config.result_dir))
            if dist.is_available() and dist.is_initialized():
                self.barrier()
            return

        checkpoint_path = self.config.get("checkpoint")
        if checkpoint_path is not None:
            self.result_dir = ensure_dir(Path(str(checkpoint_path)).parent)
            if dist.is_available() and dist.is_initialized():
                self.barrier()
            return

        if bool(self.config.get("use_numbered_result_dir", False)):
            root = self.config.get("result_root", "checkpoint")
            name = str(self.config.get("result_name", self.config.project))
            width = int(self.config.get("result_index_width", 3))
            if self.rank == 0:
                self.result_dir = ensure_dir(next_numbered_run_dir(root=root, name=name, width=width))
                pointer = ensure_dir(root) / f"latest_{name}.txt"
                pointer.write_text(str(self.result_dir), encoding="utf-8")
            if dist.is_available() and dist.is_initialized():
                self.barrier()
                if self.rank != 0:
                    pointer = Path(str(root)) / f"latest_{name}.txt"
                    self.result_dir = ensure_dir(pointer.read_text(encoding="utf-8").strip())
            self.config.result_dir = str(self.result_dir)
            return

        self.result_dir = ensure_dir(str(self.config.result_dir))
        if dist.is_available() and dist.is_initialized():
            self.barrier()

    def setup_file_logging(self) -> None:
        setup_output_log(
            result_dir=self.result_dir,
            rank=self.rank,
            enabled=bool(self.config.get("log_to_file", True)),
            rank_zero_only=bool(self.config.get("log_rank_zero_only", True)),
        )

    def setup_assets(self) -> None:
        if bool(self.config.get("assets_prepared_before_ddp", False)):
            return
        if self.rank == 0:
            ensure_hf_vidit_assets(self.config)
            ensure_vidit_assets(self.config)
        self.barrier()

    def setup_datasets(self) -> None:
        self.train_dataset = instantiate(self.config.dataset.train)
        self.val_dataset = instantiate(self.config.dataset.val)

    def setup_dataloaders(self) -> None:
        self.datamodule = instantiate(self.config.dataloader)
        self.train_dataloader = self.datamodule.get_train_dataloader(self.train_dataset)
        self.val_dataloader = self.datamodule.get_val_dataloader(self.val_dataset)

    def setup_losses(self) -> None:
        self.losses: dict[str, nn.Module] = {}
        for name, cfg in self.config.loss.items():
            self.losses[name] = instantiate(cfg)

    def setup_models(self) -> None:
        for name, model in list(self.models.items()):
            self.models[name] = model.to(self.device)

    def setup_pretrained_models(self) -> None:
        if "model" not in self.models:
            return
        if self.rank == 0:
            self.pretrained_loaded = load_pretrained_backbone(self.config, self.models["model"], self.device)
        if dist.is_available() and dist.is_initialized():
            self.barrier()
            if self.rank != 0:
                self.pretrained_loaded = load_pretrained_backbone(self.config, self.models["model"], self.device)

    def wrap_ddp_models(self) -> None:
        if self.world_size <= 1:
            return
        wrapped_names: list[str] = []
        for name, model in list(self.models.items()):
            if not any(param.requires_grad for param in model.parameters()):
                continue
            self.models[name] = DistributedDataParallel(
                model,
                device_ids=[self.local_rank] if torch.cuda.is_available() else None,
                find_unused_parameters=bool(self.config.get("ddp_find_unused_parameters", False)),
            )
            wrapped_names.append(name)
        if self.rank == 0 and bool(self.config.get("log_ddp_wrapped_modules", True)):
            print(f"DDP wrapping trainable modules: {', '.join(wrapped_names) if wrapped_names else 'none'}")

    def setup_optimizer(self) -> None:
        parameters = [param for model in self.models.values() for param in model.parameters() if param.requires_grad]
        if not parameters:
            raise ValueError("No trainable parameters were found for optimizer setup.")
        optimizer_factory = instantiate(self.config.optimizer.adamw, _convert_="all")
        self.optimizer = optimizer_factory(parameters)
        self.clip_grad_norm = float(self.config.optimizer.parameters.get("clip_grad_norm", 0.0))

    def setup_wandb(self) -> None:
        if self.rank != 0:
            return
        mode = str(self.config.wandb.get("mode", "disabled"))
        if mode == "online" and not os.environ.get("WANDB_API_KEY"):
            mode = "disabled"
        if mode == "disabled":
            return
        try:
            import wandb

            self.wandb_run = wandb.init(
                project=self.config.wandb.get("project", self.config.project),
                entity=self.config.wandb.get("entity", None),
                name=self.config.wandb.get("name", self.config.note),
                mode=mode,
                config=OmegaConf.to_container(self.config, resolve=True),
            )
        except Exception as exc:
            print(f"WandB setup skipped: {exc}")
            self.wandb_run = None

    def load_checkpoint_if_needed(self) -> None:
        checkpoint_path = self.config.get("checkpoint")
        if checkpoint_path is None:
            return
        path = Path(str(checkpoint_path))
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        model_states = checkpoint.get("models", checkpoint)
        for name, model in self.models.items():
            if name in model_states:
                unwrap_model(model).load_state_dict(model_states[name], strict=False)
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.start_epoch = int(checkpoint.get("epoch", -1)) + 1

    def train(self) -> None:
        self.start_epoch = getattr(self, "start_epoch", 0)
        try:
            self.train_process()
        finally:
            self.cleanup()

    def train_process(self) -> None:
        raise NotImplementedError

    def log_wandb(self, values: dict[str, torch.Tensor], step: int) -> None:
        if self.rank != 0:
            return
        floats = scalar_dict_to_float(values)
        if self.wandb_run is not None:
            self.wandb_run.log(floats, step=step)

    def save_checkpoint(self, epoch: int, name: str | None = None) -> None:
        if self.rank != 0:
            return
        filename = name or f"epoch_{epoch:04d}.pth"
        path = self.result_dir / filename
        checkpoint = {
            "models": state_dict_for_save(self.models),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "config": OmegaConf.to_container(self.config, resolve=True),
        }
        torch.save(checkpoint, path)
        torch.save(checkpoint, self.result_dir / "latest.pth")
        self._prune_checkpoints()

    def _prune_checkpoints(self) -> None:
        keep = int(self.config.get("keep_last_checkpoints", 0))
        if keep <= 0:
            return
        checkpoints = sorted(self.result_dir.glob("epoch_*.pth"))
        for old_path in checkpoints[:-keep]:
            old_path.unlink(missing_ok=True)

    def cleanup(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
