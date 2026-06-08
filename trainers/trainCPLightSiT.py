from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from losses.light_transfer import compute_dense_transfer_loss
from models.modules.light_utils import light_to_ray, ray_to_light, rotate_ray_z
from models.modules.simple_tokenizer import SimpleImageTokenizer
from rectified_flow.trajectory_flow import TrajectoryFlow
from trainers.base import Trainer
from trainers.utils import count_parameters, count_trainable_parameters, set_requires_grad, unwrap_model


class CPLightSiTTrainer(Trainer):
    def __init__(
        self,
        config: DictConfig,
        model: nn.Module,
        validation_start_epoch: int = 50,
        sample_seed: Optional[int] = None,
        visual_seed: Optional[int] = None,
    ) -> None:
        self.validation_start_epoch = validation_start_epoch
        self.sample_seed = sample_seed
        self.visual_seed = visual_seed
        self.stage = str(config.get("stage", "cplightsit_finetune"))
        self.loss_mode = str(config.get("loss_mode", "minimal"))
        if self.loss_mode != "minimal":
            raise ValueError("Only loss_mode='minimal' is supported. Old auxiliary losses were removed.")

        light_encoder = instantiate(config.light_encoder)
        physics_light_transfer = instantiate(config.physics_light_transfer)
        light_transfer_transformer = instantiate(config.light_transfer_transformer)
        tokenizer = SimpleImageTokenizer(
            image_size=int(config.image_size),
            token_grid_size=int(config.token_grid_size),
            feature_dim=int(config.feature_dim),
        )
        self.physics_light_transfer = physics_light_transfer
        super().__init__(
            config=config,
            models={
                "model": model,
                "light_encoder": light_encoder,
                "light_transfer_transformer": light_transfer_transformer,
                "tokenizer": tokenizer,
            },
        )
        self.model = self.models["model"]
        self.light_encoder = self.models["light_encoder"]
        self.light_transfer_transformer = self.models["light_transfer_transformer"]
        self.tokenizer = unwrap_model(self.models["tokenizer"])
        self.diffusion = TrajectoryFlow(self.model)
        self.global_step = 0

    def setup_models(self) -> None:
        super().setup_models()
        self.physics_light_transfer = self.physics_light_transfer.to(self.device)
        self._apply_trainable_policy()
        self._print_trainable_parameter_counts()

    def setup_pretrained_models(self) -> None:
        if self.stage == "ray_pretrain":
            return
        super().setup_pretrained_models()
        self.load_ray_encoder_checkpoint_if_needed()
        if bool(self.config.get("freeze_backbone", False)) and not self.pretrained_loaded:
            message = "WARNING: freeze_backbone=True but no pretrained checkpoint was loaded. This is not true fine-tuning."
            if not bool(self.config.get("allow_freeze_without_pretrain", False)):
                raise RuntimeError(message + " Set allow_freeze_without_pretrain=true to allow this explicitly.")
            if self.rank == 0:
                print(message)

    def _extract_ray_encoder_state_dict(self, checkpoint: object) -> dict[str, torch.Tensor]:
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Unsupported RayEncoder checkpoint type: {type(checkpoint).__name__}")
        candidates = [
            checkpoint.get("light_encoder"),
            checkpoint.get("models", {}).get("light_encoder") if isinstance(checkpoint.get("models"), dict) else None,
            checkpoint.get("state_dict"),
            checkpoint,
        ]
        for candidate in candidates:
            if isinstance(candidate, dict):
                tensors = {str(key).removeprefix("module."): value for key, value in candidate.items() if torch.is_tensor(value)}
                if tensors:
                    return tensors
        raise ValueError("RayEncoder checkpoint did not contain tensor weights.")

    def load_ray_encoder_checkpoint_if_needed(self) -> None:
        """Load only RayEncoder weights from a separate checkpoint for finetuning."""
        if self.stage != "cplightsit_finetune":
            return
        checkpoint_value = self.config.get("ray_encoder_checkpoint", None)
        if checkpoint_value is None or str(checkpoint_value).strip() == "":
            return
        checkpoint_path = Path(str(checkpoint_value))
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"RayEncoder checkpoint does not exist: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        source_state = self._extract_ray_encoder_state_dict(checkpoint)
        target = unwrap_model(self.models["light_encoder"])
        target_state = target.state_dict()
        compatible: dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        for key, value in source_state.items():
            if key in target_state and tuple(value.shape) == tuple(target_state[key].shape):
                compatible[key] = value
            else:
                skipped.append(key)
        if not compatible:
            raise RuntimeError(f"No compatible RayEncoder tensors were found in {checkpoint_path}.")
        missing, unexpected = target.load_state_dict(compatible, strict=False)
        self._apply_trainable_policy()
        if self.rank == 0:
            print(f"RayEncoder checkpoint path: {checkpoint_path}")
            print(f"RayEncoder loaded keys: {len(compatible)}")
            print(f"RayEncoder skipped keys: {len(skipped)}")
            print(f"RayEncoder missing keys: {list(missing)}")
            print(f"RayEncoder unexpected keys: {list(unexpected)}")

    def setup_optimizer(self) -> None:
        param_groups = self._optimizer_param_groups()
        if not param_groups:
            raise ValueError("No trainable parameters were found for optimizer setup.")
        optimizer_factory = instantiate(self.config.optimizer.adamw, _convert_="all")
        self.optimizer = optimizer_factory(param_groups)
        self.clip_grad_norm = float(self.config.optimizer.parameters.get("clip_grad_norm", 0.0))

    def _apply_trainable_policy(self) -> None:
        model = unwrap_model(self.models["model"])
        light_encoder = unwrap_model(self.models["light_encoder"])
        light_transfer = unwrap_model(self.models["light_transfer_transformer"])
        tokenizer = unwrap_model(self.models["tokenizer"])

        if self.stage == "ray_pretrain":
            set_requires_grad(model, False)
            set_requires_grad(light_encoder, True)
            set_requires_grad(light_transfer, False)
            set_requires_grad(tokenizer, False)
            return
        if self.stage != "cplightsit_finetune":
            raise ValueError(f"Unsupported stage='{self.stage}'. Expected 'cplightsit_finetune' or 'ray_pretrain'.")

        if bool(self.config.get("freeze_backbone", True)):
            set_requires_grad(model, False)
            if bool(self.config.get("train_condition_adapters_only", True)):
                for name in ["light_mlp", "dense_proj", "source_proj"]:
                    module = getattr(model, name, None)
                    if isinstance(module, nn.Module):
                        set_requires_grad(module, True)

        freeze_ray = bool(self.config.get("freeze_ray_encoder", self.config.get("freeze_light_encoder", True)))
        set_requires_grad(light_encoder, not freeze_ray)
        set_requires_grad(light_transfer, bool(self.config.get("train_light_transfer_transformer", True)))
        set_requires_grad(tokenizer, not bool(self.config.get("freeze_tokenizer", True)))

    def _module_trainable_count(self, name: str) -> int:
        return count_trainable_parameters(unwrap_model(self.models[name]))

    def _print_trainable_parameter_counts(self) -> None:
        if self.rank != 0:
            return
        model = unwrap_model(self.models["model"])
        print(f"CP-LightSiT parameters: total={count_parameters(model):,}, trainable={count_trainable_parameters(model):,}")
        print(f"RayEncoder trainable parameters: {self._module_trainable_count('light_encoder'):,}")
        print(f"LightTransferTransformer trainable parameters: {self._module_trainable_count('light_transfer_transformer'):,}")
        print(f"Tokenizer trainable parameters: {self._module_trainable_count('tokenizer'):,}")

    def _named_trainable_params(self, module: nn.Module) -> list[nn.Parameter]:
        return [param for param in module.parameters() if param.requires_grad]

    def _optimizer_param_groups(self) -> list[dict[str, object]]:
        groups: list[dict[str, object]] = []
        base_lr = float(self.config.get("lr", self.config.optimizer.adamw.get("lr", 1e-4)))
        backbone_lr = float(self.config.get("backbone_lr", base_lr))
        adapter_lr = float(self.config.get("adapter_lr", base_lr))
        light_transfer_lr = float(self.config.get("light_transfer_lr", base_lr))
        weight_decay = float(self.config.get("weight_decay", self.config.optimizer.adamw.get("weight_decay", 0.01)))

        model = unwrap_model(self.models["model"])
        adapter_ids: set[int] = set()
        adapter_params: list[nn.Parameter] = []
        for name in ["light_mlp", "dense_proj", "source_proj"]:
            module = getattr(model, name, None)
            if isinstance(module, nn.Module):
                for param in module.parameters():
                    if param.requires_grad:
                        adapter_params.append(param)
                        adapter_ids.add(id(param))
        if adapter_params:
            groups.append({"params": adapter_params, "lr": adapter_lr, "weight_decay": weight_decay})

        backbone_params = [param for param in model.parameters() if param.requires_grad and id(param) not in adapter_ids]
        if backbone_params and backbone_lr > 0:
            groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay})

        light_transfer_params = self._named_trainable_params(unwrap_model(self.models["light_transfer_transformer"]))
        if light_transfer_params:
            groups.append({"params": light_transfer_params, "lr": light_transfer_lr, "weight_decay": weight_decay})

        for name in ["light_encoder", "tokenizer"]:
            params = self._named_trainable_params(unwrap_model(self.models[name]))
            if params:
                groups.append({"params": params, "lr": base_lr, "weight_decay": weight_decay})
        return groups

    def _move_tensor_batch(self, batch: dict[str, object]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                output[key] = value.to(self.device, non_blocking=True)
            else:
                output[key] = value
        return output

    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        if bool(self.config.get("freeze_tokenizer", True)):
            with torch.no_grad():
                return self.tokenizer.encode(image).detach()
        return self.tokenizer.encode(image)

    def _run_light_encoder(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.stage != "ray_pretrain" and bool(self.config.get("freeze_ray_encoder", self.config.get("freeze_light_encoder", True))):
            with torch.no_grad():
                return self.light_encoder(image)
        return self.light_encoder(image)

    def _zero_like(self, reference: torch.Tensor) -> torch.Tensor:
        return reference.float().sum() * 0.0

    def _cosine_metric(self, pred_ray: torch.Tensor, target_ray: torch.Tensor) -> torch.Tensor:
        pred = F.normalize(pred_ray.float(), dim=1, eps=1e-6)
        target = F.normalize(target_ray.float(), dim=1, eps=1e-6)
        return torch.nan_to_num((pred * target).sum(dim=1).mean())

    def _diffusion_train_kwargs(self) -> dict[str, float]:
        return dict(OmegaConf.to_container(self.config.diffusion.train, resolve=True))

    def _ray_pretrain_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = self._move_tensor_batch(batch)  # type: ignore[assignment]
        source_image = batch["source_image"]
        source_light = batch["source_light"]
        source_ray = batch.get("source_ray")
        assert torch.is_tensor(source_image)
        assert torch.is_tensor(source_light)
        source_ray_tensor = source_ray if torch.is_tensor(source_ray) else light_to_ray(source_light)

        self.optimizer.zero_grad(set_to_none=True)
        pred = self.light_encoder(source_image)
        ray_loss = 1.0 - self._cosine_metric(pred["ray"], source_ray_tensor)
        temp_loss = F.l1_loss(pred["temperature"].float(), source_light[:, 2:3].float())
        total = torch.nan_to_num(ray_loss + 0.1 * temp_loss)
        total.backward()
        if self.clip_grad_norm > 0:
            params = [param for model in self.models.values() for param in model.parameters() if param.requires_grad]
            torch.nn.utils.clip_grad_norm_(params, self.clip_grad_norm)
        self.optimizer.step()
        return {
            "Train/total": total.detach(),
            "Train/ray_pretrain/loss": total.detach(),
            "Train/ray_pretrain/ray": torch.nan_to_num(ray_loss.detach()),
            "Train/ray_pretrain/temp": torch.nan_to_num(temp_loss.detach()),
            "Train/source_ray/cosine": self._cosine_metric(pred["ray"].detach(), source_ray_tensor),
        }

    def train_single_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.stage == "ray_pretrain":
            return self._ray_pretrain_loss(batch)

        batch = self._move_tensor_batch(batch)  # type: ignore[assignment]
        source_image = batch["source_image"]
        target_image = batch["target_image"]
        source_light = batch["source_light"]
        target_light = batch["target_light"]
        source_ray = batch.get("source_ray")
        target_ray = batch.get("target_ray")
        delta_angle = batch.get("delta_angle")
        mask = batch["mask"]
        y = batch["y"]
        depth = batch.get("depth")
        depth_valid = batch.get("depth_valid")

        assert torch.is_tensor(source_image)
        assert torch.is_tensor(target_image)
        assert torch.is_tensor(source_light)
        assert torch.is_tensor(target_light)
        assert torch.is_tensor(mask)
        assert torch.is_tensor(y)
        source_ray_tensor = source_ray if torch.is_tensor(source_ray) else light_to_ray(source_light)
        target_ray_tensor = target_ray if torch.is_tensor(target_ray) else light_to_ray(target_light)
        delta_angle_tensor = delta_angle if torch.is_tensor(delta_angle) else torch.zeros(source_light.shape[0], device=self.device)
        depth_tensor = depth if torch.is_tensor(depth) else None
        depth_valid_tensor = depth_valid if torch.is_tensor(depth_valid) else None

        self.optimizer.zero_grad(set_to_none=True)
        target_tokens = self._encode_image(target_image)
        source_tokens = self._encode_image(source_image)
        light_pred = self._run_light_encoder(source_image)
        source_ray_pred = light_pred["ray"]
        source_light_pred = light_pred["light"]

        use_gt_prob = float(self.config.get("use_gt_source_light_prob", 0.5))
        use_gt = (torch.rand(source_light.shape[0], 1, device=self.device) < use_gt_prob).to(dtype=source_light.dtype)
        source_ray_used = F.normalize(use_gt * source_ray_tensor + (1.0 - use_gt) * source_ray_pred, dim=1, eps=1e-6)
        source_temperature_used = use_gt * source_light[:, 2:3] + (1.0 - use_gt) * source_light_pred[:, 2:3]
        source_light_used = ray_to_light(source_ray_used, source_temperature_used)
        target_ray_rotated = rotate_ray_z(source_ray_used, delta_angle_tensor)
        target_light_rotated = ray_to_light(target_ray_rotated, target_light[:, 2:3])

        physics = self.physics_light_transfer(
            source_image,
            source_light_used,
            target_light_rotated,
            depth_tensor,
            source_ray=source_ray_used,
            target_ray=target_ray_rotated,
            depth_valid=depth_valid_tensor,
        )
        transfer = self.light_transfer_transformer(source_image, source_light_used, target_light_rotated, physics)
        dense_cond = transfer["dense_cond"].detach() if bool(self.config.get("detach_dense_cond_from_flow", False)) else transfer["dense_cond"]
        light_cond = torch.cat([source_light_used, target_light_rotated, target_light_rotated - source_light_used], dim=1)
        cond = {"y": y, "light_cond": light_cond, "dense_cond": dense_cond, "source_tokens": source_tokens}
        flow_loss_dict = self.diffusion(
            x=target_tokens,
            cond=cond,
            mask=mask,
            losses=self.losses,
            description="Train",
            return_outputs=True,
            **self._diffusion_train_kwargs(),
        )
        transfer_loss_dict = compute_dense_transfer_loss(
            transfer["delta_l"],
            source_image,
            target_image,
            q_clip=float(self.config.get("q_clip", 2.0)),
            beta=float(self.config.get("transfer_smooth_l1_beta", 0.1)),
        )
        flow_loss = flow_loss_dict["Train/loss"]
        transfer_loss = transfer_loss_dict["Train/light_transfer/loss"]
        total = float(self.config.lambda_flow) * flow_loss
        if float(self.config.lambda_transfer) > 0:
            total = total + float(self.config.lambda_transfer) * transfer_loss
        total = torch.nan_to_num(total)
        total.backward()
        if self.clip_grad_norm > 0:
            params = [param for model in self.models.values() for param in model.parameters() if param.requires_grad]
            torch.nn.utils.clip_grad_norm_(params, self.clip_grad_norm)
        self.optimizer.step()

        zero = self._zero_like(total.detach())
        all_losses: dict[str, torch.Tensor] = {
            "Train/total": total.detach(),
            "Train/flow/loss": flow_loss.detach(),
            "Train/transfer/loss": transfer_loss.detach(),
            "Train/light_transfer/loss": transfer_loss.detach(),
            "Train/light_transfer/q_abs": transfer_loss_dict["Train/light_transfer/q_abs"].detach(),
            "Train/light_transfer/delta_l_abs": transfer_loss_dict["Train/light_transfer/delta_l_abs"].detach(),
            "Train/source_ray/cosine": self._cosine_metric(source_ray_pred.detach(), source_ray_tensor),
            "Train/target_ray_rotated/cosine": self._cosine_metric(target_ray_rotated.detach(), target_ray_tensor),
            "Train/ray_rotation/loss": zero,
            "ray_encoder/loss": zero,
            "Train/tokenizer_recon/loss": zero,
            "Train/smooth/loss": zero,
            "Train/transfer_shadow/loss": zero,
            "Train/transfer_reflectance/loss": zero,
            "Train/transfer_physics/loss": zero,
            "Train/linear/loss": zero,
            "Train/endpoint_shadow/loss": zero,
            "Train/endpoint_transfer/loss": zero,
            "Train/endpoint_image/l1": zero,
        }
        all_losses.update({key: value.detach() for key, value in flow_loss_dict.items() if torch.is_tensor(value) and value.ndim == 0})
        return all_losses

    def save_checkpoint(self, epoch: int, name: str | None = None) -> None:
        super().save_checkpoint(epoch, name=name)
        if self.rank != 0 or self.stage != "ray_pretrain":
            return
        checkpoint = {
            "light_encoder": unwrap_model(self.models["light_encoder"]).state_dict(),
            "epoch": epoch,
            "config": OmegaConf.to_container(self.config, resolve=True),
        }
        torch.save(checkpoint, self.result_dir / f"ray_encoder_epoch_{epoch:04d}.pth")
        torch.save(checkpoint, self.result_dir / "ray_encoder_latest.pth")
        pointer = Path(str(self.config.get("result_root", "checkpoint"))) / "latest_RayEncoder.txt"
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text(str(self.result_dir), encoding="utf-8")

    def train_process(self) -> None:
        epochs = int(self.config.epochs)
        for epoch in range(self.start_epoch, epochs):
            for model in self.models.values():
                model.train()
            sampler = getattr(self.train_dataloader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            iterator: Iterable[dict[str, torch.Tensor]] = tqdm(self.train_dataloader, desc=f"Epoch {epoch}", disable=self.rank != 0)
            for batch in iterator:
                losses = self.train_single_step(batch)
                self.global_step += 1
                if self.rank == 0 and hasattr(iterator, "set_postfix"):
                    iterator.set_postfix({"loss": f"{float(losses['Train/total']):.4f}"})  # type: ignore[attr-defined]
                if self.global_step % int(self.config.log_every) == 0:
                    self.log_wandb(losses, step=self.global_step)
                if bool(self.config.get("debug_one_batch", False)):
                    self.log_wandb(losses, step=self.global_step)
                    break
            if (epoch + 1) % int(self.config.save_every) == 0:
                self.save_checkpoint(epoch)
            if bool(self.config.get("debug_one_batch", False)):
                break
