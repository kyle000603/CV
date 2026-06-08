from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler


def _collate_optional(batch: list[Any]) -> Any:
    if len(batch) == 0:
        return batch
    first = batch[0]
    if isinstance(first, dict):
        return {key: _collate_optional([item.get(key) for item in batch]) for key in first}
    if all(item is None for item in batch):
        return None
    return default_collate(batch)


class DataModule:
    def __init__(
        self,
        global_batch_size: int,
        num_workers: int,
        pin_memory: bool,
        global_seed: int,
        drop_last: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
    ) -> None:
        self.global_batch_size = global_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.global_seed = global_seed
        self.drop_last = drop_last
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.seed_everything(global_seed)

    @staticmethod
    def seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _world_size(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    def _local_batch_size(self) -> int:
        world_size = self._world_size()
        if self.global_batch_size % world_size != 0:
            raise ValueError(
                f"global_batch_size={self.global_batch_size} must be divisible by world_size={world_size}."
            )
        return self.global_batch_size // world_size

    def get_train_dataloader(self, dataset: Dataset[Any]) -> DataLoader[Any]:
        return self.get_dataloader(dataset, shuffle=True)

    def get_val_dataloader(self, dataset: Dataset[Any]) -> DataLoader[Any]:
        return self.get_dataloader(dataset, shuffle=False)

    def get_dataloader(self, dataset: Dataset[Any], shuffle: bool = True) -> DataLoader[Any]:
        sampler = None
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(dataset, shuffle=shuffle, seed=self.global_seed)
            shuffle = False

        kwargs: dict[str, Any] = {}
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
            kwargs["persistent_workers"] = self.persistent_workers

        return DataLoader(
            dataset,
            batch_size=self._local_batch_size(),
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last if shuffle else False,
            collate_fn=_collate_optional,
            **kwargs,
        )

