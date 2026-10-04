"""Deterministic split, sampler, and data-loader construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from trainlattice.config.schema import DatasetConfig
from trainlattice.datasets.builtin import DATASETS, validate_dataset_config


@dataclass(slots=True)
class DataLoaders:
    """Training and validation data loaders."""

    train: DataLoader[Any]
    validation: DataLoader[Any] | None


class EpochShuffleSampler(Sampler[int]):
    """Produce a deterministic shuffle derived only from seed and epoch."""

    def __init__(self, data_source: Dataset[Any], *, seed: int) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic order for an epoch."""
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.data_source), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.data_source)


def build_dataloaders(config: DatasetConfig, *, seed: int) -> DataLoaders:
    """Build registered datasets and deterministic data loaders."""
    validate_dataset_config(config)
    options = dict(config.options)
    options.setdefault("seed", seed)
    train_dataset = DATASETS.build(config.name, train=True, split="train", **options)
    validation_dataset: Dataset[Any] | None = DATASETS.build(
        config.name, train=False, split="validation", **options
    )
    sampler = EpochShuffleSampler(train_dataset, seed=seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
    )
    validation_loader = (
        DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        if validation_dataset is not None
        else None
    )
    return DataLoaders(train=train_loader, validation=validation_loader)
