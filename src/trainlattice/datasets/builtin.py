"""Built-in classification datasets."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset, TensorDataset

from trainlattice.config.schema import DatasetConfig
from trainlattice.errors import ConfigError
from trainlattice.registry import Registry

DATASETS: Registry[Dataset[Any]] = Registry("dataset")


@DATASETS.register("synthetic")
def build_synthetic(
    *,
    train: bool,
    split: str | None = None,
    samples: int = 256,
    val_samples: int = 64,
    input_shape: list[int] | tuple[int, ...] = (1, 28, 28),
    num_classes: int = 10,
    seed: int = 42,
    **_: Any,
) -> Dataset[Any]:
    """Build deterministic random data for examples and smoke tests."""
    split_name = split or ("train" if train else "validation")
    counts = {"train": samples, "validation": val_samples}
    if split_name not in counts:
        raise ConfigError(f"unsupported synthetic dataset split: {split_name}")
    count = int(counts[split_name])
    generator = torch.Generator().manual_seed(int(seed) + {"train": 0, "validation": 1}[split_name])
    features = torch.randn(count, *input_shape, generator=generator)
    labels = torch.randint(0, int(num_classes), (count,), generator=generator)
    return TensorDataset(features, labels)


def validate_dataset_config(config: DatasetConfig) -> None:
    """Validate built-in dataset options before constructing datasets."""
    if config.name not in DATASETS:
        choices = ", ".join(DATASETS.names())
        raise ConfigError(f"unsupported dataset.name {config.name!r}; available: {choices}")
    if config.name != "synthetic":
        return
    allowed = {"samples", "val_samples", "input_shape", "num_classes"}
    unknown = sorted(set(config.options) - allowed)
    if unknown:
        paths = ", ".join(f"dataset.{key}" for key in unknown)
        raise ConfigError(f"unknown configuration field(s): {paths}")
    defaults = {"samples": 256, "val_samples": 64, "num_classes": 10}
    for key, default in defaults.items():
        value = config.options.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ConfigError(f"dataset.{key} must be a positive integer")
    shape = config.options.get("input_shape", [1, 28, 28])
    if not isinstance(shape, (list, tuple)) or not shape or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in shape
    ):
        raise ConfigError("dataset.input_shape must contain positive integers")


