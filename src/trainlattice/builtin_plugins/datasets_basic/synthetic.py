"""Built-in classification datasets."""

from __future__ import annotations

from collections.abc import Mapping
from math import prod
from typing import Any

import torch
from torch.utils.data import Dataset, TensorDataset

from trainlattice.contracts.components import ComponentContract
from trainlattice.errors import ConfigError



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


def _validate_synthetic(path: str, options: Mapping[str, Any]) -> None:
    """Validate the supported synthetic dataset options."""
    allowed = {"samples", "val_samples", "input_shape", "num_classes"}
    unknown = sorted(set(options) - allowed)
    if unknown:
        paths = ", ".join(f"{path}.{key}" for key in unknown)
        raise ConfigError(f"unknown configuration field(s): {paths}")
    defaults = {"samples": 256, "val_samples": 64, "num_classes": 10}
    for key, default in defaults.items():
        value = options.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ConfigError(f"{path}.{key} must be a positive integer")
    shape = options.get("input_shape", [1, 28, 28])
    if not isinstance(shape, (list, tuple)) or not shape or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in shape
    ):
        raise ConfigError(f"{path}.input_shape must contain positive integers")


def _synthetic_contract(options: Mapping[str, Any]) -> ComponentContract:
    """State the synthetic input shape, flattened width, and class count."""
    shape = options.get("input_shape", [1, 28, 28])
    well_formed = isinstance(shape, (list, tuple)) and shape and all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 1 for value in shape
    )
    return ComponentContract(
        input_dim=prod(shape) if well_formed else None,
        num_classes=options.get("num_classes", 10),
        input_shape=tuple(shape) if well_formed else None,
    )
