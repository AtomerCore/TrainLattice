"""Custom dataset registered through the TrainLattice registry.

Importing this module registers the ``two_blobs`` dataset: two Gaussian
blobs centered at plus and minus one, one blob per class, so a linear
model can separate them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import Dataset, TensorDataset

from trainlattice import DATASETS, ComponentContract, ConfigError


def build_two_blobs(
    *,
    train: bool,
    split: str | None = None,
    input_dim: int = 2,
    samples: int = 128,
    val_samples: int = 32,
    noise: float = 0.4,
    seed: int | None = None,
    **_: Any,
) -> Dataset[Any]:
    """Build one split of the two-blobs classification dataset.

    The runtime calls the constructor once per split with ``train`` and
    ``split`` selecting the portion, and injects ``seed`` derived from
    ``experiment.seed``; each split derives its own generator from it.
    Samples are ``(input, long label)`` pairs, as ``CrossEntropyLoss``
    requires.
    """
    split_name = split or ("train" if train else "validation")
    counts = {"train": samples, "validation": val_samples}
    if split_name not in counts:
        raise ConfigError(f"unsupported two_blobs dataset split: {split_name}")
    offset = {"train": 0, "validation": 1}[split_name]
    generator = torch.Generator().manual_seed((seed if seed is not None else 42) + offset)
    count = int(counts[split_name])
    center = torch.ones(int(input_dim))
    labels = torch.randint(0, 2, (count,), generator=generator)
    side = (labels * 2 - 1).unsqueeze(1)
    features = torch.randn(count, int(input_dim), generator=generator) * float(noise) + side * center
    return TensorDataset(features, labels.long())


def _validate_two_blobs(path: str, options: Mapping[str, Any]) -> None:
    """Reject unknown options and invalid sizes with dotted paths."""
    allowed = {"input_dim", "samples", "val_samples", "noise"}
    unknown = sorted(set(options) - allowed)
    if unknown:
        paths = ", ".join(f"{path}.{key}" for key in unknown)
        raise ConfigError(f"unknown configuration field(s): {paths}")
    for key, default in (("input_dim", 2), ("samples", 128), ("val_samples", 32)):
        value = options.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ConfigError(f"{path}.{key} must be a positive integer")
    spread = options.get("noise", 0.4)
    if not isinstance(spread, (int, float)) or isinstance(spread, bool) or spread <= 0:
        raise ConfigError(f"{path}.noise must be a positive number")


def _two_blobs_contract(options: Mapping[str, Any]) -> ComponentContract:
    """State the flattened input width and the fixed binary class count."""
    input_dim = options.get("input_dim", 2)
    well_formed = isinstance(input_dim, int) and not isinstance(input_dim, bool) and input_dim >= 1
    return ComponentContract(
        input_dim=input_dim if well_formed else None,
        num_classes=2,
        input_shape=(input_dim,) if well_formed else None,
    )


DATASETS.register(
    "two_blobs",
    build_two_blobs,
    validate=_validate_two_blobs,
    contract=_two_blobs_contract,
)
