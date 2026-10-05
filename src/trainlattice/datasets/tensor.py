"""File-backed tensor dataset registration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, TensorDataset

from trainlattice.config.contracts import ComponentContract
from trainlattice.errors import ConfigError
from trainlattice.datasets.builtin import DATASETS


def _resolve_root(options: Mapping[str, Any]) -> Path:
    root = Path(str(options.get("root", ".")))
    if not root.is_dir():
        raise ConfigError(f"dataset.root directory not found: {root}")
    return root


def _load_split(root: Path, file_name: str, option: str) -> tuple[torch.Tensor, torch.Tensor]:
    path = root / file_name
    if not path.is_file():
        raise ConfigError(f"dataset.{option} file not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ConfigError(f"dataset.{option} file could not be read: {path}: {exc}") from exc
    if not isinstance(payload, dict) or "features" not in payload or "labels" not in payload:
        raise ConfigError(f"dataset.{option} file must hold 'features' and 'labels' tensors: {path}")
    features, labels = payload["features"], payload["labels"]
    if not isinstance(features, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise ConfigError(f"dataset.{option} file must hold 'features' and 'labels' tensors: {path}")
    if features.dim() < 1 or labels.dim() != 1 or features.shape[0] != labels.shape[0]:
        raise ConfigError(
            f"dataset.{option} tensors are inconsistent: features {tuple(features.shape)}, labels {tuple(labels.shape)}"
        )
    if labels.shape[0] < 1:
        raise ConfigError(f"dataset.{option} file must contain at least one sample: {path}")
    if labels.dtype != torch.long:
        raise ConfigError(f"dataset.{option} labels must be a Long (int64) tensor: {path}")
    if int(labels.min()) < 0:
        raise ConfigError(f"dataset.{option} labels must be non-negative: {path}")
    return features, labels


def build_tensor(
    *,
    train: bool,
    split: str | None = None,
    root: str | Path = ".",
    train_file: str = "train.pt",
    val_file: str = "validation.pt",
    num_classes: int = 10,
    **_: Any,
) -> Dataset[Any]:
    """Build a dataset from tensor files stored on disk."""
    root_path = Path(root)
    if not root_path.is_dir():
        raise ConfigError(f"dataset.root directory not found: {root_path}")
    split_name = split or ("train" if train else "validation")
    if split_name not in {"train", "validation"}:
        raise ConfigError(f"unsupported tensor dataset split: {split_name}")
    file_name = train_file if split_name == "train" else val_file
    option = "train_file" if split_name == "train" else "val_file"
    features, labels = _load_split(root_path, file_name, option)
    if labels.numel() and int(labels.max()) >= int(num_classes):
        raise ConfigError(f"dataset.num_classes must exceed the maximum label ({int(labels.max())})")
    return TensorDataset(features, labels)


def _validate_tensor(path: str, options: Mapping[str, Any]) -> None:
    """Validate the supported tensor dataset options."""
    allowed = {"root", "train_file", "val_file", "num_classes"}
    unknown = sorted(set(options) - allowed)
    if unknown:
        paths = ", ".join(f"{path}.{key}" for key in unknown)
        raise ConfigError(f"unknown configuration field(s): {paths}")
    root = options.get("root")
    if not isinstance(root, str) or not root.strip():
        raise ConfigError(f"{path}.root must be a non-empty string")
    defaults = {"train_file": "train.pt", "val_file": "validation.pt"}
    for key, default in defaults.items():
        value = options.get(key, default)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{path}.{key} must be a non-empty string")
    value = options.get("num_classes", 10)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConfigError(f"{path}.num_classes must be a positive integer")


def _tensor_contract(options: Mapping[str, Any]) -> ComponentContract:
    """State the on-disk feature shape and class count across both split files."""
    root = _resolve_root(options)
    train_file = str(options.get("train_file", "train.pt"))
    val_file = str(options.get("val_file", "validation.pt"))
    train_features, _train_labels = _load_split(root, train_file, "train_file")
    val_features, _val_labels = _load_split(root, val_file, "val_file")
    train_shape = tuple(int(size) for size in train_features.shape[1:])
    val_shape = tuple(int(size) for size in val_features.shape[1:])
    if train_shape != val_shape:
        raise ConfigError(
            f"dataset.val_file feature shape {list(val_shape)} must match dataset.train_file feature shape {list(train_shape)}"
        )
    return ComponentContract(
        input_dim=None,
        num_classes=int(options.get("num_classes", 10)),
        input_shape=train_shape,
    )


DATASETS.register(
    "tensor",
    build_tensor,
    validate=_validate_tensor,
    contract=_tensor_contract,
)
