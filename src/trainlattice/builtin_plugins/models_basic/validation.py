"""Architecture registration, construction, and validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from trainlattice.contracts.components import ComponentContract
from trainlattice.errors import ConfigError



def _positive_integer(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConfigError(f"{path} must be a positive integer")
    return value


def _positive_sequence(value: Any, path: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{path} must be a sequence")
    return tuple(_positive_integer(item, f"{path}[{index}]") for index, item in enumerate(value))


def _validate_mlp(path: str, options: Mapping[str, Any]) -> None:
    """Validate the supported MLP architecture options."""
    unknown = sorted(set(options) - {"input_dim", "hidden_dims", "num_classes"})
    if unknown:
        paths = ", ".join(f"{path}.{key}" for key in unknown)
        raise ConfigError(f"unknown configuration field(s): {paths}")
    _positive_integer(options.get("input_dim", 784), f"{path}.input_dim")
    _positive_sequence(options.get("hidden_dims", (512, 256, 128)), f"{path}.hidden_dims")
    _positive_integer(options.get("num_classes", 10), f"{path}.num_classes")


def _mlp_contract(options: Mapping[str, Any]) -> ComponentContract:
    """State the MLP flattened input width and class count."""
    return ComponentContract(
        input_dim=options.get("input_dim", 784),
        num_classes=options.get("num_classes", 10),
    )


def _validate_cnn(path: str, options: Mapping[str, Any]) -> None:
    """Validate the supported CNN architecture options."""
    unknown = sorted(set(options) - {"input_shape", "channels", "num_classes"})
    if unknown:
        paths = ", ".join(f"{path}.{key}" for key in unknown)
        raise ConfigError(f"unknown configuration field(s): {paths}")
    shape = options.get("input_shape", (1, 28, 28))
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 3
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in shape)
    ):
        raise ConfigError(f"{path}.input_shape must contain three positive integers")
    channels = _positive_sequence(options.get("channels", (32, 64)), f"{path}.channels")
    _positive_integer(options.get("num_classes", 10), f"{path}.num_classes")
    height, width = shape[1], shape[2]
    factor = 2 ** len(channels)
    if height % factor != 0 or width % factor != 0:
        raise ConfigError(
            f"{path}.input_shape spatial dimensions ({height}, {width}) must stay divisible by the pooling factor {factor}"
        )


def _cnn_contract(options: Mapping[str, Any]) -> ComponentContract:
    """State the CNN structured input shape and class count."""
    return ComponentContract(
        input_dim=None,
        num_classes=options.get("num_classes", 10),
        input_shape=tuple(options.get("input_shape", (1, 28, 28))),
    )
