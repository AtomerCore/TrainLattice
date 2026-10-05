"""Architecture registration, construction, and validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from trainlattice.config.contracts import ComponentContract
from trainlattice.config.schema import ModelConfig
from trainlattice.errors import ConfigError
from trainlattice.modeling.architectures import MLP
from trainlattice.registry import Registry

ARCHITECTURES: Registry[nn.Module] = Registry("architecture")


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


ARCHITECTURES.register("mlp", MLP, validate=_validate_mlp, contract=_mlp_contract)


def validate_model_config(config: ModelConfig) -> None:
    """Validate the options of the registered architecture."""
    if config.name not in ARCHITECTURES:
        raise ConfigError(
            f"unsupported model.name {config.name!r}; available: {', '.join(ARCHITECTURES.names())}"
        )
    declaration = ARCHITECTURES.entry(config.name)
    if declaration.validate is not None:
        declaration.validate("model", config.options)


def build_model(config: ModelConfig) -> nn.Module:
    """Construct a validated registered architecture."""
    validate_model_config(config)
    try:
        return ARCHITECTURES.build(config.name, **config.options)
    except ConfigError:
        raise
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ConfigError(f"invalid model configuration: {exc}") from exc
