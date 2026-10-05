"""Optimizer registration, validation, and construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.optim import Optimizer

from trainlattice.config.schema import OptimizerConfig
from trainlattice.errors import ConfigError
from trainlattice.registry import Registry

OPTIMIZERS: Registry[Optimizer] = Registry("optimizer")


def _reject_unknown(path: str, options: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(options) - allowed)
    if unknown:
        keys = ", ".join(f"{path}.{key}" for key in unknown)
        raise ConfigError(f"unknown configuration field(s): {keys}")


def _non_negative_number(value: Any, path: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ConfigError(f"{path} must be a non-negative number")


def _positive_number(value: Any, path: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{path} must be a positive number")


def _fraction_pair(value: Any, path: str) -> None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
        or any(not isinstance(item, (int, float)) or isinstance(item, bool) or not 0.0 <= item < 1.0 for item in value)
    ):
        raise ConfigError(f"{path} must contain two numbers in [0, 1)")


def _validate_adam_family(path: str, options: Mapping[str, Any]) -> None:
    """Validate the Adam/AdamW options."""
    _reject_unknown(path, options, {"betas", "eps", "weight_decay", "amsgrad"})
    if "betas" in options:
        _fraction_pair(options["betas"], f"{path}.betas")
    if "eps" in options:
        _positive_number(options["eps"], f"{path}.eps")
    if "weight_decay" in options:
        _non_negative_number(options["weight_decay"], f"{path}.weight_decay")
    if "amsgrad" in options and not isinstance(options["amsgrad"], bool):
        raise ConfigError(f"{path}.amsgrad must be a boolean")


def _validate_sgd(path: str, options: Mapping[str, Any]) -> None:
    """Validate the SGD options."""
    _reject_unknown(path, options, {"momentum", "dampening", "weight_decay", "nesterov"})
    for key in ("momentum", "dampening", "weight_decay"):
        if key in options:
            _non_negative_number(options[key], f"{path}.{key}")
    if "nesterov" in options and not isinstance(options["nesterov"], bool):
        raise ConfigError(f"{path}.nesterov must be a boolean")


OPTIMIZERS.register("adamw", torch.optim.AdamW, validate=_validate_adam_family)
OPTIMIZERS.register("adam", torch.optim.Adam, validate=_validate_adam_family)
OPTIMIZERS.register("sgd", torch.optim.SGD, validate=_validate_sgd)


def validate_optimizer_config(config: OptimizerConfig) -> None:
    """Validate the options of the registered optimizer."""
    if config.name not in OPTIMIZERS:
        choices = ", ".join(OPTIMIZERS.names())
        raise ConfigError(f"unsupported optimizer.name {config.name!r}; available: {choices}")
    declaration = OPTIMIZERS.entry(config.name)
    if declaration.validate is not None:
        declaration.validate("optimizer", config.options)


def build_optimizer(config: OptimizerConfig, params: Any) -> Optimizer:
    """Construct a validated registered optimizer over the model parameters."""
    validate_optimizer_config(config)
    try:
        return OPTIMIZERS.build(config.name, params, lr=config.lr, **config.options)
    except ConfigError:
        raise
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ConfigError(f"invalid optimizer configuration: {exc}") from exc
