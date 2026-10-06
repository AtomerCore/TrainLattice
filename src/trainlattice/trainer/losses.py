"""Loss registration, task declaration, and construction.

Every loss declares a task kind — ``classification`` or ``regression`` —
so the trainer knows which metrics apply: classification runs also report
accuracy, regression runs report loss only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from torch import nn

from trainlattice.config.schema import LossConfig
from trainlattice.errors import ConfigError, RegistryError
from trainlattice.registry import Registry

LOSSES: Registry[nn.Module] = Registry("loss")

TASKS = ("classification", "regression")

LOSS_TASKS: dict[str, str] = {}


def register_loss(
    name: str,
    target: Callable[..., nn.Module] | None = None,
    *,
    task: str,
    validate: Callable[[str, Mapping[str, Any]], None] | None = None,
):
    """Register a loss constructor and declare its task kind.

    Mirrors ``Registry.register`` and additionally records the task kind
    that decides whether the trainer computes accuracy alongside loss.
    """
    if task not in TASKS:
        raise RegistryError(f"loss task must be one of: {', '.join(TASKS)}")

    def decorator(value: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        LOSSES.register(name, value, validate=validate)
        LOSS_TASKS[Registry._normalize(name)] = task
        return value

    return decorator(target) if target is not None else decorator


def _validate_reduction(path: str, options: Mapping[str, Any]) -> None:
    reduction = options.get("reduction", "mean")
    if reduction not in ("mean", "sum", "none"):
        raise ConfigError(f"{path}.reduction must be one of: mean, sum, none")


def _validate_cross_entropy(path: str, options: Mapping[str, Any]) -> None:
    """Validate the supported cross-entropy options."""
    unknown = sorted(set(options) - {"label_smoothing", "ignore_index", "reduction"})
    if unknown:
        keys = ", ".join(f"{path}.{key}" for key in unknown)
        raise ConfigError(f"unknown configuration field(s): {keys}")
    _validate_reduction(path, options)
    smoothing = options.get("label_smoothing", 0.0)
    if not isinstance(smoothing, (int, float)) or isinstance(smoothing, bool) or not 0.0 <= smoothing <= 1.0:
        raise ConfigError(f"{path}.label_smoothing must be a number in [0, 1]")
    if "ignore_index" in options and not isinstance(options["ignore_index"], int):
        raise ConfigError(f"{path}.ignore_index must be an integer")


def _validate_distance_loss(path: str, options: Mapping[str, Any]) -> None:
    """Validate the supported MSE/L1 options."""
    unknown = sorted(set(options) - {"reduction"})
    if unknown:
        keys = ", ".join(f"{path}.{key}" for key in unknown)
        raise ConfigError(f"unknown configuration field(s): {keys}")
    _validate_reduction(path, options)


register_loss("cross_entropy", nn.CrossEntropyLoss, task="classification", validate=_validate_cross_entropy)
register_loss("mse", nn.MSELoss, task="regression", validate=_validate_distance_loss)
register_loss("l1", nn.L1Loss, task="regression", validate=_validate_distance_loss)


def loss_task(name: str) -> str:
    """Return the declared task kind of a registered loss."""
    key = Registry._normalize(name)
    try:
        return LOSS_TASKS[key]
    except KeyError as exc:
        if key in LOSSES:
            raise RegistryError(
                f"loss {name!r} was registered without a task kind; use register_loss to declare one"
            ) from exc
        choices = ", ".join(LOSSES.names()) or "none"
        raise RegistryError(f"unknown loss {name!r}; available: {choices}") from exc


def validate_loss_config(config: LossConfig) -> None:
    """Validate the options of the registered loss."""
    if config.name not in LOSSES:
        choices = ", ".join(LOSSES.names())
        raise ConfigError(f"unsupported loss.name {config.name!r}; available: {choices}")
    declaration = LOSSES.entry(config.name)
    if declaration.validate is not None:
        declaration.validate("loss", config.options)


def build_loss(config: LossConfig) -> nn.Module:
    """Construct a validated registered loss."""
    validate_loss_config(config)
    try:
        return LOSSES.build(config.name, **config.options)
    except ConfigError:
        raise
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ConfigError(f"invalid loss configuration: {exc}") from exc
