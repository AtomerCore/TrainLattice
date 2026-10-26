"""Loss registration, task declaration, and construction.

Every loss declares a task kind — ``classification`` or ``regression`` —
so the trainer knows which metrics apply: classification runs also report
accuracy, regression runs report loss only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from torch import nn

from trainlattice.errors import ConfigError







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
