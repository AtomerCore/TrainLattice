"""Custom learning-rate scheduler registered through the TrainLattice registry.

Importing this module registers the ``exponential`` scheduler, making the name
usable in the ``scheduler`` section of any experiment configuration. The
constructor receives the optimizer positionally, then the resolved step budget
(which a per-step-decay schedule ignores), then the YAML options as keywords.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR

from trainlattice import SCHEDULERS, ConfigError


def build_exponential(optimizer: Optimizer, _total_steps: int | None, *, gamma: float = 0.9) -> ExponentialLR:
    """Build the multiplicative decay schedule; the step budget is unused."""

    return ExponentialLR(optimizer, gamma=gamma)


def _validate_exponential(path: str, options: Mapping[str, Any]) -> None:
    """Reject unknown options and a non-positive decay factor with dotted paths."""
    unknown = sorted(set(options) - {"gamma"})
    if unknown:
        paths = ", ".join(f"{path}.{key}" for key in unknown)
        raise ConfigError(f"unknown configuration field(s): {paths}")
    gamma = options.get("gamma", 0.9)
    if not isinstance(gamma, (int, float)) or isinstance(gamma, bool) or gamma <= 0:
        raise ConfigError(f"{path}.gamma must be a positive number")


SCHEDULERS.register("exponential", build_exponential, validate=_validate_exponential)
