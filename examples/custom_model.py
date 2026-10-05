"""Custom architecture registered through the TrainLattice registry.

Importing this module registers the ``logistic`` architecture, making the
name usable in the ``model`` section of any experiment configuration. The
constructor takes flat keyword options mirroring the YAML ``model`` keys.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from trainlattice import ARCHITECTURES, ComponentContract, ConfigError


class Logistic(nn.Module):
    """Single linear layer for linearly separable classification."""

    def __init__(self, *, input_dim: int = 2, num_classes: int = 2) -> None:
        super().__init__()
        self.network = nn.Linear(input_dim, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


def _validate_logistic(path: str, options: Mapping[str, Any]) -> None:
    """Reject unknown options and non-positive dimensions with dotted paths."""
    unknown = sorted(set(options) - {"input_dim", "num_classes"})
    if unknown:
        paths = ", ".join(f"{path}.{key}" for key in unknown)
        raise ConfigError(f"unknown configuration field(s): {paths}")
    for key, default in (("input_dim", 2), ("num_classes", 2)):
        value = options.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ConfigError(f"{path}.{key} must be a positive integer")


def _logistic_contract(options: Mapping[str, Any]) -> ComponentContract:
    """State the flattened input width and class count for preflight checks."""
    return ComponentContract(
        input_dim=options.get("input_dim", 2),
        num_classes=options.get("num_classes", 2),
    )


ARCHITECTURES.register(
    "logistic",
    Logistic,
    validate=_validate_logistic,
    contract=_logistic_contract,
)
