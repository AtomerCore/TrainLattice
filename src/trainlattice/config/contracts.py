"""Comparable I/O contracts for registered components."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod


@dataclass(frozen=True, slots=True)
class ComponentContract:
    """The interface one component offers its counterpart.

    ``input_dim`` is the flattened input width and ``input_shape`` the structured
    input shape; each is ``None`` when a side cannot state one, in which case that
    dimension is not compared on its own track.
    """

    input_dim: int | None
    num_classes: int
    input_shape: tuple[int, ...] | None = None


def compare_contracts(model: ComponentContract, dataset: ComponentContract) -> list[str]:
    """Describe every mismatch between a model contract and a dataset contract."""
    errors: list[str] = []
    if (
        model.input_shape is not None
        and dataset.input_shape is not None
        and tuple(model.input_shape) != tuple(dataset.input_shape)
    ):
        errors.append(
            f"model.input_shape ({list(model.input_shape)}) must match dataset.input_shape ({list(dataset.input_shape)})"
        )
    if (
        model.input_dim is not None
        and dataset.input_dim is not None
        and model.input_dim != dataset.input_dim
    ):
        errors.append(
            f"model.input_dim ({model.input_dim}) must match dataset.input_shape product ({dataset.input_dim})"
        )
    if (
        model.input_dim is not None
        and model.input_shape is None
        and dataset.input_dim is None
        and dataset.input_shape is not None
        and model.input_dim != prod(dataset.input_shape)
    ):
        errors.append(
            f"model.input_dim ({model.input_dim}) must match dataset.input_shape product ({prod(dataset.input_shape)})"
        )
    if model.num_classes != dataset.num_classes:
        errors.append(
            f"model.num_classes ({model.num_classes}) must match dataset.num_classes ({dataset.num_classes})"
        )
    return errors
