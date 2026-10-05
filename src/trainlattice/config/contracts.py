"""Comparable I/O contracts for registered components."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ComponentContract:
    """The interface one component offers its counterpart.

    ``input_dim`` is the flattened input width; it is ``None`` when a side
    cannot state one, in which case the dimension is not compared.
    """

    input_dim: int | None
    num_classes: int


def compare_contracts(model: ComponentContract, dataset: ComponentContract) -> list[str]:
    """Describe every mismatch between a model contract and a dataset contract."""
    errors: list[str] = []
    if (
        model.input_dim is not None
        and dataset.input_dim is not None
        and model.input_dim != dataset.input_dim
    ):
        errors.append(
            f"model.input_dim ({model.input_dim}) must match dataset.input_shape product ({dataset.input_dim})"
        )
    if model.num_classes != dataset.num_classes:
        errors.append(
            f"model.num_classes ({model.num_classes}) must match dataset.num_classes ({dataset.num_classes})"
        )
    return errors
