"""Component factories and comparable model/data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from collections.abc import Callable, Mapping
from typing import Any

from trainlattice.errors import ConfigError


@dataclass(frozen=True, slots=True)
class ComponentFactory:
    """A component constructor and its preflight declarations."""

    name: str
    constructor: Callable[..., Any]
    validator: Callable[[str, Mapping[str, Any]], None] | None = None
    contract: Callable[[Mapping[str, Any]], Any] | None = None

    def validate(self, path: str, options: Mapping[str, Any]) -> None:
        """Validate supported options before component construction."""
        if self.validator is not None:
            self.validator(path, options)

    def build(self, *args: Any, **options: Any) -> Any:
        """Construct the selected implementation with explicit arguments."""
        try:
            return self.constructor(*args, **options)
        except ConfigError:
            raise
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ConfigError(f"invalid {self.name} component configuration: {exc}") from exc


@dataclass(frozen=True, slots=True)
class DatasetFactory(ComponentFactory):
    """Dataset construction plus its explicit data-loader assembly function."""

    loader_builder: Callable[..., Any] | None = None

    def build_loaders(self, config: Any, *, seed: int) -> Any:
        """Create deterministic train and validation data loaders."""
        if self.loader_builder is None:
            raise ConfigError(f"dataset factory {self.name} has no loader builder")
        return self.loader_builder(config, seed=seed, factory=self)


@dataclass(frozen=True, slots=True)
class Task:
    """A task's metric semantics shared by loss and trainer services."""

    kind: str


@dataclass(frozen=True, slots=True)
class LossFactory(ComponentFactory):
    """A loss factory declaring the task it optimizes."""

    task: Task = Task("classification")


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
