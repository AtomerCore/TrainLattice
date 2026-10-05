"""Typed experiment configuration and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from re import fullmatch
from typing import Any, Mapping

from trainlattice.errors import ConfigError


def _section(data: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key} must be a mapping")
    return dict(value)


def _fixed_section(section_type: type[Any], data: Mapping[str, Any], path: str) -> Any:
    """Build a fixed-schema dataclass with dotted-path errors."""
    allowed = {item.name for item in fields(section_type)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        joined = ", ".join(f"{path}.{key}" for key in unknown)
        raise ConfigError(f"unknown configuration field(s): {joined}")
    try:
        return section_type(**dict(data))
    except TypeError as exc:
        raise ConfigError(f"invalid {path} configuration: {exc}") from exc


def _require_string(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{path} must be a string")
    return value.strip().lower()


@dataclass(slots=True)
class ExperimentSection:
    """Experiment identity and artifact location."""

    name: str = "experiment"
    seed: int = 42
    runs_dir: str = "runs"


@dataclass(slots=True)
class ModelConfig:
    """A model name and architecture-specific options."""

    name: str
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelConfig":
        values = dict(data)
        name = _require_string(values.pop("name", ""), "model.name")
        return cls(name=name, options=values)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, **self.options}


@dataclass(slots=True)
class DatasetConfig:
    """Dataset and data-loader settings."""

    name: str
    batch_size: int = 64
    num_workers: int = 0
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetConfig":
        values = dict(data)
        known = {key: values.pop(key) for key in list(values) if key in {"name", "batch_size", "num_workers"}}
        name = _require_string(known.pop("name", ""), "dataset.name")
        return cls(name=name, options=values, **known)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            **self.options,
        }


@dataclass(slots=True)
class OptimizerConfig:
    """Learning-rate settings for the fixed AdamW optimizer."""

    lr: float = 0.001


@dataclass(slots=True)
class TrainerConfig:
    """Training-loop controls."""

    epochs: int = 1
    device: str = "auto"
    validate_every: int = 1
    log_every: int = 10
    monitor: str = "val_loss"
    max_grad_norm: float | None = None


@dataclass(slots=True)
class ExperimentConfig:
    """Validated root configuration for one training experiment."""

    experiment: ExperimentSection
    model: ModelConfig
    dataset: DatasetConfig
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentConfig":
        if not isinstance(data, Mapping):
            raise ConfigError("configuration root must be a mapping")
        allowed_sections = {"experiment", "model", "dataset", "optimizer", "trainer"}
        unknown = sorted(set(data) - allowed_sections)
        if unknown:
            raise ConfigError(f"unknown configuration section(s): {', '.join(unknown)}")
        config = cls(
            experiment=_fixed_section(ExperimentSection, _section(data, "experiment"), "experiment"),
            model=ModelConfig.from_dict(_section(data, "model")),
            dataset=DatasetConfig.from_dict(_section(data, "dataset")),
            optimizer=_fixed_section(OptimizerConfig, _section(data, "optimizer"), "optimizer"),
            trainer=_fixed_section(TrainerConfig, _section(data, "trainer"), "trainer"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        errors: list[str] = []
        if not isinstance(self.experiment.name, str) or not self.experiment.name.strip():
            errors.append("experiment.name must not be empty")
        if not isinstance(self.experiment.seed, int) or isinstance(self.experiment.seed, bool):
            errors.append("experiment.seed must be an integer")
        if not isinstance(self.experiment.runs_dir, str) or not self.experiment.runs_dir.strip():
            errors.append("experiment.runs_dir must be a non-empty string")
        if not isinstance(self.model.name, str) or not self.model.name:
            errors.append("model.name is required")
        if not isinstance(self.dataset.name, str) or not self.dataset.name:
            errors.append("dataset.name is required")
        if (
            not isinstance(self.optimizer.lr, (int, float))
            or isinstance(self.optimizer.lr, bool)
            or not self.optimizer.lr > 0
        ):
            errors.append("optimizer.lr must be a positive number")
        if not isinstance(self.dataset.batch_size, int) or isinstance(self.dataset.batch_size, bool):
            errors.append("dataset.batch_size must be an integer")
        elif self.dataset.batch_size < 1:
            errors.append("dataset.batch_size must be at least 1")
        if not isinstance(self.dataset.num_workers, int) or isinstance(self.dataset.num_workers, bool):
            errors.append("dataset.num_workers must be an integer")
        elif self.dataset.num_workers < 0:
            errors.append("dataset.num_workers must be non-negative")
        if not isinstance(self.trainer.epochs, int) or isinstance(self.trainer.epochs, bool):
            errors.append("trainer.epochs must be an integer")
        elif self.trainer.epochs < 1:
            errors.append("trainer.epochs must be at least 1")
        if not isinstance(self.trainer.validate_every, int) or isinstance(self.trainer.validate_every, bool):
            errors.append("trainer.validate_every must be an integer")
        elif self.trainer.validate_every < 1:
            errors.append("trainer.validate_every must be at least 1")
        if not isinstance(self.trainer.log_every, int) or isinstance(self.trainer.log_every, bool):
            errors.append("trainer.log_every must be an integer")
        elif self.trainer.log_every < 1:
            errors.append("trainer.log_every must be at least 1")
        device = self.trainer.device.strip().lower() if isinstance(self.trainer.device, str) else ""
        if not fullmatch(r"(auto|cpu|cuda|cuda:\d+)", device):
            errors.append("trainer.device must be one of: auto, cpu, cuda, cuda:N")
        if (
            not isinstance(self.trainer.monitor, str)
            or self.trainer.monitor not in {"val_loss", "val_accuracy"}
        ):
            errors.append("trainer.monitor must be one of: val_loss, val_accuracy")
        max_grad_norm = self.trainer.max_grad_norm
        if isinstance(max_grad_norm, bool) or (
            max_grad_norm is not None
            and (not isinstance(max_grad_norm, (int, float)) or not max_grad_norm > 0)
        ):
            errors.append("trainer.max_grad_norm must be a positive number")
        if errors:
            raise ConfigError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": asdict(self.experiment),
            "model": self.model.to_dict(),
            "dataset": self.dataset.to_dict(),
            "optimizer": asdict(self.optimizer),
            "trainer": asdict(self.trainer),
        }
