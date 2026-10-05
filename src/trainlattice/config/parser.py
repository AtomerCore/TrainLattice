"""YAML configuration input and output."""

from __future__ import annotations

from pathlib import Path

import yaml

from trainlattice.config.schema import ExperimentConfig
from trainlattice.errors import ConfigError


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""
    source = Path(path)
    if source.is_dir():
        raise ConfigError(f"configuration path is a directory, not a file: {source}")
    if not source.is_file():
        raise ConfigError(f"configuration file not found: {source}")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {source}: {exc}") from exc
    if raw is None:
        raise ConfigError(f"configuration file is empty: {source}")
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a mapping")
    return ExperimentConfig.from_dict(raw)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Write a resolved configuration snapshot."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")

