"""Resume-compatibility checks for resolved experiment configurations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from trainlattice.config.schema import ExperimentConfig


def recovery_config(config: ExperimentConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Return fields that must remain stable across a resumed run."""
    data = deepcopy(config.to_dict() if isinstance(config, ExperimentConfig) else dict(config))
    experiment = data.get("experiment") or {}
    experiment.pop("runs_dir", None)
    trainer = data.get("trainer") or {}
    for key in ("device", "log_every", "epochs"):
        trainer.pop(key, None)
    return data


def configs_compatible(
    saved: ExperimentConfig | Mapping[str, Any], expected: ExperimentConfig | Mapping[str, Any]
) -> bool:
    """Compare recovery-sensitive configuration values one by one."""
    return recovery_config(saved) == recovery_config(expected)
