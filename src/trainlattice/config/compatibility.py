"""Resume-compatibility checks for resolved experiment configurations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from trainlattice.config.schema import ExperimentConfig


def recovery_config(config: ExperimentConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Return fields that must remain stable across a resumed run.

    Keys introduced after a checkpoint was saved compare as their default
    values — optimizer names, the loss section, and trainer additions —
    so checkpoints from earlier feature releases stay resumable.
    """
    data = deepcopy(config.to_dict() if isinstance(config, ExperimentConfig) else dict(config))
    experiment = data.get("experiment") or {}
    experiment.pop("runs_dir", None)
    trainer = data.get("trainer") or {}
    for key in ("device", "log_every", "epochs"):
        trainer.pop(key, None)
    trainer.setdefault("monitor", "val_loss")
    trainer.setdefault("max_grad_norm", None)
    optimizer = data.get("optimizer") or {}
    optimizer.setdefault("name", "adamw")
    if not data.get("loss"):
        data["loss"] = {"name": "cross_entropy"}
    return data


def configs_compatible(
    saved: ExperimentConfig | Mapping[str, Any], expected: ExperimentConfig | Mapping[str, Any]
) -> bool:
    """Compare recovery-sensitive configuration values one by one."""
    return recovery_config(saved) == recovery_config(expected)
