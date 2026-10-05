"""Run-directory creation and manifest state."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from trainlattice.config import ExperimentConfig, save_config
from trainlattice.runtime.environment import collect_environment, git_commit, write_json
from trainlattice.trainer.state import utc_now


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-.").lower()
    return slug or "experiment"


@dataclass(slots=True)
class RunContext:
    """Paths and manifest for one experiment run."""

    path: Path
    manifest: dict[str, Any]

    @classmethod
    def create(
        cls,
        config: ExperimentConfig,
        *,
        command: str,
        model_source: str | None = None,
        dataset_source: str | None = None,
        optimizer_source: str | None = None,
        loss_source: str | None = None,
    ) -> "RunContext":
        runs_dir = Path(config.experiment.runs_dir)
        runs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = runs_dir / f"{stamp}_{_slug(config.experiment.name)}"
        path = base
        suffix = 1
        while path.exists():
            path = Path(f"{base}_{suffix:02d}")
            suffix += 1
        path.mkdir(parents=True)
        (path / "checkpoints").mkdir()
        save_config(config, path / "config.yaml")
        write_json(collect_environment(), path / "environment.json")
        manifest = {
            "format_version": 1,
            "experiment": config.experiment.name,
            "model": config.model.name,
            "dataset": config.dataset.name,
            "command": command,
            "git_commit": git_commit(),
            "seed": config.experiment.seed,
            "created_at": utc_now(),
            "started_at": None,
            "finished_at": None,
            "status": "created",
            "resume_source": None,
            "epoch": -1,
            "global_step": 0,
        }
        if model_source is not None:
            manifest["model_source"] = model_source
        if dataset_source is not None:
            manifest["dataset_source"] = dataset_source
        if optimizer_source is not None:
            manifest["optimizer_source"] = optimizer_source
        if loss_source is not None:
            manifest["loss_source"] = loss_source
        context = cls(path=path, manifest=manifest)
        context.write_manifest()
        return context

    def write_manifest(self) -> None:
        write_json(self.manifest, self.path / "manifest.json")

    def update(self, **values: Any) -> None:
        self.manifest.update(values)
        self.write_manifest()
