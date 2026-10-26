"""Run-directory creation and manifest state."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from trainlattice.contracts.config import ExperimentConfig
from trainlattice.contracts.state import utc_now


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-.").lower()
    return slug or "experiment"


@dataclass(slots=True)
class RunContext:
    """Paths and manifest for one experiment run."""

    path: Path
    manifest: dict[str, Any]
    runtime: Any

    @classmethod
    def create(
        cls,
        config: ExperimentConfig,
        *,
        command: str,
        runtime: Any,
        native_config: dict[str, Any],
        snapshot: Any,
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
        (path / "config.yaml").write_text(yaml.safe_dump(native_config, sort_keys=False), encoding="utf-8")
        runtime.write_json(runtime.environment(), path / "environment.json")
        runtime.write_json(snapshot.to_dict(), path / "plugins.lock.json")
        manifest = {
            "format_version": 2,
            "experiment": config.experiment.name,
            "model": config.model.name,
            "dataset": config.dataset.name,
            "command": command,
            "git_commit": runtime.git_commit(),
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
        context = cls(path=path, manifest=manifest, runtime=runtime)
        context.write_manifest()
        return context

    def write_manifest(self) -> None:
        """Persist the current run status atomically."""
        self.runtime.write_json(self.manifest, self.path / "manifest.json")

    def update(self, **values: Any) -> None:
        """Apply and persist a manifest transition."""
        self.manifest.update(values)
        self.write_manifest()
