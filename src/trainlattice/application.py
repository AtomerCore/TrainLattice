"""Native Python bootstrap for manifest-defined experiment applications."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from trainlattice.contracts.application import ExecutionRequest
from trainlattice.kernel.bootstrap import BootstrapConfig, read_yaml
from trainlattice.kernel.lifecycle import Application, PluginManager
from trainlattice.kernel.sources import PluginSourceResolver


def build_application(config: Mapping[str, Any], *, base_dir: Path | None = None) -> Application:
    """Plan and activate the configured graph without starting a command."""
    bootstrap = BootstrapConfig.from_mapping(config)
    manager = PluginManager(PluginSourceResolver(base_dir))
    return manager.activate(manager.plan(bootstrap.plugins, bindings=bootstrap.bindings))


def run_experiment(config: Mapping[str, Any], *, base_dir: Path | None = None, command: str = "trainlattice train", resume: str | Path | None = None, callbacks: Sequence[Any] = ()) -> Any:
    """Execute the bound train command and dispose all application resources."""
    raw = deepcopy(dict(config))
    with build_application(raw, base_dir=base_dir) as app:
        request = ExecutionRequest(raw, command, Path(resume) if resume is not None else None, app.snapshot, app.state_providers, tuple(callbacks))
        with app.run_scope():
            return app.require("trainlattice.command/train")(request)


def run_from_file(path: str | Path, *, command: str = "trainlattice train", resume: str | Path | None = None, callbacks: Sequence[Any] = ()) -> Any:
    """Execute native YAML, resolving local plugin paths beside that YAML."""
    source = Path(path).resolve()
    return run_experiment(read_yaml(source), base_dir=source.parent, command=command, resume=resume, callbacks=callbacks)
