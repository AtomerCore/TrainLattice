"""Training command requests without concrete implementation dependencies."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trainlattice.kernel.api import StateProvider
from trainlattice.kernel.snapshot import PluginSnapshot
from .state import TrainerState


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    """One command execution and its effective plugin graph."""

    config: dict[str, Any]
    command: str
    resume: Path | None
    snapshot: PluginSnapshot
    state_providers: Mapping[str, StateProvider]
    callbacks: Sequence[Any] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class RunResult:
    """A run's artifact directory and final training state."""

    run_dir: Path
    state: TrainerState
