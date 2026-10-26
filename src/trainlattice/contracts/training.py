"""Structural service contracts shared by independent training plugins."""

from typing import Any, Protocol
from pathlib import Path

from .state import TrainerState


class TrainerView(Protocol):
    """The training state visible to lifecycle observers."""

    model: Any
    optimizer: Any
    scheduler: Any
    scaler: Any
    state: TrainerState
    batch_metrics: dict[str, float]

    def request_stop(self, reason: str) -> None:
        """Finish training after the current epoch's callbacks."""
        ...


class MetricLogger(Protocol):
    """A structured metric sink with an owned lifetime."""

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Record one metric event."""
        ...

    def close(self) -> None:
        """Release the sink's resources."""
        ...


class CheckpointStore(Protocol):
    """Persistence boundary consumed by callbacks and experiment assembly."""

    def save(self, name: str, **state: Any) -> Path:
        """Persist a named complete snapshot."""
        ...

    def preflight(self, path: str | Path, *, expected_config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Check recovery compatibility before constructing training objects."""
        ...

    def restore(self, payload: dict[str, Any], **objects: Any) -> TrainerState:
        """Restore validated state into bound training objects."""
        ...
