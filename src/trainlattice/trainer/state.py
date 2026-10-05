"""Serializable trainer runtime state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    """Return a stable UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class TrainerState:
    """A snapshot of the active training runtime."""

    epoch: int = -1
    next_epoch: int = 0
    global_step: int = 0
    latest_metrics: dict[str, float] = field(default_factory=dict)
    best_metric: float | None = None
    status: str = "initialized"
    started_at: str | None = None
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainerState":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in data.items() if key in allowed})
