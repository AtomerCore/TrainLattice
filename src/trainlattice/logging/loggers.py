"""Built-in structured metric sinks."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, TextIO


class Logger(ABC):
    """Metric sink interface."""

    @abstractmethod
    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Record scalar metrics for one step."""

    def close(self) -> None:
        """Release resources held by the logger."""


class ConsoleLogger(Logger):
    """Compact human-readable terminal logger."""

    def log(self, metrics: dict[str, Any], step: int) -> None:
        values = " ".join(
            f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in metrics.items()
        )
        print(f"step={step} {values}")


class JSONLLogger(Logger):
    """Append-only JSON Lines metric logger."""

    def __init__(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._stream: TextIO = target.open("a", encoding="utf-8")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        self._stream.write(json.dumps({"step": step, **metrics}, sort_keys=True) + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


class CompositeLogger(Logger):
    """Fan out metric records to multiple loggers."""

    def __init__(self, loggers: Iterable[Logger]) -> None:
        self.loggers = list(loggers)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        for logger in self.loggers:
            logger.log(metrics, step)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()

