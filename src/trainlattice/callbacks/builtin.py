"""Logging and checkpointing callbacks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from trainlattice.callbacks.base import Callback
from trainlattice.checkpoint import CheckpointManager
from trainlattice.logging import Logger

if TYPE_CHECKING:
    from trainlattice.trainer.trainer import Trainer


class LoggingCallback(Callback):
    """Send batch and epoch metrics to a structured logger."""

    def __init__(self, logger: Logger, *, log_every: int = 10) -> None:
        self.logger = logger
        self.log_every = log_every

    def on_batch_end(self, trainer: "Trainer") -> None:
        if trainer.state.global_step % self.log_every == 0:
            self.logger.log({"phase": "train", "epoch": trainer.state.epoch, **trainer.batch_metrics}, trainer.state.global_step)

    def on_epoch_end(self, trainer: "Trainer") -> None:
        self.logger.log({"phase": "epoch", "epoch": trainer.state.epoch, **trainer.state.latest_metrics}, trainer.state.global_step)

    def on_train_end(self, trainer: "Trainer") -> None:
        self.logger.close()

    def on_exception(self, trainer: "Trainer", exc: BaseException) -> None:
        self.logger.close()


class CheckpointCallback(Callback):
    """Persist the latest, best, and emergency checkpoints."""

    def __init__(self, manager: CheckpointManager, config: dict, *, monitor: str = "val_loss") -> None:
        self.manager = manager
        self.config = config
        self.monitor = monitor

    def _save(self, trainer: "Trainer", name: str, *, resume_safe: bool = True):
        return self.manager.save(
            name,
            model=trainer.model,
            optimizer=trainer.optimizer,
            trainer_state=trainer.state,
            config=self.config,
            resume_safe=resume_safe,
        )

    def _improved(self, current: float, best: float | None) -> bool:
        if best is None or not math.isfinite(best):
            return True
        return current < best if self.monitor == "val_loss" else current > best

    def on_epoch_end(self, trainer: "Trainer") -> None:
        """Update the best metric before saving, so last.pt carries the fresh value."""
        metric = trainer.state.latest_metrics.get(self.monitor)
        improved = (
            metric is not None
            and math.isfinite(metric)
            and self._improved(metric, trainer.state.best_metric)
        )
        if improved:
            trainer.state.best_metric = metric
        self._save(trainer, "last.pt")
        if improved:
            self._save(trainer, "best.pt")

    def on_exception(self, trainer: "Trainer", exc: BaseException) -> None:
        self._save(trainer, "emergency.pt", resume_safe=False)
