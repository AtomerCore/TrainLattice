"""Logging and checkpointing callbacks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from trainlattice.contracts.callbacks import Callback
from trainlattice.contracts.training import CheckpointStore as CheckpointManager
from trainlattice.contracts.training import MetricLogger as Logger

if TYPE_CHECKING:
    from trainlattice.contracts.training import TrainerView as Trainer


class LoggingCallback(Callback):
    """Send batch and epoch metrics to a structured logger."""

    def __init__(self, logger: Logger, *, log_every: int = 10) -> None:
        self.logger = logger
        self.log_every = log_every
        self._last_batch_step = 0

    def on_train_start(self, trainer: "Trainer") -> None:
        # steps restored by a resume have already been logged by their own run
        self._last_batch_step = trainer.state.global_step

    def on_batch_end(self, trainer: "Trainer") -> None:
        # the hook fires per micro-batch; a record is emitted once per optimizer step
        step = trainer.state.global_step
        if step > self._last_batch_step and step % self.log_every == 0:
            self._last_batch_step = step
            self.logger.log({"phase": "train", "epoch": trainer.state.epoch, **trainer.batch_metrics}, step)

    def on_epoch_end(self, trainer: "Trainer") -> None:
        self.logger.log({"phase": "epoch", "epoch": trainer.state.epoch, **trainer.state.latest_metrics}, trainer.state.global_step)

    def on_train_end(self, trainer: "Trainer") -> None:
        self.logger.close()

    def on_exception(self, trainer: "Trainer", exc: BaseException) -> None:
        self.logger.close()


class CheckpointCallback(Callback):
    """Persist the latest, best, and emergency checkpoints."""

    def __init__(
        self,
        manager: CheckpointManager,
        config: dict,
        *,
        monitor: str = "val_loss",
        scheduler_budget: int | None = None,
    ) -> None:
        self.manager = manager
        self.config = config
        self.monitor = monitor
        self.scheduler_budget = scheduler_budget

    def _save(self, trainer: "Trainer", name: str, *, resume_safe: bool = True):
        return self.manager.save(
            name,
            model=trainer.model,
            optimizer=trainer.optimizer,
            trainer_state=trainer.state,
            config=self.config,
            resume_safe=resume_safe,
            scheduler_state=trainer.scheduler.state_dict() if trainer.scheduler is not None else None,
            scaler_state=trainer.scaler.state_dict() if trainer.scaler is not None else None,
            scheduler_budget=self.scheduler_budget,
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
