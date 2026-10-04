"""Logging and checkpointing callbacks."""

from __future__ import annotations

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
    """Persist the latest and emergency checkpoints."""

    def __init__(self, manager: CheckpointManager, config: dict) -> None:
        self.manager = manager
        self.config = config

    def _save(self, trainer: "Trainer", name: str, *, resume_safe: bool = True):
        return self.manager.save(
            name,
            model=trainer.model,
            optimizer=trainer.optimizer,
            trainer_state=trainer.state,
            config=self.config,
            resume_safe=resume_safe,
        )

    def on_epoch_end(self, trainer: "Trainer") -> None:
        self._save(trainer, "last.pt")

    def on_exception(self, trainer: "Trainer", exc: BaseException) -> None:
        self._save(trainer, "emergency.pt", resume_safe=False)
