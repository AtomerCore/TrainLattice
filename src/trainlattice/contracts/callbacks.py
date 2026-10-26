"""Callback lifecycle interface."""

from __future__ import annotations

from trainlattice.contracts.training import TrainerView as Trainer


class Callback:
    """No-op lifecycle hooks for training extensions."""

    def on_train_start(self, trainer: "Trainer") -> None:
        pass

    def on_epoch_start(self, trainer: "Trainer") -> None:
        pass

    def on_batch_start(self, trainer: "Trainer") -> None:
        pass

    def on_batch_end(self, trainer: "Trainer") -> None:
        pass

    def on_epoch_end(self, trainer: "Trainer") -> None:
        pass

    def on_train_end(self, trainer: "Trainer") -> None:
        pass

    def on_exception(self, trainer: "Trainer", exc: BaseException) -> None:
        pass
