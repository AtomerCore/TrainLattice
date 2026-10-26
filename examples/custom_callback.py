"""Custom callback registered into an experiment's lifecycle.

Callbacks observe the trainer through the ``Callback`` hooks and extend
the run between the built-in logging and checkpointing callbacks; any
hook may call ``trainer.request_stop(reason)`` to end the run
cooperatively — the current epoch completes, including its checkpoint,
and the run finishes with the reason recorded in ``state.stop_reason``.
"""

from __future__ import annotations

from time import perf_counter

from trainlattice import Callback
from trainlattice.trainer.trainer import Trainer


class EpochTimeCallback(Callback):
    """Print the wall-clock duration of every completed epoch."""

    def __init__(self) -> None:
        self._started: float | None = None

    def on_epoch_start(self, trainer: Trainer) -> None:
        self._started = perf_counter()

    def on_epoch_end(self, trainer: Trainer) -> None:
        if self._started is None:
            return
        elapsed = perf_counter() - self._started
        print(f"epoch {trainer.state.epoch} took {elapsed:.3f}s")
        self._started = None
