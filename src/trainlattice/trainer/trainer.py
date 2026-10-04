"""Transparent single-process classification training lifecycle."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from trainlattice.trainer.metrics import ClassificationTotals
from trainlattice.trainer.state import TrainerState, utc_now

if TYPE_CHECKING:
    from trainlattice.callbacks.base import Callback

logger = logging.getLogger(__name__)


class Trainer:
    """Run training and validation while exposing lifecycle events."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        epochs: int,
        callbacks: Iterable[Callback] = (),
        validate_every: int = 1,
        state: TrainerState | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.epochs = epochs
        self.callbacks = list(callbacks)
        self.validate_every = validate_every
        self.state = state or TrainerState()
        self.batch_metrics: dict[str, float] = {}

    def _emit(self, event: str, *args: Any) -> None:
        for callback in self.callbacks:
            getattr(callback, event)(self, *args)

    def _emit_exception(self, exc: BaseException) -> None:
        for callback in self.callbacks:
            try:
                callback.on_exception(self, exc)
            except Exception as callback_exc:
                logger.warning(
                    "callback %s failed while handling %s: %s",
                    type(callback).__name__,
                    type(exc).__name__,
                    callback_exc,
                )

    @staticmethod
    def _unpack_batch(batch: Any) -> tuple[Tensor, Tensor]:
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise TypeError("training batches must contain inputs and targets")
        return batch[0], batch[1]

    def fit(self, train_loader: DataLoader[Any], validation_loader: DataLoader[Any] | None = None) -> TrainerState:
        """Execute the configured lifecycle from the state's next epoch."""
        try:
            self.model.to(self.device)
            self.state.status = "running"
            self.state.started_at = self.state.started_at or utc_now()
            self._emit("on_train_start")
            for epoch in range(self.state.next_epoch, self.epochs):
                self.state.epoch = epoch
                sampler = getattr(train_loader, "sampler", None)
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
                self._emit("on_epoch_start")
                train_metrics = self.train_epoch(train_loader)
                metrics = {f"train_{key}": value for key, value in train_metrics.items()}
                if validation_loader is not None and (epoch + 1) % self.validate_every == 0:
                    validation_metrics = self.validate(validation_loader)
                    metrics.update({f"val_{key}": value for key, value in validation_metrics.items()})
                self.state.next_epoch = epoch + 1
                self.state.latest_metrics = metrics
                self._emit("on_epoch_end")
            self.state.status = "finished"
            self.state.finished_at = utc_now()
            self._emit("on_train_end")
            return self.state
        except BaseException as exc:
            self.state.status = "failed"
            self.state.finished_at = utc_now()
            self._emit_exception(exc)
            raise

    def train_epoch(self, loader: DataLoader[Any]) -> dict[str, float]:
        """Train for one epoch and return sample-weighted metrics."""
        self.model.train()
        totals = ClassificationTotals()
        for batch in loader:
            self._emit("on_batch_start")
            inputs, targets = self._unpack_batch(batch)
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(inputs)
            loss = self.loss_fn(logits, targets)
            loss.backward()
            self.optimizer.step()
            self.batch_metrics = totals.update(logits.detach(), targets, loss.detach())
            self.state.global_step += 1
            self._emit("on_batch_end")
        return totals.compute("training")

    def validate(self, loader: DataLoader[Any]) -> dict[str, float]:
        """Evaluate without gradients, preserving the model's previous mode."""
        was_training = self.model.training
        self.model.eval()
        totals = ClassificationTotals()
        try:
            with torch.no_grad():
                for batch in loader:
                    inputs, targets = self._unpack_batch(batch)
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    logits = self.model(inputs)
                    totals.update(logits, targets, self.loss_fn(logits, targets))
            return totals.compute("validation")
        finally:
            self.model.train(was_training)
