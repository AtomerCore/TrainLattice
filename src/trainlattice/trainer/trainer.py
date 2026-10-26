"""Transparent single-process training lifecycle with task-aware metrics."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from trainlattice.config.schema import EarlyStoppingConfig
from trainlattice.trainer.losses import TASKS
from trainlattice.trainer.metrics import ClassificationTotals, LossTotals
from trainlattice.trainer.state import TrainerState, utc_now

if TYPE_CHECKING:
    from trainlattice.callbacks.base import Callback

try:
    from torch.amp import GradScaler  # torch >= 2.3 exposes the device-generic scaler
except ImportError:  # torch 2.1 and 2.2 expose it under torch.cuda.amp only
    from torch.cuda.amp import GradScaler

logger = logging.getLogger(__name__)

PRECISION_MODES = ("fp32", "bf16", "fp16")
MONITORED_METRICS = ("val_loss", "val_accuracy")
# epoch-scoped timing keys that enter epoch records without the train_ prefix
EPOCH_TIMING_KEYS = ("epoch_seconds", "data_seconds")


def build_grad_scaler() -> GradScaler:
    """Create the cuda loss scaler used by fp16 training.

    ``torch.amp.GradScaler`` gained its ``device`` argument after the
    package's torch 2.1 floor, so the older constructor is tried without
    arguments; fp16 training itself only ever runs on cuda devices.
    """
    try:
        return GradScaler(device="cuda")
    except TypeError:
        return GradScaler()


class Trainer:
    """Run training and validation while exposing lifecycle events.

    ``precision`` selects the numeric mode: ``fp32`` keeps every computation
    in the model's dtype, while ``bf16`` and ``fp16`` run training and
    validation forward passes under ``torch.amp.autocast`` with the matching
    dtype. Only ``fp16`` adds a ``GradScaler`` — its scaled backward calls
    accumulate across an accumulation window and one
    unscale/clip/step/update sequence closes it — and ``fp16`` requires a
    cuda device, because cpu autocast supports bfloat16 only; ``bf16`` is
    the mixed-precision mode for cpu.

    ``early_stopping`` arms the built-in convergence check on the monitored
    metric, and any callback can end the run cooperatively through
    ``request_stop``; either path finishes the current epoch — including its
    checkpoint save — before the loop ends with status ``finished`` and the
    reason recorded in ``state.stop_reason``.
    """

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
        max_grad_norm: float | None = None,
        state: TrainerState | None = None,
        scheduler: LRScheduler | None = None,
        accumulate: int = 1,
        precision: str = "fp32",
        task: str = "classification",
        monitor: str = "val_loss",
        early_stopping: EarlyStoppingConfig | None = None,
    ) -> None:
        if task not in TASKS:
            raise ValueError(f"task must be one of: {', '.join(TASKS)}")
        if not isinstance(accumulate, int) or isinstance(accumulate, bool) or accumulate < 1:
            raise ValueError("accumulate must be an integer of at least 1")
        if precision not in PRECISION_MODES:
            raise ValueError(f"precision must be one of: {', '.join(PRECISION_MODES)}")
        if precision == "fp16" and device.type != "cuda":
            raise ValueError(
                f"precision 'fp16' requires a cuda device (got {device.type}); "
                "use precision 'bf16' for mixed precision on this device"
            )
        if monitor not in MONITORED_METRICS:
            raise ValueError(f"monitor must be one of: {', '.join(MONITORED_METRICS)}")
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.epochs = epochs
        self.callbacks = list(callbacks)
        self.validate_every = validate_every
        self.max_grad_norm = max_grad_norm
        self.state = state or TrainerState()
        self.scheduler = scheduler
        self.accumulate = accumulate
        self.precision = precision
        self.scaler = build_grad_scaler() if precision == "fp16" else None
        self.monitor = monitor
        self.early_stopping = early_stopping
        self.batch_metrics: dict[str, float] = {}
        self._amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
        # the task kind decides whether accuracy joins loss in the aggregates
        self._totals_cls = ClassificationTotals if task == "classification" else LossTotals

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

    def _autocast(self) -> AbstractContextManager[None]:
        """Wrap forward computation in autocast under the configured precision."""
        if self._amp_dtype is None:
            return nullcontext()
        return torch.amp.autocast(device_type=self.device.type, dtype=self._amp_dtype)

    def _backward_loss(self, loss: Tensor) -> None:
        """Backpropagate the natural scalar; element-wise losses collapse to their mean.

        Under fp16 the scalar goes through the GradScaler so the accumulated
        window gradients stay within float16 range; the window-closing step
        unscales them again.
        """
        scalar = loss.mean() if loss.numel() > 1 else loss
        if self.scaler is not None:
            self.scaler.scale(scalar).backward()
        else:
            scalar.backward()

    def _mean_loss(self, loss: Tensor, samples: int) -> Tensor:
        """Normalize a batch loss to its per-sample mean for sample-weighted metrics."""
        reduction = getattr(self.loss_fn, "reduction", "mean")
        if reduction == "none":
            return loss.mean()
        if reduction == "sum":
            return loss / samples
        return loss

    def fit(self, train_loader: DataLoader[Any], validation_loader: DataLoader[Any] | None = None) -> TrainerState:
        """Execute the configured lifecycle from the state's next epoch.

        Early stopping and ``request_stop`` end the loop once the current
        epoch's callbacks — including the checkpoint save — have finished;
        the run still completes with status ``finished`` and records why it
        ended in ``state.stop_reason``. Each new ``fit`` clears the reason,
        so the field always describes the lifecycle call that produced it.

        Epoch records carry the epoch's timing — ``epoch_seconds`` total,
        ``data_seconds`` spent waiting for batches, and the epoch-wide
        ``train_samples_per_sec`` rate — and, on cuda devices,
        ``max_memory_mb``, the epoch's peak ``torch.cuda.max_memory_allocated``
        measured from a peak statistic reset at the epoch start.
        """
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
                timing = {key: value for key, value in train_metrics.items() if key in EPOCH_TIMING_KEYS}
                metrics = {
                    f"train_{key}": value
                    for key, value in train_metrics.items()
                    if key not in EPOCH_TIMING_KEYS
                }
                metrics.update(timing)
                # the end-of-epoch rate the next optimizer step would apply
                metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
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

    @staticmethod
    def _rate(count: float, seconds: float) -> float:
        """Divide a completed count by elapsed seconds, tolerating a zero-resolution clock."""
        return count / seconds if seconds > 0 else 0.0

    def _throughput(self, started_at: float, steps: int, samples: int) -> dict[str, float]:
        """Return the cumulative-average rates since the epoch started.

        ``steps_per_sec`` and ``samples_per_sec`` divide the optimizer steps
        and samples completed so far in this epoch by the wall-clock seconds
        elapsed since the epoch began, data waits included.
        """
        elapsed = time.perf_counter() - started_at
        return {
            "steps_per_sec": self._rate(steps, elapsed),
            "samples_per_sec": self._rate(samples, elapsed),
        }

    def _optimizer_step(self, window: LossTotals | ClassificationTotals, throughput: dict[str, float]) -> None:
        """Apply one optimizer update over the gradients accumulated in a window.

        Under fp16 the window's losses were backwarded through the shared
        GradScaler: gradients are unscaled first, an optional clip then sees
        true magnitudes, the scaler applies the update — skipping it when the
        scaled gradients were non-finite — and its scale adapts once per
        optimizer step. A skipped window applies no update, advances no
        scheduler, and counts no optimizer step. ``batch_metrics`` joins the
        window aggregate with the learning rate the step applied, or would
        have applied, and the epoch-to-date throughput rates.
        """
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        step_lr = float(self.optimizer.param_groups[0]["lr"])
        stepped = True
        if self.scaler is not None:
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # a backed-off scale marks the window the scaler just skipped
            stepped = self.scaler.get_scale() >= scale_before
        else:
            self.optimizer.step()
        if stepped and self.scheduler is not None:
            self.scheduler.step()
        self.batch_metrics = {**window.compute("training"), "lr": step_lr, **throughput}
        if stepped:
            self.state.global_step += 1

    def train_epoch(self, loader: DataLoader[Any]) -> dict[str, float]:
        """Train for one epoch and return sample-weighted metrics.

        Micro-batches are consumed in windows of ``accumulate`` batches:
        gradients are zeroed only when a window opens, each micro-batch loss
        is scaled by ``1 / accumulate`` before backward — so with a
        mean-reduced loss over equal-sized micro-batches the accumulated
        gradient equals the gradient of the full batch — and one optimizer
        step closes a window. A partial window at the epoch end is flushed as
        one final step with the same scaling. Hooks fire per micro-batch, so
        callbacks observe data batches, while ``state.global_step`` counts
        optimizer steps and ``batch_metrics`` describes the window the latest
        step consumed.

        ``batch_metrics["lr"]`` is the learning rate that step applied, read
        from the first parameter group before the step; a configured scheduler
        advances once per optimizer step that applies an update, after it. An
        epoch therefore runs at most ``ceil(batches / accumulate)`` optimizer
        steps — fewer only when the fp16 scaler skips windows with non-finite
        gradients.

        The epoch is timed with ``time.perf_counter``, which touches no
        random-number generator. Batches are fetched through an explicit
        iterator so the wait for each ``next`` call accumulates into
        ``data_seconds``; every other second — transfer, forward, backward,
        and the optimizer step — is compute time. ``steps_per_sec`` and
        ``samples_per_sec`` in ``batch_metrics`` are cumulative averages from
        the start of the epoch: optimizer steps and samples completed so far
        divided by the elapsed seconds, data waits included. The returned
        dict carries the same epoch-wide ``samples_per_sec`` alongside
        ``epoch_seconds``, the epoch's total duration, and ``data_seconds``,
        the share of it spent waiting for batches.

        Under ``bf16``/``fp16`` precision the forward pass and loss run
        inside autocast while the backward stays outside it; ``fp16`` scales
        each backward through the GradScaler, and the window-closing step
        unscales, clips, applies the scaled update, then adapts the scale.
        """
        self.model.train()
        totals = self._totals_cls()
        window = self._totals_cls()
        batches_in_window = 0
        started_at = time.perf_counter()
        data_seconds = 0.0
        steps_in_epoch = 0
        batches = iter(loader)
        while True:
            fetch_started_at = time.perf_counter()
            try:
                batch = next(batches)
            except StopIteration:
                break
            data_seconds += time.perf_counter() - fetch_started_at
            self._emit("on_batch_start")
            if batches_in_window == 0:
                self.optimizer.zero_grad(set_to_none=True)
                window = self._totals_cls()
            inputs, targets = self._unpack_batch(batch)
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with self._autocast():
                logits = self.model(inputs)
                loss = self.loss_fn(logits, targets)
            scaled = loss / self.accumulate if self.accumulate > 1 else loss
            self._backward_loss(scaled)
            per_sample = self._mean_loss(loss.detach(), targets.shape[0])
            totals.update(logits.detach(), targets, per_sample)
            window.update(logits.detach(), targets, per_sample)
            batches_in_window += 1
            if batches_in_window == self.accumulate:
                steps_in_epoch += 1
                self._optimizer_step(window, self._throughput(started_at, steps_in_epoch, totals.samples))
                batches_in_window = 0
            self._emit("on_batch_end")
        if batches_in_window != 0:
            steps_in_epoch += 1
            self._optimizer_step(window, self._throughput(started_at, steps_in_epoch, totals.samples))
        epoch_seconds = time.perf_counter() - started_at
        return {
            **totals.compute("training"),
            "samples_per_sec": self._rate(totals.samples, epoch_seconds),
            "epoch_seconds": epoch_seconds,
            "data_seconds": data_seconds,
        }

    def validate(self, loader: DataLoader[Any]) -> dict[str, float]:
        """Evaluate without gradients, preserving the model's previous mode."""
        was_training = self.model.training
        self.model.eval()
        totals = self._totals_cls()
        try:
            with torch.no_grad():
                for batch in loader:
                    inputs, targets = self._unpack_batch(batch)
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    with self._autocast():
                        logits = self.model(inputs)
                        loss = self.loss_fn(logits, targets)
                    totals.update(logits, targets, self._mean_loss(loss.detach(), targets.shape[0]))
            return totals.compute("validation")
        finally:
            self.model.train(was_training)
