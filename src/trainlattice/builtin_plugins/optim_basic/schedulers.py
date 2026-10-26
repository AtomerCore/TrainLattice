"""Learning-rate scheduler registration, validation, and construction.

Both warmup schedulers drive ``torch.optim.lr_scheduler.LambdaLR`` with an
explicit multiplier over the step counter ``t`` — ``t`` is 0 when the scheduler
is created and grows by one per ``scheduler.step()`` call, so the learning rate
applied by optimizer update ``t + 1`` is ``lr(t)``:

- warmup phase, ``0 <= t < warmup_steps``:
  ``lr(t) = base_lr * t / warmup_steps`` — the rate grows linearly from 0 and
  reaches the base learning rate exactly at step ``warmup_steps``;
- cosine decay, ``warmup_steps <= t < total_steps``: with
  ``p = (t - warmup_steps) / (total_steps - warmup_steps)``,
  ``lr(t) = min_lr + (base_lr - min_lr) * (1 + cos(pi * p)) / 2``;
- linear decay over the same span:
  ``lr(t) = min_lr + (base_lr - min_lr) * (1 - p)``;
- from ``total_steps`` on: ``lr(t) = min_lr``.

``min_lr`` is an absolute learning rate floor converted internally to a
multiplier of the optimizer's base rate, and it must not exceed that base
rate — a higher floor would turn the decay into a climb. ``warmup_steps == 0``
skips the warmup phase, and ``warmup_steps`` must stay below
``total_steps``. ``total_steps`` may be stated in the scheduler options or
derived by the runtime and passed to ``build_scheduler``; the explicit option
wins.

``step_decay`` wraps ``torch.optim.lr_scheduler.StepLR``: with step counter
``t`` as above, ``lr(t) = base_lr * gamma ** floor(t / step_size)``. It takes a
required integer ``step_size >= 1`` and a positive ``gamma`` defaulting to 0.1.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import cos, pi
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, LambdaLR, StepLR

from trainlattice.contracts.config import SchedulerConfig
from trainlattice.errors import ConfigError
from trainlattice.contracts.components import ComponentFactory


_WARMUP_NAMES = frozenset({"warmup_cosine", "warmup_linear"})


def _reject_unknown(path: str, options: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(options) - allowed)
    if unknown:
        keys = ", ".join(f"{path}.{key}" for key in unknown)
        raise ConfigError(f"unknown configuration field(s): {keys}")


def _non_negative_int(value: Any, path: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigError(f"{path} must be a non-negative integer")


def _positive_int(value: Any, path: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConfigError(f"{path} must be a positive integer")


def _non_negative_number(value: Any, path: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ConfigError(f"{path} must be a non-negative number")


def _positive_number(value: Any, path: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{path} must be a positive number")


def _check_warmup_bounds(warmup_steps: int, total_steps: int) -> None:
    """Require at least one decay step after the warmup phase."""
    if warmup_steps >= total_steps:
        raise ConfigError("scheduler.warmup_steps must be less than scheduler.total_steps")


def _check_floor(min_lr: float, base_lr: float) -> None:
    """Reject a decay floor above the base rate, which would make the schedule climb."""
    if min_lr > base_lr:
        raise ConfigError("scheduler.min_lr must not exceed optimizer.lr")


def _validate_warmup(path: str, options: Mapping[str, Any]) -> None:
    """Validate the warmup_cosine and warmup_linear options."""
    _reject_unknown(path, options, {"warmup_steps", "total_steps", "min_lr"})
    if "warmup_steps" in options:
        _non_negative_int(options["warmup_steps"], f"{path}.warmup_steps")
    if "total_steps" in options:
        _positive_int(options["total_steps"], f"{path}.total_steps")
    if "min_lr" in options:
        _non_negative_number(options["min_lr"], f"{path}.min_lr")
    if "total_steps" in options:
        _check_warmup_bounds(options.get("warmup_steps", 0), options["total_steps"])


def _validate_step_decay(path: str, options: Mapping[str, Any]) -> None:
    """Validate the step_decay options."""
    _reject_unknown(path, options, {"step_size", "gamma"})
    if "step_size" not in options:
        raise ConfigError(f"{path}.step_size is required")
    _positive_int(options["step_size"], f"{path}.step_size")
    if "gamma" in options:
        _positive_number(options["gamma"], f"{path}.gamma")


def _decay_fraction(step: int, warmup_steps: int, total_steps: int) -> float:
    """Return the decay progress in [0, 1], saturating after total_steps."""
    return min((step - warmup_steps) / (total_steps - warmup_steps), 1.0)


def _build_warmup_cosine(
    optimizer: Optimizer, total_steps: int, *, warmup_steps: int = 0, min_lr: float = 0.0
) -> LambdaLR:
    """Build the linear-warmup, cosine-decay schedule."""
    _check_warmup_bounds(warmup_steps, total_steps)
    base_lr = optimizer.param_groups[0]["lr"]
    _check_floor(min_lr, base_lr)

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = _decay_fraction(step, warmup_steps, total_steps)
        lr = min_lr + (base_lr - min_lr) * (1.0 + cos(pi * progress)) / 2.0
        return lr / base_lr

    return LambdaLR(optimizer, multiplier)


def _build_warmup_linear(
    optimizer: Optimizer, total_steps: int, *, warmup_steps: int = 0, min_lr: float = 0.0
) -> LambdaLR:
    """Build the linear-warmup, linear-decay schedule."""
    _check_warmup_bounds(warmup_steps, total_steps)
    base_lr = optimizer.param_groups[0]["lr"]
    _check_floor(min_lr, base_lr)

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = _decay_fraction(step, warmup_steps, total_steps)
        lr = min_lr + (base_lr - min_lr) * (1.0 - progress)
        return lr / base_lr

    return LambdaLR(optimizer, multiplier)


def _build_step_decay(optimizer: Optimizer, _total_steps: int | None, *, step_size: int, gamma: float = 0.1) -> StepLR:
    """Build the staircase decay schedule; total_steps is unused."""
    return StepLR(optimizer, step_size=step_size, gamma=gamma)


class SchedulerFactory(ComponentFactory):
    """A scheduler factory that resolves and validates its step budget."""

    def create(self, config: SchedulerConfig, optimizer: Optimizer, total_steps: int | None = None) -> LRScheduler:
        """Construct the selected schedule with explicit-budget precedence."""
        self.validate("scheduler", config.options)
        resolved = config.options.get("total_steps", total_steps)
        if resolved is not None:
            _positive_int(resolved, "scheduler.total_steps")
        elif self.name in _WARMUP_NAMES:
            raise ConfigError(f"scheduler.total_steps is required by {self.name}; set it in the scheduler section or derive it from the run")
        options = {key: value for key, value in config.options.items() if key != "total_steps"}
        return self.build(optimizer, resolved, **options)
