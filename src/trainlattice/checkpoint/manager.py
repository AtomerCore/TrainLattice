"""Atomic, state-complete PyTorch checkpoints."""

from __future__ import annotations

import logging
import random
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from trainlattice.errors import CheckpointError
from trainlattice.config import configs_compatible
from trainlattice.trainer.state import TrainerState

CHECKPOINT_FORMAT_VERSION = 4
SUPPORTED_CHECKPOINT_FORMATS = {3, 4}

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torch.amp import GradScaler


def capture_rng_state() -> dict[str, Any]:
    """Capture available process random-number-generator states."""
    state: dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        state["numpy"] = np.random.get_state()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore all RNG states available in the current process."""
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "numpy" in state:
        try:
            import numpy as np
        except ImportError:
            return
        np.random.set_state(state["numpy"])


def _restore_scheduler_position(
    scheduler: LRScheduler, state: dict[str, Any] | None, global_step: int
) -> None:
    """Restore a freshly built scheduler to the checkpointed training position.

    Checkpoints of format 4 onward carry the scheduler state verbatim, which
    restores in constant time. A checkpoint without scheduler state — format 3,
    or one saved by a run without a scheduler — can still seed a scheduler
    handed to ``load`` directly: the schedule is replayed from the base
    learning rate to the checkpointed step, which closed-form and recursive
    schedules (LambdaLR, StepLR, multiplicative variants) both reconstruct
    exactly. The replay is linear in the checkpointed step count.
    """
    if state is not None:
        scheduler.load_state_dict(state)
        return
    if global_step <= 0:
        return
    logger.info("replaying scheduler position up to global_step %d", global_step)
    with warnings.catch_warnings():
        # stepping without optimizer updates is expected during a replay
        warnings.filterwarnings(
            "ignore", message=re.escape("Detected call of `lr_scheduler.step()` before `optimizer.step()`")
        )
        # recursive schedules compound on the current group rates, so both the
        # rates and the counter restart from the schedule's own basis
        for group, base_lr in zip(scheduler.optimizer.param_groups, scheduler.base_lrs):
            group["lr"] = base_lr
        scheduler.last_epoch = 0
        for _ in range(global_step):
            scheduler.step()


def read_scheduler_budget(path: str | Path) -> int | None:
    """Read the persisted scheduler step budget from a checkpoint file.

    The budget a run's schedule was built from — the explicit
    ``scheduler.total_steps`` option or the value derived from its epochs —
    travels with the checkpoint so a resumed run keeps the schedule it started
    with; ``None`` means the checkpoint carries no budget to reuse.
    """
    source = Path(path)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CheckpointError(f"failed to read checkpoint {source}: {exc}") from exc
    budget = payload.get("scheduler_budget")
    if isinstance(budget, int) and not isinstance(budget, bool):
        return budget
    return None


class CheckpointManager:
    """Save and restore complete training state under one directory."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        name: str,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        trainer_state: TrainerState,
        config: dict[str, Any],
        resume_safe: bool = True,
        scheduler_state: dict[str, Any] | None = None,
        scaler_state: dict[str, Any] | None = None,
        scheduler_budget: int | None = None,
    ) -> Path:
        """Atomically write a named checkpoint, with scheduler and scaler state when given."""
        target = self.directory / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "trainer_state": trainer_state.to_dict(),
            "config": config,
            "rng_state": capture_rng_state(),
            "resume_safe": resume_safe,
        }
        if scheduler_state is not None:
            payload["scheduler"] = scheduler_state
            if scheduler_budget is not None:
                payload["scheduler_budget"] = scheduler_budget
        if scaler_state is not None:
            payload["scaler"] = scaler_state
        try:
            torch.save(payload, temporary)
            temporary.replace(target)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise CheckpointError(f"failed to save checkpoint {target}: {exc}") from exc
        logger.info("checkpoint saved to %s", target)
        return target

    def load(
        self,
        path: str | Path,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        map_location: str | torch.device = "cpu",
        expected_config: dict[str, Any] | None = None,
        scheduler: LRScheduler | None = None,
        scaler: GradScaler | None = None,
    ) -> TrainerState:
        """Restore model, optimizer, trainer, scheduler, scaler, and RNG state.

        A scheduler restores verbatim from the checkpoint's scheduler state; a
        scheduler handed to ``load`` for a checkpoint without one is replayed
        from the base learning rate to the saved step. A scaler restores
        verbatim when the checkpoint carries scaler state and the resumed run
        has one; through ``run_experiment`` a scaler exists exactly when the
        checkpointed run trained in fp16 — ``trainer.precision`` must match —
        so a direct ``load`` caller resuming a stateless checkpoint starts from
        the scaler's initial scale.
        """
        source = Path(path)
        if not source.is_absolute() and not source.exists():
            source = self.directory / source
        if not source.is_file():
            raise CheckpointError(f"checkpoint not found: {source}")
        try:
            payload = torch.load(source, map_location=map_location, weights_only=False)
            version = payload.get("format_version")
            if version not in SUPPORTED_CHECKPOINT_FORMATS:
                raise CheckpointError(f"unsupported checkpoint format version: {version}")
            if not payload.get("resume_safe", True):
                raise CheckpointError(
                    "checkpoint is a diagnostic emergency snapshot and cannot be resumed safely"
                )
            if expected_config is not None and not configs_compatible(
                payload.get("config", {}), expected_config
            ):
                raise CheckpointError("checkpoint configuration is incompatible with the requested run")
            model.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            trainer_state = TrainerState.from_dict(payload["trainer_state"])
            if scheduler is not None:
                _restore_scheduler_position(scheduler, payload.get("scheduler"), trainer_state.global_step)
            if scaler is not None and payload.get("scaler") is not None:
                scaler.load_state_dict(payload["scaler"])
            restore_rng_state(payload.get("rng_state", {}))
            logger.info("checkpoint restored from %s", source)
            return trainer_state
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError(f"failed to restore checkpoint {source}: {exc}") from exc
