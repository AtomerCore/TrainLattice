"""Atomic, state-complete PyTorch checkpoints."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from trainlattice.errors import CheckpointError
from trainlattice.config import configs_compatible
from trainlattice.trainer.state import TrainerState

CHECKPOINT_FORMAT_VERSION = 3
SUPPORTED_CHECKPOINT_FORMATS = {3}

logger = logging.getLogger(__name__)


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
    ) -> Path:
        """Atomically write a named checkpoint."""
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
    ) -> TrainerState:
        """Restore model, optimizer, trainer state, and available RNG state."""
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
            restore_rng_state(payload.get("rng_state", {}))
            trainer_state = TrainerState.from_dict(payload["trainer_state"])
            logger.info("checkpoint restored from %s", source)
            return trainer_state
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError(f"failed to restore checkpoint {source}: {exc}") from exc
