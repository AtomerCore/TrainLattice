"""Atomic checkpoint v5 persistence with graph and state preflight."""

from __future__ import annotations

from collections.abc import Mapping
import random
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from trainlattice.contracts.state import TrainerState
from trainlattice.errors import CheckpointError
from trainlattice.kernel.api import StateProvider
from trainlattice.kernel.snapshot import PluginSnapshot
from .config import configs_compatible

CHECKPOINT_FORMAT_VERSION = 5
logger = logging.getLogger("trainlattice.plugins.checkpoint.local")


def capture_rng_state() -> dict[str, Any]:
    """Capture process generators using weights-only-loadable values."""
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        name, keys, position, gaussian, cached = np.random.get_state()
        state["numpy"] = (name, keys.tolist(), position, gaussian, cached)
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore generators after every component and plugin has restored."""
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    if "numpy" in state:
        import numpy as np
        name, keys, position, gaussian, cached = state["numpy"]
        np.random.set_state((name, np.array(keys, dtype="uint32"), position, gaussian, cached))


class CheckpointStore:
    """Save and restore training state for one frozen native plugin graph."""

    def __init__(self, directory: str | Path, *, snapshot: PluginSnapshot,
                 state_providers: Mapping[str, StateProvider]) -> None:
        self.directory = Path(directory)
        self.snapshot = snapshot
        self.state_providers = state_providers

    def save(self, name: str, *, model: Any, optimizer: Any, trainer_state: TrainerState,
             config: dict[str, Any], resume_safe: bool = True,
             scheduler_state: dict[str, Any] | None = None,
             scaler_state: dict[str, Any] | None = None,
             scheduler_budget: int | None = None) -> Path:
        """Atomically persist a complete epoch boundary or diagnostic snapshot."""
        if Path(name).name != name:
            raise CheckpointError("checkpoint name must be a filename")
        target = self.directory / name
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            payload = {
                "format_version": CHECKPOINT_FORMAT_VERSION, "resume_safe": resume_safe,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "trainer_state": trainer_state.to_dict(), "config": config,
                "plugins_snapshot": self.snapshot.to_dict(),
                "plugin_states": {key: {"schema_version": provider.state_schema_version,
                                        "payload": provider.snapshot_state()}
                                  for key, provider in self.state_providers.items()},
                "scheduler": scheduler_state, "scheduler_budget": scheduler_budget,
                "scaler": scaler_state,
            }
            payload["rng_state"] = capture_rng_state()
            self.directory.mkdir(parents=True, exist_ok=True)
            torch.save(payload, temporary)
            temporary.replace(target)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise CheckpointError(f"failed to save checkpoint {target}: {exc}") from exc
        logger.info("checkpoint saved to %s", target)
        return target

    def preflight(self, path: str | Path, *, expected_config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Read and validate the full recovery protocol before mutating objects."""
        source = Path(path)
        if not source.is_absolute() and not source.exists():
            source = self.directory / source
        try:
            payload = torch.load(source, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
                raise CheckpointError("unsupported checkpoint format version; expected 5")
            if payload.get("resume_safe") is not True:
                raise CheckpointError("checkpoint is a diagnostic emergency snapshot and cannot be resumed safely")
            required = {"model", "optimizer", "trainer_state", "config", "rng_state", "plugins_snapshot", "plugin_states", "scheduler", "scheduler_budget", "scaler"}
            missing = required - payload.keys()
            if missing:
                raise CheckpointError(f"checkpoint missing state: {', '.join(sorted(missing))}")
            saved = PluginSnapshot(payload["plugins_snapshot"]).identity()
            expected = self.snapshot.identity()
            if saved != expected:
                saved_plugins = {item["id"]: item for item in saved["plugins"]}
                expected_plugins = {item["id"]: item for item in expected["plugins"]}
                changed = sorted(key for key in saved_plugins.keys() | expected_plugins.keys()
                                 if saved_plugins.get(key) != expected_plugins.get(key))
                detail = ", ".join(changed) or "shared contracts fingerprint"
                raise CheckpointError(f"checkpoint plugin graph or fingerprint mismatch: {detail}")
            states = payload["plugin_states"]
            if set(states) != set(self.state_providers):
                raise CheckpointError("checkpoint plugin state providers do not match the active graph")
            for key, provider in self.state_providers.items():
                state = states[key]
                if not isinstance(state, dict) or state.get("schema_version") != provider.state_schema_version:
                    raise CheckpointError(f"checkpoint plugin state schema mismatch: {key}")
                if not isinstance(state.get("payload"), dict):
                    raise CheckpointError(f"checkpoint missing plugin state payload: {key}")
            if not isinstance(payload["rng_state"], dict) or not {"python", "torch"} <= payload["rng_state"].keys():
                raise CheckpointError("checkpoint missing RNG state")
            TrainerState.from_dict(payload["trainer_state"])
            if expected_config is not None:
                if not configs_compatible(payload["config"], expected_config):
                    raise CheckpointError("checkpoint configuration is incompatible with the requested run")
                if expected_config.get("scheduler") is not None and payload["scheduler"] is None:
                    raise CheckpointError("checkpoint missing scheduler state")
                if expected_config.get("trainer", {}).get("precision") == "fp16" and payload["scaler"] is None:
                    raise CheckpointError("checkpoint missing scaler state")
            return payload
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError(f"failed to read checkpoint {source}: {exc}") from exc

    def restore(self, payload: dict[str, Any], *, model: Any, optimizer: Any,
                scheduler: Any = None, scaler: Any = None) -> TrainerState:
        """Restore a preflighted payload, ordered plugin state, and finally RNG."""
        try:
            model.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            if scheduler is not None:
                if payload["scheduler"] is None:
                    raise CheckpointError("checkpoint missing scheduler state")
                scheduler.load_state_dict(payload["scheduler"])
            if scaler is not None:
                if payload["scaler"] is None:
                    raise CheckpointError("checkpoint missing scaler state")
                scaler.load_state_dict(payload["scaler"])
            state = TrainerState.from_dict(payload["trainer_state"])
            for key, provider in self.state_providers.items():
                provider.restore_state(payload["plugin_states"][key]["payload"])
            restore_rng_state(payload["rng_state"])
            return state
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError(f"failed to restore checkpoint state: {exc}") from exc

    def load(self, path: str | Path, *, expected_config: dict[str, Any] | None = None,
             **objects: Any) -> TrainerState:
        """Preflight and restore directly into existing training objects."""
        state = self.restore(self.preflight(path, expected_config=expected_config), **objects)
        logger.info("checkpoint restored from %s", path)
        return state


class CheckpointFactory:
    """Construct graph-bound stores without creating run artifacts."""

    def create(self, directory: str | Path, *, snapshot: PluginSnapshot,
               state_providers: Mapping[str, StateProvider]) -> CheckpointStore:
        """Bind persistence to the application's snapshot and state providers."""
        return CheckpointStore(directory, snapshot=snapshot, state_providers=state_providers)
