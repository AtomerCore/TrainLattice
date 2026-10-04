"""Compose components and execute an experiment."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from trainlattice.callbacks import CheckpointCallback, LoggingCallback
from trainlattice.checkpoint import CheckpointManager
from trainlattice.config import ExperimentConfig, load_config
from trainlattice.datasets import build_dataloaders, validate_dataset_config
from trainlattice.errors import TrainLatticeError
from trainlattice.logging import CompositeLogger, ConsoleLogger, JSONLLogger
from trainlattice.modeling import build_model
from trainlattice.modeling.builder import validate_model_config
from trainlattice.runtime.device import select_device
from trainlattice.runtime.experiment import RunContext
from trainlattice.runtime.seed import seed_everything
from trainlattice.trainer import Trainer, TrainerState
from trainlattice.trainer.state import utc_now

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RunResult:
    """Completed run identity and final state."""

    run_dir: Path
    state: TrainerState


def _checkpoint_source(resume: str | Path) -> Path:
    candidate = Path(resume)
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        direct = candidate / "last.pt"
        nested = candidate / "checkpoints" / "last.pt"
        if direct.is_file():
            return direct
        if nested.is_file():
            return nested
    raise TrainLatticeError(f"resume checkpoint not found: {resume}")


def run_experiment(
    config: ExperimentConfig,
    *,
    command: str,
    resume: str | Path | None = None,
) -> RunResult:
    """Build and execute one configured experiment."""
    config.validate()
    seed_everything(config.experiment.seed)
    explicit_source: Path | None = None
    if resume is not None:
        explicit_source = _checkpoint_source(resume)
    validate_model_config(config.model)
    validate_dataset_config(config.dataset)
    device = select_device(config.trainer.device)
    model = build_model(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.lr)
    loaders = build_dataloaders(config.dataset, seed=config.experiment.seed)
    context = RunContext.create(config, command=command)
    logger.info("run directory %s", context.path)
    manager = CheckpointManager(context.path / "checkpoints")
    state = TrainerState()
    source: Path | None = explicit_source
    metric_logger = CompositeLogger([
        ConsoleLogger(),
        JSONLLogger(context.path / "metrics.jsonl"),
    ])
    callbacks = [
        LoggingCallback(metric_logger, log_every=config.trainer.log_every),
    ]
    callbacks.append(CheckpointCallback(manager, config.to_dict()))
    if resume is not None:
        try:
            state = manager.load(
                source,
                model=model,
                optimizer=optimizer,
                map_location=device,
                expected_config=config.to_dict(),
            )
        except BaseException as exc:
            logger.error("marking run manifest failed after checkpoint restore error: %s", exc)
            metric_logger.close()
            context.update(status="failed", finished_at=utc_now())
            raise
        logger.info("resumed from checkpoint %s; next epoch %d", source, state.next_epoch)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        device=device,
        epochs=config.trainer.epochs,
        callbacks=callbacks,
        validate_every=config.trainer.validate_every,
        state=state,
    )
    state.started_at = state.started_at or utc_now()
    context.update(status="running", started_at=state.started_at, resume_source=str(source) if source else None)
    try:
        final_state = trainer.fit(loaders.train, loaders.validation)
    except BaseException as exc:
        logger.error("marking run manifest failed after training error: %s", exc)
        context.update(
            status=trainer.state.status,
            finished_at=trainer.state.finished_at,
            epoch=trainer.state.epoch,
            global_step=trainer.state.global_step,
        )
        raise
    context.update(
        status=final_state.status,
        started_at=final_state.started_at,
        finished_at=final_state.finished_at,
        epoch=final_state.epoch,
        global_step=final_state.global_step,
    )
    return RunResult(run_dir=context.path, state=final_state)


def run_from_file(
    path: str | Path,
    *,
    resume: str | Path | None = None,
    command: str = "trainlattice train",
) -> RunResult:
    """Load a YAML file and execute it."""
    return run_experiment(load_config(path), command=command, resume=resume)
