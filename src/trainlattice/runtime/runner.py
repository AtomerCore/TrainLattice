"""Compose components and execute an experiment."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from trainlattice.callbacks import Callback, CheckpointCallback, LoggingCallback
from trainlattice.checkpoint import CheckpointManager
from trainlattice.config import ExperimentConfig, compare_contracts, load_config
from trainlattice.datasets import DATASETS, build_dataloaders, validate_dataset_config
from trainlattice.errors import ConfigError, TrainLatticeError
from trainlattice.logging import CompositeLogger, ConsoleLogger, JSONLLogger
from trainlattice.modeling import ARCHITECTURES, build_model, validate_model_config
from trainlattice.optim import OPTIMIZERS, build_optimizer, validate_optimizer_config
from trainlattice.runtime.device import select_device
from trainlattice.runtime.experiment import RunContext
from trainlattice.runtime.seed import seed_everything
from trainlattice.trainer import (
    LOSSES,
    Trainer,
    TrainerState,
    build_loss,
    loss_task,
    validate_loss_config,
)
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


def _check_contracts(config: ExperimentConfig) -> None:
    """Compare the registered model and dataset contracts before anything is built."""
    model_contract = ARCHITECTURES.entry(config.model.name).contract
    dataset_contract = DATASETS.entry(config.dataset.name).contract
    if model_contract is None or dataset_contract is None:
        return
    errors = compare_contracts(
        model_contract(config.model.options),
        dataset_contract(config.dataset.options),
    )
    if errors:
        raise ConfigError("; ".join(errors))


def _check_monitor_matches_task(config: ExperimentConfig) -> None:
    """Reject accuracy monitoring for losses that do not produce it."""
    if config.trainer.monitor == "val_accuracy" and loss_task(config.loss.name) != "classification":
        raise ConfigError(
            f"trainer.monitor 'val_accuracy' requires a classification loss; loss.name {config.loss.name!r} is regression"
        )


def _constructor_source(constructor: Callable) -> str:
    """Locate the module that defined a registered constructor."""
    module = getattr(constructor, "__module__", None) or type(constructor).__module__
    try:
        file = inspect.getfile(constructor)
    except (TypeError, OSError):
        return module
    return f"{module} ({file})"


def run_experiment(
    config: ExperimentConfig,
    *,
    command: str,
    resume: str | Path | None = None,
    extra_callbacks: Sequence[Callback] | None = None,
) -> RunResult:
    """Build and execute one configured experiment.

    ``extra_callbacks`` extend the lifecycle between the built-in logging
    and checkpointing callbacks; they observe the trainer and cannot
    interrupt it.
    """
    config.validate()
    seed_everything(config.experiment.seed)
    explicit_source: Path | None = None
    if resume is not None:
        explicit_source = _checkpoint_source(resume)
    validate_model_config(config.model)
    validate_dataset_config(config.dataset)
    validate_optimizer_config(config.optimizer)
    validate_loss_config(config.loss)
    _check_monitor_matches_task(config)
    _check_contracts(config)
    device = select_device(config.trainer.device)
    model = build_model(config.model)
    optimizer = build_optimizer(config.optimizer, model.parameters())
    loss_fn = build_loss(config.loss)
    loaders = build_dataloaders(config.dataset, seed=config.experiment.seed)
    context = RunContext.create(
        config,
        command=command,
        model_source=_constructor_source(ARCHITECTURES.get(config.model.name)),
        dataset_source=_constructor_source(DATASETS.get(config.dataset.name)),
        optimizer_source=_constructor_source(OPTIMIZERS.get(config.optimizer.name)),
        loss_source=_constructor_source(LOSSES.get(config.loss.name)),
    )
    logger.info("run directory %s", context.path)
    manager = CheckpointManager(context.path / "checkpoints")
    state = TrainerState()
    source: Path | None = explicit_source
    metric_logger = CompositeLogger([
        ConsoleLogger(),
        JSONLLogger(context.path / "metrics.jsonl"),
    ])
    callbacks: list[Callback] = [
        LoggingCallback(metric_logger, log_every=config.trainer.log_every),
        *(extra_callbacks or ()),
    ]
    callbacks.append(
        CheckpointCallback(manager, config.to_dict(), monitor=config.trainer.monitor)
    )
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
        loss_fn=loss_fn,
        device=device,
        epochs=config.trainer.epochs,
        callbacks=callbacks,
        validate_every=config.trainer.validate_every,
        max_grad_norm=config.trainer.max_grad_norm,
        state=state,
        task=loss_task(config.loss.name),
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
    extra_callbacks: Sequence[Callback] | None = None,
) -> RunResult:
    """Load a YAML file and execute it."""
    return run_experiment(
        load_config(path), command=command, resume=resume, extra_callbacks=extra_callbacks
    )
