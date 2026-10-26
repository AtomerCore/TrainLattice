"""Compose a single experiment exclusively through bound native services."""

from pathlib import Path
import inspect
import logging
from typing import Any

from trainlattice.contracts.application import ExecutionRequest, RunResult
from trainlattice.contracts.state import TrainerState, utc_now
from trainlattice.errors import ConfigError, TrainLatticeError
from .artifacts import RunContext
from .config import validate, native_config

logger = logging.getLogger("trainlattice.plugins.experiment.default")


def checkpoint_source(resume: Path) -> Path:
    """Resolve an explicit file, checkpoint directory, or run directory."""
    for candidate in (resume, resume / "last.pt", resume / "checkpoints" / "last.pt"):
        if candidate.is_file():
            return candidate
    raise TrainLatticeError(f"resume checkpoint not found: {resume}")


def constructor_source(factory: Any) -> str:
    """Identify the actual selected component implementation for inspection."""
    constructor = factory.constructor
    module = getattr(constructor, "__module__", type(constructor).__module__)
    try:
        return f"{module} ({inspect.getfile(constructor)})"
    except (TypeError, OSError):
        return module


def execute(request: ExecutionRequest, services: dict[str, Any]) -> RunResult:
    """Validate, construct, restore, run, and record one frozen composition."""
    config = validate(request.config, services)
    runtime = services["runtime"]
    device = runtime.device(config.trainer.device)
    if config.trainer.precision == "fp16" and device.type != "cuda":
        raise ConfigError("trainer.precision 'fp16' requires a cuda device; use 'bf16' on cpu")
    source = checkpoint_source(request.resume) if request.resume is not None else None
    # A store can preflight before any training object or run directory exists.
    preflight_store = services["checkpoint"].create(Path("."), snapshot=request.snapshot, state_providers=request.state_providers)
    payload = preflight_store.preflight(source, expected_config=config.to_dict()) if source is not None else None
    runtime.seed(config.experiment.seed)
    model = services["model"].build(**config.model.options).to(device)
    optimizer = services["optimizer"].build(model.parameters(), lr=config.optimizer.lr, **config.optimizer.options)
    loss_fn = services["loss"].build(**config.loss.options).to(device)
    loaders = services["dataset"].build_loaders(config.dataset, seed=config.experiment.seed)
    scheduler = None
    scheduler_budget = None
    if config.scheduler is not None:
        steps = (len(loaders.train) + config.trainer.accumulate - 1) // config.trainer.accumulate
        scheduler_budget = config.scheduler.options.get("total_steps", steps * config.trainer.epochs)
        if payload is not None and "total_steps" not in config.scheduler.options:
            scheduler_budget = payload["scheduler_budget"]
            logger.info("scheduler budget anchored to %s checkpointed steps", scheduler_budget)
        scheduler = services["scheduler"].create(config.scheduler, optimizer, total_steps=scheduler_budget)
    context = RunContext.create(config, command=request.command, runtime=runtime,
                                native_config=native_config(config, request.snapshot), snapshot=request.snapshot,
                                **{f"{key}_source": constructor_source(services[key]) for key in ("model", "dataset", "optimizer", "loss")})
    logger.info("run directory %s", context.path)
    try:
        store = services["checkpoint"].create(context.path / "checkpoints", snapshot=request.snapshot, state_providers=request.state_providers)
        metric_logger = services["logging"](context.path)
        callbacks = services["callbacks"](logger=metric_logger, checkpoint=store, config=config.to_dict(),
                                          log_every=config.trainer.log_every, monitor=config.trainer.monitor,
                                          scheduler_budget=scheduler_budget, extra=request.callbacks)
        trainer = services["trainer"](model=model, optimizer=optimizer, loss_fn=loss_fn, device=device,
                                      epochs=config.trainer.epochs, callbacks=callbacks,
                                      validate_every=config.trainer.validate_every,
                                      max_grad_norm=config.trainer.max_grad_norm, state=TrainerState(),
                                      scheduler=scheduler, accumulate=config.trainer.accumulate,
                                      precision=config.trainer.precision, task=services["loss"].task.kind,
                                      monitor=config.trainer.monitor, early_stopping=config.trainer.early_stopping)
        if payload is not None:
            trainer.state = store.restore(payload, model=model, optimizer=optimizer, scheduler=scheduler, scaler=trainer.scaler)
            logger.info("resumed from checkpoint %s; next epoch %d", source, trainer.state.next_epoch)
        trainer.state.started_at = trainer.state.started_at or utc_now()
        context.update(status="running", started_at=trainer.state.started_at, resume_source=str(source) if source else None)
        final_state = trainer.fit(loaders.train, loaders.validation)
    except BaseException:
        context.update(status="failed", finished_at=utc_now())
        raise
    context.update(status=final_state.status, started_at=final_state.started_at, finished_at=final_state.finished_at,
                   epoch=final_state.epoch, global_step=final_state.global_step)
    return RunResult(run_dir=context.path, state=final_state)
