"""Scheduler configuration section, registry, and construction behavior."""

from __future__ import annotations

import json
from math import cos, pi
from pathlib import Path

import pytest

pytest.importorskip("torch")

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, LambdaLR, StepLR

import trainlattice
from trainlattice.builtin_plugins.checkpoint_local.config import configs_compatible
from helpers import load_config
from trainlattice.contracts.config import DatasetConfig, ExperimentConfig, ExperimentSection, ModelConfig, OptimizerConfig, SchedulerConfig, TrainerConfig
from trainlattice.errors import ConfigError, TrainLatticeError
from trainlattice.contracts.callbacks import Callback
from helpers import build_scheduler, validate_scheduler_config
from helpers import run_experiment


BASE_CONFIG = """
experiment:
  name: unit-test
model:
  name: mlp
  input_dim: 4
dataset:
  name: synthetic
  batch_size: 2
  input_shape: [4]
optimizer:
  lr: 0.01
trainer:
  epochs: 2
"""

SCHEDULER_SECTION = """
scheduler:
  name: warmup_cosine
  warmup_steps: 2
  total_steps: 6
  min_lr: 0.1
"""


def write_config(tmp_path: Path, text: str = BASE_CONFIG) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _optimizer(lr: float = 1.0) -> Optimizer:
    import torch

    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(2))], lr=lr)


def _learning_rates(optimizer: Optimizer, scheduler: LRScheduler, steps: int) -> list[float]:
    """Collect the learning rate at t=0 and after each scheduler step."""
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(steps):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    return rates


def test_services_contain_the_builtin_schedulers() -> None:
    from helpers import get_component
    assert tuple(get_component("scheduler", name).name for name in ("step_decay", "warmup_cosine", "warmup_linear")) == ("step_decay", "warmup_cosine", "warmup_linear")


def test_scheduler_section_defaults_to_constant_learning_rate(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))

    assert config.scheduler is None
    assert "scheduler" not in config.to_dict()


def test_loads_scheduler_section_and_round_trips(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, BASE_CONFIG + SCHEDULER_SECTION))

    assert config.scheduler == SchedulerConfig(
        "warmup_cosine", {"warmup_steps": 2, "total_steps": 6, "min_lr": 0.1}
    )
    assert config.to_dict()["scheduler"] == {
        "name": "warmup_cosine",
        "warmup_steps": 2,
        "total_steps": 6,
        "min_lr": 0.1,
    }


def test_explicit_null_scheduler_section_means_constant_learning_rate(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, BASE_CONFIG + "scheduler:\n"))

    assert config.scheduler is None


def test_rejects_invalid_scheduler_sections(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="scheduler must be a mapping"):
        load_config(write_config(tmp_path, BASE_CONFIG + "scheduler: [warmup_cosine]\n"))
    with pytest.raises(ConfigError, match="scheduler.name must be a string"):
        load_config(write_config(tmp_path, BASE_CONFIG + "scheduler:\n  name: [x]\n"))
    with pytest.raises(ConfigError, match="scheduler.name must be a non-empty string"):
        load_config(write_config(tmp_path, BASE_CONFIG + "scheduler:\n  warmup_steps: 2\n"))


def test_rejects_unknown_and_invalid_options() -> None:
    with pytest.raises(
        TrainLatticeError, match=r"service unavailable: trainlattice.scheduler/cosine"
    ):
        validate_scheduler_config(SchedulerConfig("cosine"))
    with pytest.raises(ConfigError, match="unknown configuration field\\(s\\): scheduler.warmupstep"):
        validate_scheduler_config(SchedulerConfig("warmup_cosine", {"warmupstep": 3}))
    with pytest.raises(ConfigError, match="unknown configuration field\\(s\\): scheduler.total_steps"):
        validate_scheduler_config(SchedulerConfig("step_decay", {"step_size": 2, "total_steps": 10}))


def test_rejects_invalid_warmup_options() -> None:
    with pytest.raises(ConfigError, match="scheduler.warmup_steps must be a non-negative integer"):
        validate_scheduler_config(SchedulerConfig("warmup_cosine", {"warmup_steps": -1}))
    with pytest.raises(ConfigError, match="scheduler.warmup_steps must be a non-negative integer"):
        validate_scheduler_config(SchedulerConfig("warmup_cosine", {"warmup_steps": 1.5}))
    with pytest.raises(ConfigError, match="scheduler.total_steps must be a positive integer"):
        validate_scheduler_config(SchedulerConfig("warmup_cosine", {"total_steps": 0}))
    with pytest.raises(ConfigError, match="scheduler.total_steps must be a positive integer"):
        validate_scheduler_config(SchedulerConfig("warmup_linear", {"total_steps": "many"}))
    with pytest.raises(ConfigError, match="scheduler.min_lr must be a non-negative number"):
        validate_scheduler_config(SchedulerConfig("warmup_cosine", {"min_lr": -0.1}))


def test_rejects_warmup_that_leaves_no_decay_steps() -> None:
    with pytest.raises(ConfigError, match="scheduler.warmup_steps must be less than scheduler.total_steps"):
        validate_scheduler_config(
            SchedulerConfig("warmup_cosine", {"warmup_steps": 5, "total_steps": 5, "min_lr": 0.1})
        )
    with pytest.raises(ConfigError, match="scheduler.warmup_steps must be less than scheduler.total_steps"):
        build_scheduler(SchedulerConfig("warmup_linear", {"warmup_steps": 5}), _optimizer(), total_steps=3)


def test_rejects_invalid_step_decay_options() -> None:
    with pytest.raises(ConfigError, match="scheduler.step_size is required"):
        validate_scheduler_config(SchedulerConfig("step_decay"))
    with pytest.raises(ConfigError, match="scheduler.step_size must be a positive integer"):
        validate_scheduler_config(SchedulerConfig("step_decay", {"step_size": 0}))
    with pytest.raises(ConfigError, match="scheduler.gamma must be a positive number"):
        validate_scheduler_config(SchedulerConfig("step_decay", {"step_size": 2, "gamma": 0}))
    with pytest.raises(ConfigError, match="scheduler.gamma must be a positive number"):
        validate_scheduler_config(SchedulerConfig("step_decay", {"step_size": 2, "gamma": -0.5}))


def test_warmup_cosine_learning_rate_sequence() -> None:
    scheduler = build_scheduler(
        SchedulerConfig("warmup_cosine", {"warmup_steps": 2, "total_steps": 6, "min_lr": 0.1}), _optimizer(lr=1.0)
    )

    assert isinstance(scheduler, LambdaLR)
    assert _learning_rates(scheduler.optimizer, scheduler, steps=7) == pytest.approx(
        [
            0.0,
            0.5,
            1.0,
            0.1 + 0.9 * (1 + cos(pi / 4)) / 2,
            0.1 + 0.9 * (1 + cos(pi / 2)) / 2,
            0.1 + 0.9 * (1 + cos(3 * pi / 4)) / 2,
            0.1,
            0.1,
        ]
    )


def test_warmup_linear_learning_rate_sequence() -> None:
    scheduler = build_scheduler(
        SchedulerConfig("warmup_linear", {"total_steps": 4}), _optimizer(lr=1.0)
    )

    assert _learning_rates(scheduler.optimizer, scheduler, steps=5) == pytest.approx(
        [1.0, 0.75, 0.5, 0.25, 0.0, 0.0]
    )


def test_min_lr_is_an_absolute_floor() -> None:
    scheduler = build_scheduler(
        SchedulerConfig("warmup_linear", {"warmup_steps": 1, "total_steps": 3, "min_lr": 0.1}),
        _optimizer(lr=0.4),
    )

    assert _learning_rates(scheduler.optimizer, scheduler, steps=4) == pytest.approx(
        [0.0, 0.4, 0.25, 0.1, 0.1]
    )


def test_step_decay_learning_rate_sequence() -> None:
    scheduler = build_scheduler(
        SchedulerConfig("step_decay", {"step_size": 2, "gamma": 0.5}), _optimizer(lr=0.1)
    )

    assert isinstance(scheduler, StepLR)
    assert _learning_rates(scheduler.optimizer, scheduler, steps=5) == pytest.approx(
        [0.1, 0.1, 0.05, 0.05, 0.025, 0.025]
    )


def test_total_steps_resolves_from_options_then_runtime_argument() -> None:
    from_options = build_scheduler(
        SchedulerConfig("warmup_linear", {"total_steps": 8}), _optimizer(lr=1.0), total_steps=100
    )
    assert _learning_rates(from_options.optimizer, from_options, steps=4)[-1] == pytest.approx(0.5)

    from_runtime = build_scheduler(SchedulerConfig("warmup_linear"), _optimizer(lr=1.0), total_steps=4)
    assert _learning_rates(from_runtime.optimizer, from_runtime, steps=4)[-1] == pytest.approx(0.0)


def test_requires_total_steps_for_warmup_schedulers() -> None:
    with pytest.raises(
        ConfigError, match="scheduler.total_steps is required by warmup_cosine; set it in the scheduler section"
    ):
        build_scheduler(SchedulerConfig("warmup_cosine"), _optimizer())
    with pytest.raises(ConfigError, match="scheduler.total_steps must be a positive integer"):
        build_scheduler(SchedulerConfig("warmup_cosine"), _optimizer(), total_steps=0)
    with pytest.raises(ConfigError, match="scheduler.total_steps must be a positive integer"):
        build_scheduler(SchedulerConfig("warmup_cosine"), _optimizer(), total_steps=6.5)


def test_scheduler_section_is_recovery_sensitive(tmp_path: Path) -> None:
    without_scheduler = load_config(write_config(tmp_path))
    with_scheduler = load_config(write_config(tmp_path, BASE_CONFIG + SCHEDULER_SECTION))

    assert configs_compatible(without_scheduler, load_config(write_config(tmp_path)))
    assert not configs_compatible(without_scheduler, with_scheduler)


def runtime_config(tmp_path: Path, *, epochs: int, scheduler: SchedulerConfig | None) -> ExperimentConfig:
    """A minimal runnable configuration with an optional scheduler section."""
    return ExperimentConfig(
        experiment=ExperimentSection(name="scheduler-run", seed=3, runs_dir=str(tmp_path / "runs")),
        model=ModelConfig("mlp", {"input_dim": 4, "hidden_dims": [8], "num_classes": 2}),
        dataset=DatasetConfig(
            name="synthetic",
            batch_size=4,
            options={"samples": 12, "val_samples": 8, "input_shape": [4], "num_classes": 2},
        ),
        optimizer=OptimizerConfig(lr=0.05),
        trainer=TrainerConfig(epochs=epochs, device="cpu", validate_every=1, log_every=1),
        scheduler=scheduler,
    )


def _metric_records(run_dir: Path, phase: str) -> list[dict]:
    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    return [record for record in records if record["phase"] == phase]


def test_run_logs_warmup_cosine_learning_rate_sequence(tmp_path: Path) -> None:
    # 3 batches per epoch over 2 epochs derive total_steps=6; the warmup ends at step 2
    result = run_experiment(
        runtime_config(
            tmp_path, epochs=2, scheduler=SchedulerConfig("warmup_cosine", {"warmup_steps": 2, "min_lr": 0.005})
        ),
        command="pytest scheduler",
    )

    batches = _metric_records(result.run_dir, "train")
    epochs = _metric_records(result.run_dir, "epoch")
    base, floor = 0.05, 0.005
    span = 6 - 2
    used = [
        base * 0 / 2,
        base * 1 / 2,
        floor + (base - floor) * (1 + cos(pi * 0 / span)) / 2,
        floor + (base - floor) * (1 + cos(pi * 1 / span)) / 2,
        floor + (base - floor) * (1 + cos(pi * 2 / span)) / 2,
        floor + (base - floor) * (1 + cos(pi * 3 / span)) / 2,
    ]
    assert [record["step"] for record in batches] == [1, 2, 3, 4, 5, 6]
    assert [record["lr"] for record in batches] == pytest.approx(used)
    # the epoch record carries the rate waiting for the next optimizer step
    assert [record["lr"] for record in epochs] == pytest.approx([used[3], floor])


def test_resume_rejects_modified_scheduler_section(tmp_path: Path) -> None:
    first = run_experiment(
        runtime_config(
            tmp_path, epochs=1, scheduler=SchedulerConfig("warmup_linear", {"warmup_steps": 1, "min_lr": 0.001})
        ),
        command="pytest scheduler",
    )

    changed = runtime_config(
        tmp_path, epochs=2, scheduler=SchedulerConfig("warmup_linear", {"warmup_steps": 2, "min_lr": 0.001})
    )
    with pytest.raises(TrainLatticeError, match="incompatible"):
        run_experiment(changed, command="pytest scheduler resume", resume=first.run_dir)


def test_resume_rejects_adding_a_scheduler_to_a_schedulerless_checkpoint(tmp_path: Path) -> None:
    first = run_experiment(
        runtime_config(tmp_path, epochs=1, scheduler=None), command="pytest schedulerless"
    )

    with pytest.raises(TrainLatticeError, match="plugin graph.*mismatch"):
        run_experiment(
            runtime_config(
                tmp_path, epochs=2, scheduler=SchedulerConfig("step_decay", {"step_size": 2})
            ),
            command="pytest scheduler resume",
            resume=first.run_dir,
        )


def test_min_lr_above_the_base_rate_is_rejected() -> None:
    with pytest.raises(ConfigError, match="scheduler.min_lr must not exceed optimizer.lr"):
        build_scheduler(
            SchedulerConfig("warmup_cosine", {"warmup_steps": 1, "total_steps": 6, "min_lr": 2.0}),
            _optimizer(lr=1.0),
        )
    with pytest.raises(ConfigError, match="scheduler.min_lr must not exceed optimizer.lr"):
        build_scheduler(
            SchedulerConfig("warmup_linear", {"total_steps": 6, "min_lr": 0.5}), _optimizer(lr=0.05)
        )


def derived_budget_config(tmp_path: Path, *, epochs: int, total_steps: int | None = None) -> ExperimentConfig:
    """A warmup-cosine run whose step budget derives from the epochs unless pinned."""
    options = {"warmup_steps": 1, "min_lr": 0.005}
    if total_steps is not None:
        options["total_steps"] = total_steps
    return runtime_config(tmp_path, epochs=epochs, scheduler=SchedulerConfig("warmup_cosine", options))


def logged_rates(result) -> dict[int, float]:
    return {record["step"]: record["lr"] for record in _metric_records(result.run_dir, "train")}


def test_derived_budget_resume_anchors_to_the_checkpointed_schedule(tmp_path: Path) -> None:
    # stage one spans a derived budget of 3 steps (3 batches x 1 epoch); the
    # resume asks for 2 epochs (derived 6) but must continue the schedule it
    # started with — equal to an uninterrupted run pinned to budget 3
    import torch

    first = run_experiment(derived_budget_config(tmp_path, epochs=1), command="pytest first stage")
    payload = torch.load(first.run_dir / "checkpoints" / "last.pt", weights_only=False)
    assert payload["scheduler_budget"] == 3

    resumed = run_experiment(
        derived_budget_config(tmp_path, epochs=2), command="pytest resume", resume=first.run_dir
    )
    pinned = run_experiment(
        derived_budget_config(tmp_path, epochs=2, total_steps=3), command="pytest reference"
    )

    resumed_rates = logged_rates(resumed)
    assert sorted(resumed_rates) == [4, 5, 6]
    # the budget-3 schedule saturates at min_lr from step 4 on
    assert [resumed_rates[step] for step in (4, 5, 6)] == pytest.approx([0.005, 0.005, 0.005])
    pinned_tail = {step: rate for step, rate in logged_rates(pinned).items() if step >= 4}
    assert resumed_rates == pinned_tail


class StopAfterFirstEpoch(Callback):
    """Interrupt a run after its first epoch, as a crash mid-training would."""

    def on_epoch_end(self, trainer) -> None:
        trainer.request_stop("interrupted for the test")


def test_derived_budget_resume_of_an_interrupted_run_matches_uninterrupted(tmp_path: Path) -> None:
    # both stages configured for 2 epochs, so the derived budget is 6 either
    # way; the interrupted run resumes onto the very schedule it started
    uninterrupted = run_experiment(derived_budget_config(tmp_path, epochs=2), command="pytest uninterrupted")
    interrupted = run_experiment(
        derived_budget_config(tmp_path, epochs=2),
        command="pytest interrupted",
        extra_callbacks=[StopAfterFirstEpoch()],
    )
    assert interrupted.state.global_step == 3

    resumed = run_experiment(
        derived_budget_config(tmp_path, epochs=2), command="pytest resume", resume=interrupted.run_dir
    )

    assert resumed.state.global_step == uninterrupted.state.global_step == 6
    # the resumed run directory only logs its own steps; compare the overlap
    expected_tail = {step: rate for step, rate in logged_rates(uninterrupted).items() if step >= 4}
    assert logged_rates(resumed) == expected_tail
