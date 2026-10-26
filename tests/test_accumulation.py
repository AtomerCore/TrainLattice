"""Gradient-accumulation windows: equivalence, step counting, records, and resume."""

from __future__ import annotations

import json
from math import cos, pi
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trainlattice.config.schema import (
    DatasetConfig,
    ExperimentConfig,
    ExperimentSection,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainerConfig,
)
from trainlattice.errors import TrainLatticeError
from trainlattice.runtime.runner import run_experiment
from trainlattice.trainer import Trainer


def test_accumulated_window_matches_the_full_batch_step() -> None:
    """Two equal micro-batches under accumulate=2 equal one batch_size=8 step."""
    generator = torch.Generator().manual_seed(11)
    inputs = torch.randn(8, 4, generator=generator)
    targets = torch.randint(0, 2, (8,), generator=generator)
    full_loader = DataLoader(TensorDataset(inputs, targets), batch_size=8)
    micro_loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)

    torch.manual_seed(0)
    full_model = nn.Linear(4, 2)
    full_run = Trainer(
        model=full_model,
        optimizer=torch.optim.SGD(full_model.parameters(), lr=0.1),
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        epochs=1,
    )
    full_run.train_epoch(full_loader)

    torch.manual_seed(0)
    accumulated_model = nn.Linear(4, 2)
    accumulated_run = Trainer(
        model=accumulated_model,
        optimizer=torch.optim.SGD(accumulated_model.parameters(), lr=0.1),
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        epochs=1,
        accumulate=2,
    )
    accumulated_run.train_epoch(micro_loader)

    assert accumulated_run.state.global_step == 1
    for expected, actual in zip(full_model.parameters(), accumulated_model.parameters()):
        assert torch.allclose(expected, actual)
    # the window aggregate is the sample-weighted mean over the whole window
    assert accumulated_run.batch_metrics["loss"] == pytest.approx(full_run.batch_metrics["loss"])


def test_partial_window_at_epoch_end_flushes_one_step() -> None:
    """Seven micro-batches under accumulate=3 take ceil(7/3) = 3 optimizer steps."""
    generator = torch.Generator().manual_seed(3)
    inputs = torch.randn(7, 4, generator=generator)
    targets = torch.randint(0, 2, (7,), generator=generator)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=1)
    model = nn.Linear(4, 2)
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        epochs=1,
        accumulate=3,
    )

    trainer.train_epoch(loader)

    assert trainer.state.global_step == 3


def accumulation_config(
    tmp_path: Path, *, epochs: int, accumulate: int, total_steps: int | None = None
) -> ExperimentConfig:
    """A runnable configuration with a warmup-cosine scheduler under accumulation.

    ``batch_size=2`` over 12 samples yields 6 micro-batches per epoch;
    ``accumulate=2`` turns those into 3 optimizer steps per epoch, so two
    epochs derive ``total_steps=6``. Resume stages pin ``total_steps``
    explicitly because ``trainer.epochs`` is runtime-only and would otherwise
    shift the derived budget between stages.
    """
    options = {"warmup_steps": 2, "min_lr": 0.005}
    if total_steps is not None:
        options["total_steps"] = total_steps
    return ExperimentConfig(
        experiment=ExperimentSection(name="accumulate", seed=3, runs_dir=str(tmp_path / "runs")),
        model=ModelConfig("mlp", {"input_dim": 4, "hidden_dims": [8], "num_classes": 2}),
        dataset=DatasetConfig(
            name="synthetic",
            batch_size=2,
            options={"samples": 12, "val_samples": 8, "input_shape": [4], "num_classes": 2},
        ),
        optimizer=OptimizerConfig(lr=0.05),
        trainer=TrainerConfig(
            epochs=epochs, device="cpu", validate_every=1, log_every=1, accumulate=accumulate
        ),
        scheduler=SchedulerConfig("warmup_cosine", options),
    )


def _metric_records(run_dir: Path, phase: str) -> list[dict]:
    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    return [record for record in records if record["phase"] == phase]


def test_batch_records_arrive_once_per_optimizer_step_with_window_lr(tmp_path: Path) -> None:
    result = run_experiment(accumulation_config(tmp_path, epochs=2, accumulate=2), command="pytest accumulate")

    batches = _metric_records(result.run_dir, "train")
    epochs = _metric_records(result.run_dir, "epoch")
    assert [record["step"] for record in batches] == [1, 2, 3, 4, 5, 6]
    for record in batches:
        assert {"loss", "accuracy", "lr"} <= set(record)
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
    assert [record["lr"] for record in batches] == pytest.approx(used)
    # the epoch record carries the rate waiting for the next optimizer step
    assert [record["lr"] for record in epochs] == pytest.approx([used[3], floor])
    assert result.state.global_step == 6


def test_resume_rejects_a_changed_accumulate(tmp_path: Path) -> None:
    first = run_experiment(
        accumulation_config(tmp_path, epochs=1, accumulate=2, total_steps=6), command="pytest accumulate"
    )

    with pytest.raises(TrainLatticeError, match="incompatible"):
        run_experiment(
            accumulation_config(tmp_path, epochs=2, accumulate=1, total_steps=6),
            command="pytest accumulate resume",
            resume=first.run_dir,
        )


def test_resume_with_matching_accumulate_reproduces_the_uninterrupted_run(tmp_path: Path) -> None:
    uninterrupted = run_experiment(
        accumulation_config(tmp_path, epochs=2, accumulate=2, total_steps=6), command="pytest uninterrupted"
    )
    first = run_experiment(
        accumulation_config(tmp_path, epochs=1, accumulate=2, total_steps=6), command="pytest first stage"
    )
    resumed = run_experiment(
        accumulation_config(tmp_path, epochs=2, accumulate=2, total_steps=6),
        command="pytest resume",
        resume=first.run_dir,
    )

    assert resumed.state.global_step == uninterrupted.state.global_step == 6
    expected = torch.load(uninterrupted.run_dir / "checkpoints" / "last.pt", weights_only=False)["model"]
    actual = torch.load(resumed.run_dir / "checkpoints" / "last.pt", weights_only=False)["model"]
    for key in expected:
        assert torch.equal(expected[key], actual[key])
