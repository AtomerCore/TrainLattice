"""Throughput and memory metrics carried by batch and epoch records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

from trainlattice.config.schema import (
    DatasetConfig,
    ExperimentConfig,
    ExperimentSection,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig,
)
from trainlattice.runtime.runner import run_experiment

# wall-clock and memory fields vary between runs and are excluded from equality checks
TIMING_KEYS = {"steps_per_sec", "samples_per_sec", "epoch_seconds", "data_seconds", "train_samples_per_sec"}


def throughput_config(tmp_path: Path, *, epochs: int = 2) -> ExperimentConfig:
    """A minimal cpu configuration whose batch records arrive at every optimizer step."""
    return ExperimentConfig(
        experiment=ExperimentSection(name="throughput", seed=3, runs_dir=str(tmp_path / "runs")),
        model=ModelConfig("mlp", {"input_dim": 4, "hidden_dims": [8], "num_classes": 2}),
        dataset=DatasetConfig(
            name="synthetic",
            batch_size=4,
            options={"samples": 12, "val_samples": 8, "input_shape": [4], "num_classes": 2},
        ),
        optimizer=OptimizerConfig(lr=0.05),
        trainer=TrainerConfig(epochs=epochs, device="cpu", validate_every=1, log_every=1),
    )


def metric_records(run_dir: Path, phase: str) -> list[dict[str, Any]]:
    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    return [record for record in records if record["phase"] == phase]


def without_timing(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in TIMING_KEYS}


def test_batch_records_carry_cumulative_throughput_rates(tmp_path: Path) -> None:
    result = run_experiment(throughput_config(tmp_path), command="pytest throughput")

    batches = metric_records(result.run_dir, "train")
    assert [record["step"] for record in batches] == [1, 2, 3, 4, 5, 6]
    for record in batches:
        assert record["steps_per_sec"] > 0
        assert record["samples_per_sec"] > 0
        # each optimizer step consumes one batch of 4 samples, so the cumulative
        # averages keep the batch size as their ratio
        assert record["samples_per_sec"] == pytest.approx(record["steps_per_sec"] * 4)


def test_epoch_records_carry_bounded_epoch_timing(tmp_path: Path) -> None:
    result = run_experiment(throughput_config(tmp_path), command="pytest throughput")

    epochs = metric_records(result.run_dir, "epoch")
    assert [record["epoch"] for record in epochs] == [0, 1]
    for record in epochs:
        assert record["epoch_seconds"] > 0
        assert 0 <= record["data_seconds"] <= record["epoch_seconds"]
        assert record["train_samples_per_sec"] > 0


def test_cpu_records_omit_the_memory_key(tmp_path: Path) -> None:
    result = run_experiment(throughput_config(tmp_path, epochs=1), command="pytest throughput")

    records = metric_records(result.run_dir, "train") + metric_records(result.run_dir, "epoch")
    assert records
    assert all("max_memory_mb" not in record for record in records)


def test_two_runs_of_one_configuration_agree_except_timing(tmp_path: Path) -> None:
    first = run_experiment(throughput_config(tmp_path), command="pytest throughput first")
    second = run_experiment(throughput_config(tmp_path), command="pytest throughput second")

    for phase in ("train", "epoch"):
        first_records = [without_timing(record) for record in metric_records(first.run_dir, phase)]
        second_records = [without_timing(record) for record in metric_records(second.run_dir, phase)]
        assert first_records
        assert first_records == second_records
    assert first.state.latest_metrics["train_loss"] == second.state.latest_metrics["train_loss"]
    assert first.state.latest_metrics["val_accuracy"] == second.state.latest_metrics["val_accuracy"]
