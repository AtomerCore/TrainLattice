"""Loss registry, task declaration, and regression training behavior."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset

from trainlattice.config.schema import (
    DatasetConfig,
    ExperimentConfig,
    ExperimentSection,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig,
)
from trainlattice.errors import ConfigError, RegistryError
from trainlattice.runtime.runner import run_experiment
from trainlattice.trainer.losses import (
    LOSSES,
    build_loss,
    loss_task,
    register_loss,
    validate_loss_config,
)
from trainlattice.config import ComponentContract


def test_registry_declares_tasks_for_builtin_losses() -> None:
    assert LOSSES.names() == ("cross_entropy", "l1", "mse")
    assert loss_task("cross_entropy") == "classification"
    assert loss_task("mse") == "regression"
    assert loss_task("l1") == "regression"


def test_builds_and_validates_losses() -> None:
    assert isinstance(build_loss(LossConfig()), nn.CrossEntropyLoss)
    assert isinstance(build_loss(LossConfig("mse")), nn.MSELoss)

    with pytest.raises(ConfigError, match=r"unsupported loss.name 'huber'; available: cross_entropy, l1, mse"):
        validate_loss_config(LossConfig("huber"))
    with pytest.raises(ConfigError, match="loss.label_smoothingx"):
        validate_loss_config(LossConfig("cross_entropy", {"label_smoothingx": 0.1}))
    with pytest.raises(ConfigError, match="loss.label_smoothing must be a number"):
        validate_loss_config(LossConfig("cross_entropy", {"label_smoothing": 2}))
    with pytest.raises(ConfigError, match="loss.reduction must be one of"):
        validate_loss_config(LossConfig("mse", {"reduction": "avg"}))


def test_register_loss_rejects_unknown_task_kinds() -> None:
    with pytest.raises(RegistryError, match="loss task must be one of"):
        register_loss("huber", nn.SmoothL1Loss, task="autoencoding")


def test_register_loss_is_importable_from_the_trainer_package() -> None:
    from trainlattice.trainer import register_loss as package_entry

    assert package_entry is register_loss


def test_trainer_rejects_unknown_task_kinds() -> None:
    from trainlattice.trainer.trainer import Trainer

    model = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="task must be one of"):
        Trainer(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            loss_fn=nn.MSELoss(),
            device=torch.device("cpu"),
            epochs=1,
            task="autoencoding",
        )


def test_trainer_aggregates_reductions_as_per_sample_means() -> None:
    from torch.utils.data import DataLoader
    from trainlattice.trainer.trainer import Trainer

    inputs = torch.linspace(-1.0, 1.0, 6).unsqueeze(1)
    loader = DataLoader(TensorDataset(inputs, inputs * 0.5), batch_size=4)

    def run_with(reduction: str) -> dict[str, float]:
        torch.manual_seed(0)
        model = nn.Linear(1, 1)
        trainer = Trainer(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
            loss_fn=nn.MSELoss(reduction=reduction),
            device=torch.device("cpu"),
            epochs=1,
        )
        return trainer.train_epoch(loader)

    mean_metrics = run_with("mean")
    assert run_with("sum") == pytest.approx(mean_metrics)
    assert run_with("none") == pytest.approx(mean_metrics)


def build_sine(
    *,
    train: bool,
    split: str | None = None,
    samples: int = 16,
    val_samples: int = 8,
    seed: int | None = None,
    **_: Any,
) -> Dataset[Any]:
    """One-dimensional sine regression with float targets shaped like the outputs."""
    split_name = split or ("train" if train else "validation")
    count = {"train": samples, "validation": val_samples}[split_name]
    offset = {"train": 0, "validation": 1}[split_name]
    generator = torch.Generator().manual_seed((seed if seed is not None else 42) + offset)
    inputs = torch.rand(count, 1, generator=generator) * 2 * math.pi
    targets = torch.sin(inputs)
    return TensorDataset(inputs, targets)


def _validate_sine(path: str, options: Mapping[str, Any]) -> None:
    unknown = sorted(set(options) - {"samples", "val_samples"})
    if unknown:
        keys = ", ".join(f"{path}.{key}" for key in unknown)
        raise ConfigError(f"unknown configuration field(s): {keys}")


def _sine_contract(options: Mapping[str, Any]) -> ComponentContract:
    return ComponentContract(input_dim=1, num_classes=1, input_shape=(1,))


def regression_config(tmp_path: Path, epochs: int = 1) -> ExperimentConfig:
    return ExperimentConfig(
        experiment=ExperimentSection(name="sine", seed=3, runs_dir=str(tmp_path / "runs")),
        model=ModelConfig("mlp", {"input_dim": 1, "hidden_dims": [8], "num_classes": 1}),
        dataset=DatasetConfig("sine", batch_size=4, options={"samples": 16, "val_samples": 8}),
        optimizer=OptimizerConfig("sgd", lr=0.05),
        trainer=TrainerConfig(epochs=epochs, device="cpu", validate_every=1, log_every=100),
        loss=LossConfig("mse"),
    )


def test_regression_run_reports_loss_only_and_resumes(tmp_path: Path) -> None:
    from trainlattice.datasets import DATASETS

    DATASETS.register("sine", build_sine, validate=_validate_sine, contract=_sine_contract)
    first = run_experiment(regression_config(tmp_path), command="pytest sine")

    assert first.state.status == "finished"
    assert set(first.state.latest_metrics) == {"train_loss", "val_loss"}
    manifest = json.loads((first.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "torch.nn.modules.loss" in manifest["loss_source"]
    metrics_lines = (first.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    epoch_records = [json.loads(line) for line in metrics_lines if json.loads(line)["phase"] == "epoch"]
    assert all("accuracy" not in record for record in epoch_records)

    resumed = run_experiment(regression_config(tmp_path, epochs=2), command="pytest sine resume", resume=first.run_dir)
    assert resumed.state.next_epoch == 2


def test_regression_run_accepts_sum_and_none_reductions(tmp_path: Path) -> None:
    from trainlattice.datasets import DATASETS

    DATASETS.register("sine", build_sine, validate=_validate_sine, contract=_sine_contract)
    for reduction in ("sum", "none"):
        config = regression_config(tmp_path)
        config.loss = LossConfig("mse", {"reduction": reduction})
        result = run_experiment(config, command=f"pytest mse {reduction}")
        assert result.state.status == "finished"
        assert math.isfinite(result.state.latest_metrics["val_loss"])


def test_accuracy_monitor_requires_classification_loss(tmp_path: Path) -> None:
    config = regression_config(tmp_path)
    config.trainer.monitor = "val_accuracy"
    config.dataset = DatasetConfig(
        name="synthetic",
        batch_size=4,
        options={"samples": 12, "val_samples": 6, "input_shape": [1], "num_classes": 1},
    )

    with pytest.raises(ConfigError, match="requires a classification loss"):
        run_experiment(config, command="pytest invalid")
    assert not Path(config.experiment.runs_dir).exists()
