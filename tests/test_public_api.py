"""Public package interface behavior for custom components."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, TensorDataset

import trainlattice
from trainlattice import (
    ARCHITECTURES,
    DATASETS,
    Callback,
    ComponentContract,
    ConfigError,
    run_experiment,
    run_from_file,
)
from trainlattice.config.schema import (
    DatasetConfig,
    ExperimentConfig,
    ExperimentSection,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig,
)
from trainlattice.trainer.trainer import Trainer


class TinyNet(nn.Module):
    """Single linear layer sized for tests."""

    def __init__(self, *, input_dim: int = 4, num_classes: int = 2) -> None:
        super().__init__()
        self.network = nn.Linear(input_dim, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


def _validate_tiny_net(path: str, options: Mapping[str, Any]) -> None:
    unknown = sorted(set(options) - {"input_dim", "num_classes"})
    if unknown:
        keys = ", ".join(f"{path}.{key}" for key in unknown)
        raise ConfigError(f"unknown configuration field(s): {keys}")


def _tiny_net_contract(options: Mapping[str, Any]) -> ComponentContract:
    return ComponentContract(
        input_dim=options.get("input_dim", 4),
        num_classes=options.get("num_classes", 2),
    )


def build_tiny_blobs(
    *,
    train: bool,
    split: str | None = None,
    input_dim: int = 4,
    samples: int = 12,
    val_samples: int = 6,
    seed: int | None = None,
    **_: Any,
) -> Dataset[Any]:
    split_name = split or ("train" if train else "validation")
    count = {"train": samples, "validation": val_samples}[split_name]
    offset = {"train": 0, "validation": 1}[split_name]
    generator = torch.Generator().manual_seed((seed if seed is not None else 42) + offset)
    center = torch.ones(input_dim)
    labels = torch.randint(0, 2, (count,), generator=generator)
    side = (labels * 2 - 1).unsqueeze(1)
    features = torch.randn(count, input_dim, generator=generator) * 0.3 + side * center
    return TensorDataset(features, labels.long())


def _validate_tiny_blobs(path: str, options: Mapping[str, Any]) -> None:
    unknown = sorted(set(options) - {"input_dim", "samples", "val_samples"})
    if unknown:
        keys = ", ".join(f"{path}.{key}" for key in unknown)
        raise ConfigError(f"unknown configuration field(s): {keys}")


def _tiny_blobs_contract(options: Mapping[str, Any]) -> ComponentContract:
    input_dim = options.get("input_dim", 4)
    return ComponentContract(input_dim=input_dim, num_classes=2, input_shape=(input_dim,))


def _register_custom_components() -> None:
    ARCHITECTURES.register("tiny_net", TinyNet, validate=_validate_tiny_net, contract=_tiny_net_contract)
    DATASETS.register(
        "tiny_blobs",
        build_tiny_blobs,
        validate=_validate_tiny_blobs,
        contract=_tiny_blobs_contract,
    )


def custom_config(tmp_path: Path, epochs: int = 1) -> ExperimentConfig:
    return ExperimentConfig(
        experiment=ExperimentSection(name="custom-api", seed=3, runs_dir=str(tmp_path / "runs")),
        model=ModelConfig("tiny_net", {"input_dim": 4, "num_classes": 2}),
        dataset=DatasetConfig(
            name="tiny_blobs",
            batch_size=4,
            options={"input_dim": 4, "samples": 12, "val_samples": 6},
        ),
        optimizer=OptimizerConfig(lr=0.05),
        trainer=TrainerConfig(epochs=epochs, device="cpu", validate_every=1, log_every=100),
    )


def test_package_root_exports_the_entry_points() -> None:
    from trainlattice import datasets, modeling
    from trainlattice.runtime import run_experiment as runner_entry

    assert trainlattice.ARCHITECTURES is modeling.ARCHITECTURES
    assert trainlattice.DATASETS is datasets.DATASETS
    assert trainlattice.run_experiment is runner_entry
    assert callable(trainlattice.run_from_file)
    assert trainlattice.Callback is Callback
    for name in trainlattice.__all__:
        assert hasattr(trainlattice, name)


class EventRecorder(Callback):
    """Record lifecycle hook calls with their epoch indexes."""

    def __init__(self) -> None:
        self.events: list[tuple[str, int | None]] = []

    def on_train_start(self, trainer: Trainer) -> None:
        self.events.append(("train_start", None))

    def on_epoch_end(self, trainer: Trainer) -> None:
        self.events.append(("epoch_end", trainer.state.epoch))

    def on_train_end(self, trainer: Trainer) -> None:
        self.events.append(("train_end", None))


def test_extra_callbacks_observe_the_lifecycle_between_builtins(tmp_path: Path) -> None:
    recorder = EventRecorder()

    result = run_experiment(
        ExperimentConfig(
            experiment=ExperimentSection(name="callbacks", seed=3, runs_dir=str(tmp_path / "runs")),
            model=ModelConfig("mlp", {"input_dim": 4, "hidden_dims": [8], "num_classes": 2}),
            dataset=DatasetConfig(
                name="synthetic",
                batch_size=4,
                options={"samples": 12, "val_samples": 6, "input_shape": [4], "num_classes": 2},
            ),
            optimizer=OptimizerConfig(lr=0.05),
            trainer=TrainerConfig(epochs=2, device="cpu", validate_every=1, log_every=100),
        ),
        command="pytest callbacks",
        extra_callbacks=[recorder],
    )

    assert result.state.status == "finished"
    assert recorder.events == [
        ("train_start", None),
        ("epoch_end", 0),
        ("epoch_end", 1),
        ("train_end", None),
    ]
    # the injected callback runs alongside the built-in logging and checkpoint outputs
    assert (result.run_dir / "metrics.jsonl").is_file()
    assert (result.run_dir / "checkpoints" / "last.pt").is_file()


def test_run_from_file_passes_extra_callbacks_through(tmp_path: Path) -> None:
    recorder = EventRecorder()
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: callbacks-file",
                "  seed: 3",
                f"  runs_dir: {tmp_path / 'runs'}",
                "model:",
                "  name: mlp",
                "  input_dim: 4",
                "  hidden_dims: [8]",
                "  num_classes: 2",
                "dataset:",
                "  name: synthetic",
                "  batch_size: 4",
                "  samples: 12",
                "  val_samples: 6",
                "  input_shape: [4]",
                "  num_classes: 2",
                "optimizer:",
                "  lr: 0.05",
                "trainer:",
                "  epochs: 1",
                "  device: cpu",
            ]
        ),
        encoding="utf-8",
    )

    result = run_from_file(config_file, command="pytest file", extra_callbacks=[recorder])

    assert result.state.status == "finished"
    assert recorder.events == [("train_start", None), ("epoch_end", 0), ("train_end", None)]


def test_custom_components_run_end_to_end_and_resume(tmp_path: Path) -> None:
    _register_custom_components()
    first = run_experiment(custom_config(tmp_path), command="pytest custom")

    assert first.state.status == "finished"
    manifest = json.loads((first.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"] == "tiny_net"
    assert manifest["dataset"] == "tiny_blobs"
    assert manifest["model_source"].startswith(f"{__name__} (")
    assert manifest["dataset_source"].startswith(f"{__name__} (")

    resumed = run_experiment(
        custom_config(tmp_path, epochs=2), command="pytest custom resume", resume=first.run_dir
    )
    assert resumed.state.next_epoch == 2


def test_custom_components_validate_like_builtins(tmp_path: Path) -> None:
    _register_custom_components()

    config = custom_config(tmp_path)
    config.model.options["layerz"] = 1
    with pytest.raises(ConfigError, match="model.layerz"):
        run_experiment(config, command="pytest invalid")
    assert not Path(config.experiment.runs_dir).exists()

    misspelled = custom_config(tmp_path)
    misspelled.model = ModelConfig("tiny_net2", {})
    with pytest.raises(
        ConfigError, match=r"unsupported model.name 'tiny_net2'; available: cnn, mlp, tiny_net"
    ):
        run_experiment(misspelled, command="pytest invalid")


def test_custom_components_mix_with_builtin_datasets(tmp_path: Path) -> None:
    _register_custom_components()
    config = custom_config(tmp_path)
    config.dataset = DatasetConfig(
        name="synthetic",
        batch_size=4,
        options={"samples": 12, "val_samples": 6, "input_shape": [4], "num_classes": 2},
    )

    result = run_experiment(config, command="pytest mixed")

    assert result.state.status == "finished"
    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "trainlattice.datasets.builtin" in manifest["dataset_source"]


def test_builtin_components_record_their_source(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment=ExperimentSection(name="builtin-api", seed=3, runs_dir=str(tmp_path / "runs")),
        model=ModelConfig("mlp", {"input_dim": 4, "hidden_dims": [8], "num_classes": 2}),
        dataset=DatasetConfig(
            name="synthetic",
            batch_size=4,
            options={"samples": 12, "val_samples": 6, "input_shape": [4], "num_classes": 2},
        ),
        optimizer=OptimizerConfig(lr=0.05),
        trainer=TrainerConfig(epochs=1, device="cpu", validate_every=1, log_every=100),
    )

    result = run_experiment(config, command="pytest builtin")

    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_source"].startswith("trainlattice.modeling.architectures (")
    assert manifest["dataset_source"].startswith("trainlattice.datasets.builtin (")
