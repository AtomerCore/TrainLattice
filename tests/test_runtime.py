from pathlib import Path

import json

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
from trainlattice.errors import ConfigError


def tiny_config(tmp_path: Path, epochs: int = 1) -> ExperimentConfig:
    return ExperimentConfig(
        experiment=ExperimentSection(name="runtime-test", seed=3, runs_dir=str(tmp_path / "runs")),
        model=ModelConfig("mlp", {"input_dim": 4, "hidden_dims": [8], "num_classes": 2}),
        dataset=DatasetConfig(
            name="synthetic",
            batch_size=4,
            options={"samples": 12, "val_samples": 8, "input_shape": [4], "num_classes": 2},
        ),
        optimizer=OptimizerConfig(lr=0.05),
        trainer=TrainerConfig(epochs=epochs, device="cpu", validate_every=1, log_every=100),
    )


def test_run_creates_reproducible_artifacts_and_resumes(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    first = run_experiment(config, command="pytest runtime")

    for name in ("config.yaml", "environment.json", "manifest.json", "metrics.jsonl"):
        assert (first.run_dir / name).is_file()
    assert (first.run_dir / "checkpoints" / "last.pt").is_file()
    first_manifest = json.loads((first.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert first_manifest["status"] == "finished"

    resumed_config = tiny_config(tmp_path, epochs=2)
    resumed = run_experiment(resumed_config, command="pytest resume", resume=first.run_dir)
    assert resumed.run_dir != first.run_dir
    assert resumed.state.next_epoch == 2
    assert resumed.state.global_step == 6
    resumed_manifest = json.loads((resumed.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert resumed_manifest["resume_source"].endswith("last.pt")


def test_model_validation_fails_before_creating_run_directory(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    config.model.options["hidden_dimz"] = [8]

    with pytest.raises(ConfigError, match="model.hidden_dimz"):
        run_experiment(config, command="pytest invalid")

    assert not Path(config.experiment.runs_dir).exists()


def test_cross_component_validation_rejects_input_mismatch(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    config.model.options["input_dim"] = 5

    with pytest.raises(ConfigError, match="model.input_dim"):
        run_experiment(config, command="pytest incompatible")

    assert not Path(config.experiment.runs_dir).exists()


def test_cross_component_validation_rejects_class_count_mismatch(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    config.model.options["num_classes"] = 3

    with pytest.raises(ConfigError, match="model.num_classes"):
        run_experiment(config, command="pytest incompatible")

    assert not Path(config.experiment.runs_dir).exists()
