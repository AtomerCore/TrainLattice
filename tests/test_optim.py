"""Optimizer registry and construction behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch
from torch.optim import SGD, Adam, AdamW

from trainlattice.contracts.config import OptimizerConfig
from trainlattice.errors import ConfigError
from helpers import get_component, build_optimizer, validate_optimizer_config
from trainlattice.kernel.errors import PluginError


def _params() -> list[torch.nn.Parameter]:
    return [torch.nn.Parameter(torch.zeros(2))]


def test_services_expose_the_builtin_optimizers() -> None:
    assert [get_component("optimizer", name).name for name in ("adam", "adamw", "sgd")] == ["adam", "adamw", "sgd"]


def test_builds_each_builtin_with_options() -> None:
    plain = build_optimizer(OptimizerConfig(), _params())
    assert isinstance(plain, AdamW)
    assert plain.defaults["lr"] == 0.001

    sgd = build_optimizer(
        OptimizerConfig("sgd", lr=0.1, options={"momentum": 0.9, "nesterov": True}), _params()
    )
    assert isinstance(sgd, SGD)
    assert sgd.defaults["momentum"] == 0.9

    adam = build_optimizer(OptimizerConfig("adam", lr=0.01, options={"eps": 1e-8}), _params())
    assert isinstance(adam, Adam)


def test_rejects_unknown_and_invalid_options() -> None:
    with pytest.raises(PluginError, match="trainlattice.optimizer/rmsprop"):
        validate_optimizer_config(OptimizerConfig("rmsprop"))
    with pytest.raises(ConfigError, match="optimizer.momentumum"):
        validate_optimizer_config(OptimizerConfig("sgd", options={"momentumum": 0.9}))
    with pytest.raises(ConfigError, match="optimizer.momentum must be a non-negative number"):
        validate_optimizer_config(OptimizerConfig("sgd", options={"momentum": -0.1}))
    with pytest.raises(ConfigError, match="optimizer.betas must contain two numbers"):
        validate_optimizer_config(OptimizerConfig("adamw", options={"betas": [0.5]}))


def test_runs_end_to_end_with_sgd_and_records_source(tmp_path: Path) -> None:
    import json

    from trainlattice.contracts.config import (
        DatasetConfig,
        ExperimentConfig,
        ExperimentSection,
        ModelConfig,
        TrainerConfig,
    )
    from helpers import run_experiment

    config = ExperimentConfig(
        experiment=ExperimentSection(name="sgd-run", seed=3, runs_dir=str(tmp_path / "runs")),
        model=ModelConfig("mlp", {"input_dim": 4, "hidden_dims": [8], "num_classes": 2}),
        dataset=DatasetConfig(
            name="synthetic",
            batch_size=4,
            options={"samples": 12, "val_samples": 6, "input_shape": [4], "num_classes": 2},
        ),
        optimizer=OptimizerConfig("sgd", lr=0.05, options={"momentum": 0.9}),
        trainer=TrainerConfig(epochs=1, device="cpu", validate_every=1, log_every=100),
    )

    result = run_experiment(config, command="pytest sgd")

    assert result.state.status == "finished"
    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "torch.optim.sgd" in manifest["optimizer_source"]
