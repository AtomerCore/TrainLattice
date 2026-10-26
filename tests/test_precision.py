"""Mixed precision: schema limits, device gating, bf16 runs, and scaler checkpoints."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trainlattice.builtin_plugins.callbacks_basic.callbacks import CheckpointCallback
from helpers import CheckpointManager
from trainlattice.contracts.config import DatasetConfig, ExperimentConfig, ExperimentSection, ModelConfig, OptimizerConfig, TrainerConfig
from trainlattice.errors import ConfigError, TrainLatticeError
from helpers import run_experiment
from helpers import Trainer
from trainlattice.contracts.state import TrainerState
from trainlattice.builtin_plugins.trainer_supervised.trainer import GradScaler


def cpu_scaler() -> GradScaler:
    """Build a scaler the checkpoint path can exercise without a cuda device."""
    try:
        return GradScaler(device="cpu")
    except TypeError:
        return GradScaler()


class FlakyScaler:
    """A scaler double that exercises the skipped-window path by backing off."""

    def __init__(self, *, backed_off: bool) -> None:
        self.backed_off = backed_off
        self.scale = 65536.0

    def get_scale(self) -> float:
        return self.scale

    def unscale_(self, optimizer) -> None: ...

    def step(self, optimizer) -> None: ...

    def update(self) -> None:
        if self.backed_off:
            self.scale /= 2.0


@pytest.mark.filterwarnings("ignore:Detected call of")
def test_skipped_scaler_window_advances_neither_scheduler_nor_step_count() -> None:
    from torch.optim.lr_scheduler import StepLR

    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        epochs=1,
        scheduler=scheduler,
    )
    window = trainer._totals_cls()
    logits = torch.tensor([[2.0, 0.0]])
    window.update(logits, torch.tensor([0]), torch.tensor(0.5))
    before_epoch = scheduler.last_epoch

    trainer.scaler = FlakyScaler(backed_off=True)
    trainer._optimizer_step(window, {})
    assert trainer.state.global_step == 0
    assert scheduler.last_epoch == before_epoch

    trainer.scaler = FlakyScaler(backed_off=False)
    trainer._optimizer_step(window, {})
    assert trainer.state.global_step == 1
    assert scheduler.last_epoch == before_epoch + 1


@pytest.mark.parametrize("precision", ["tf32", "fp8", 16, None])
def test_schema_rejects_unknown_precision(precision) -> None:
    config = ExperimentConfig(
        experiment=ExperimentSection(name="precision"),
        model=ModelConfig("mlp", {"input_dim": 4, "hidden_dims": [8], "num_classes": 2}),
        dataset=DatasetConfig(name="synthetic", batch_size=4, options={"samples": 12}),
        trainer=TrainerConfig(epochs=1, device="cpu", precision=precision),
    )

    with pytest.raises(ConfigError, match="trainer.precision must be one of: fp32, bf16, fp16"):
        config.validate()


def test_trainer_rejects_unknown_precision() -> None:
    model = nn.Linear(4, 2)
    with pytest.raises(ValueError, match="precision must be one of: fp32, bf16, fp16"):
        Trainer(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            loss_fn=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            epochs=1,
            precision="tf32",
        )


def test_fp16_requires_a_cuda_device_with_bf16_guidance() -> None:
    model = nn.Linear(4, 2)
    with pytest.raises(ValueError, match="bf16"):
        Trainer(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            loss_fn=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            epochs=1,
            precision="fp16",
        )


def test_bf16_accumulated_window_matches_the_full_batch_step() -> None:
    """accumulate=2 under bf16 matches one batch_size=8 step within bf16 rounding."""
    generator = torch.Generator().manual_seed(11)
    inputs = torch.randn(8, 4, generator=generator)
    targets = torch.randint(0, 2, (8,), generator=generator)

    def run(batch_size: int, accumulate: int) -> Trainer:
        torch.manual_seed(0)
        model = nn.Linear(4, 2)
        trainer = Trainer(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            loss_fn=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            epochs=1,
            accumulate=accumulate,
            precision="bf16",
        )
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=batch_size)
        trainer.train_epoch(loader)
        return trainer

    full_run = run(8, 1)
    accumulated_run = run(4, 2)

    assert accumulated_run.state.global_step == 1
    for expected, actual in zip(full_run.model.parameters(), accumulated_run.model.parameters()):
        assert torch.allclose(expected, actual, rtol=1e-3, atol=1e-3)
    assert accumulated_run.batch_metrics["loss"] == pytest.approx(full_run.batch_metrics["loss"], rel=1e-3)


def precision_config(tmp_path: Path, *, epochs: int, precision: str) -> ExperimentConfig:
    """A minimal runnable cpu configuration at the requested precision."""
    return ExperimentConfig(
        experiment=ExperimentSection(name="precision", seed=3, runs_dir=str(tmp_path / "runs")),
        model=ModelConfig("mlp", {"input_dim": 4, "hidden_dims": [8], "num_classes": 2}),
        dataset=DatasetConfig(
            name="synthetic",
            batch_size=4,
            options={"samples": 12, "val_samples": 8, "input_shape": [4], "num_classes": 2},
        ),
        optimizer=OptimizerConfig(lr=0.05),
        trainer=TrainerConfig(
            epochs=epochs, device="cpu", validate_every=1, log_every=1, precision=precision
        ),
    )


def test_bf16_training_and_resume_reproduce_the_uninterrupted_run(tmp_path: Path) -> None:
    uninterrupted = run_experiment(
        precision_config(tmp_path, epochs=2, precision="bf16"), command="pytest uninterrupted"
    )
    first = run_experiment(
        precision_config(tmp_path, epochs=1, precision="bf16"), command="pytest first stage"
    )
    resumed = run_experiment(
        precision_config(tmp_path, epochs=2, precision="bf16"),
        command="pytest resume",
        resume=first.run_dir,
    )

    for key in ("train_loss", "val_loss"):
        assert math.isfinite(uninterrupted.state.latest_metrics[key])
    assert resumed.state.global_step == uninterrupted.state.global_step == 6
    expected = torch.load(uninterrupted.run_dir / "checkpoints" / "last.pt", weights_only=False)["model"]
    actual = torch.load(resumed.run_dir / "checkpoints" / "last.pt", weights_only=False)["model"]
    for key in expected:
        assert torch.equal(expected[key], actual[key])


def test_resume_rejects_a_changed_precision(tmp_path: Path) -> None:
    first = run_experiment(
        precision_config(tmp_path, epochs=1, precision="fp32"), command="pytest first stage"
    )

    with pytest.raises(TrainLatticeError, match="incompatible"):
        run_experiment(
            precision_config(tmp_path, epochs=2, precision="bf16"),
            command="pytest resume",
            resume=first.run_dir,
        )


def test_scaler_state_round_trips_through_checkpoints(tmp_path: Path) -> None:
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(tmp_path / "checkpoints")
    callback = CheckpointCallback(manager, {})
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        epochs=1,
        callbacks=[callback],
        state=TrainerState(latest_metrics={}),
    )

    scaler = cpu_scaler()
    forged = dict(scaler.state_dict())
    initial_scale = scaler.get_scale()
    forged["scale"] = forged["scale"] * 4
    forged["growth_interval"] = 17
    scaler.load_state_dict(forged)
    # the callback forwards whatever scaler the trainer carries into the payload
    trainer.scaler = scaler
    callback.on_epoch_end(trainer)

    payload = torch.load(manager.directory / "last.pt", weights_only=False)
    assert payload["format_version"] == 5
    assert payload["scaler"]["growth_interval"] == 17

    restored = cpu_scaler()
    manager.load(manager.directory / "last.pt", model=model, optimizer=optimizer, scaler=restored)
    assert restored.state_dict()["growth_interval"] == 17
    assert restored.get_scale() == pytest.approx(initial_scale * 4)

    payload["scaler"] = None
    incomplete = manager.directory / "incomplete.pt"
    torch.save(payload, incomplete)
    with pytest.raises(TrainLatticeError, match="missing scaler state"):
        manager.load(incomplete, model=model, optimizer=optimizer, scaler=cpu_scaler())
