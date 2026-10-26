from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trainlattice.contracts.callbacks import Callback
from trainlattice.builtin_plugins.callbacks_basic.callbacks import CheckpointCallback
from helpers import CheckpointManager
from trainlattice.contracts.config import DatasetConfig, EarlyStoppingConfig, ExperimentConfig, ExperimentSection, ModelConfig, OptimizerConfig, TrainerConfig
from helpers import run_experiment
from helpers import Trainer
from trainlattice.contracts.state import TrainerState


class BiasSchedule(Callback):
    """Drive validation metrics along a script by pinning the model bias.

    The weights are zeroed once and the optimizer holds lr=0, so the
    scripted bias fully decides every epoch: with targets all zero, cross
    entropy on logits ``[b, 0]`` is ``log(1 + exp(-b))`` — strictly
    decreasing in ``b`` — and argmax flips to class 1 once the second
    component leads.
    """

    def __init__(self, biases: list[tuple[float, float]]) -> None:
        self.biases = biases
        self._index = -1

    def on_train_start(self, trainer: Trainer) -> None:
        with torch.no_grad():
            trainer.model.weight.zero_()

    def on_epoch_start(self, trainer: Trainer) -> None:
        self._index += 1
        with torch.no_grad():
            trainer.model.bias.copy_(torch.tensor(self.biases[self._index]))


class LifecycleRecorder(Callback):
    """Record the lifecycle events the run actually reached."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def on_train_start(self, trainer: Trainer) -> None:
        self.events.append("train_start")

    def on_epoch_end(self, trainer: Trainer) -> None:
        self.events.append(f"epoch_end:{trainer.state.epoch}")

    def on_train_end(self, trainer: Trainer) -> None:
        self.events.append("train_end")


class StopAtEpoch(Callback):
    """Request a cooperative stop from inside the callback chain."""

    def __init__(self, epoch: int, reason: str = "custom criterion met") -> None:
        self.epoch = epoch
        self.reason = reason

    def on_epoch_end(self, trainer: Trainer) -> None:
        if trainer.state.epoch == self.epoch:
            trainer.request_stop(self.reason)


def make_loaders(target: int) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(11)
    inputs = torch.randn(12, 4, generator=generator)
    targets = torch.full((12,), target, dtype=torch.long)
    val_inputs = torch.randn(6, 4, generator=generator)
    val_targets = torch.full((6,), target, dtype=torch.long)
    train = DataLoader(TensorDataset(inputs, targets), batch_size=3, shuffle=False)
    validation = DataLoader(TensorDataset(val_inputs, val_targets), batch_size=3, shuffle=False)
    return train, validation


def scripted_trainer(
    tmp_path: Path,
    *,
    biases: list[tuple[float, float]],
    early_stopping: EarlyStoppingConfig | None,
    monitor: str = "val_loss",
    validate_every: int = 1,
    target: int = 0,
) -> tuple[Trainer, CheckpointManager, LifecycleRecorder]:
    """Build a trainer whose per-epoch validation metric follows ``biases``."""
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    manager = CheckpointManager(tmp_path / "checkpoints")
    recorder = LifecycleRecorder()
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        epochs=len(biases),
        callbacks=[BiasSchedule(biases), recorder, CheckpointCallback(manager, {}, monitor=monitor)],
        validate_every=validate_every,
        monitor=monitor,
        early_stopping=early_stopping,
    )
    return trainer, manager, recorder


def run_config(
    tmp_path: Path, *, epochs: int, early_stopping: EarlyStoppingConfig | None = None
) -> ExperimentConfig:
    """A minimal runnable configuration with optional early stopping."""
    return ExperimentConfig(
        experiment=ExperimentSection(name="early-stopping", seed=3, runs_dir=str(tmp_path / "runs")),
        model=ModelConfig("mlp", {"input_dim": 4, "hidden_dims": [8], "num_classes": 2}),
        dataset=DatasetConfig(
            name="synthetic",
            batch_size=4,
            options={"samples": 12, "val_samples": 8, "input_shape": [4], "num_classes": 2},
        ),
        optimizer=OptimizerConfig(lr=0.01),
        trainer=TrainerConfig(
            epochs=epochs, device="cpu", validate_every=1, log_every=1, early_stopping=early_stopping
        ),
    )


def test_plateau_stops_after_patience_exceeded(tmp_path: Path) -> None:
    # one improvement, then a constant metric: epochs 2, 3, and 4 are the
    # three non-improving evaluations, and patience=2 stops after the third
    biases = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    trainer, manager, recorder = scripted_trainer(
        tmp_path, biases=biases, early_stopping=EarlyStoppingConfig(patience=2)
    )

    state = trainer.fit(*make_loaders(0))

    assert state.status == "finished"
    assert state.epoch == 4
    assert state.next_epoch == 5
    assert state.stop_reason is not None
    assert "early stopping" in state.stop_reason
    assert "epoch_end:4" in recorder.events
    assert "epoch_end:5" not in recorder.events
    assert recorder.events[-1] == "train_end"
    last = manager.directory / "last.pt"
    assert last.is_file()
    payload = torch.load(last, weights_only=False)
    assert payload["trainer_state"]["epoch"] == 4


def test_min_delta_treats_small_gains_as_no_improvement(tmp_path: Path) -> None:
    # each scripted step lowers the loss by ~0.02-0.05, all below min_delta
    biases = [(0.0, 0.0), (0.1, 0.0), (0.15, 0.0), (0.2, 0.0), (0.25, 0.0)]
    trainer, _, _ = scripted_trainer(
        tmp_path,
        biases=biases,
        early_stopping=EarlyStoppingConfig(patience=1, min_delta=0.1),
    )

    state = trainer.fit(*make_loaders(0))

    # epochs 1 and 2 are the two sub-threshold evaluations
    assert state.next_epoch == 3
    assert "early stopping" in state.stop_reason


def test_continued_improvement_runs_every_epoch(tmp_path: Path) -> None:
    biases = [(0.0, 0.0), (0.1, 0.0), (0.15, 0.0), (0.2, 0.0), (0.25, 0.0)]
    trainer, _, _ = scripted_trainer(
        tmp_path, biases=biases, early_stopping=EarlyStoppingConfig(patience=1)
    )

    state = trainer.fit(*make_loaders(0))

    assert state.status == "finished"
    assert state.next_epoch == 5
    assert state.stop_reason is None


def test_val_accuracy_is_monitored_upward(tmp_path: Path) -> None:
    # accuracy jumps to 1.0 in epoch 1 and then plateaus there
    biases = [(0.0, 0.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
    trainer, _, _ = scripted_trainer(
        tmp_path,
        biases=biases,
        early_stopping=EarlyStoppingConfig(patience=1),
        monitor="val_accuracy",
        target=1,
    )

    state = trainer.fit(*make_loaders(1))

    # epochs 2 and 3 are the two evaluations without an upward gain
    assert state.next_epoch == 4
    assert "early stopping" in state.stop_reason


def test_non_validation_epochs_skip_the_check(tmp_path: Path) -> None:
    # validation runs on epochs 1, 3, 5, ...; only those may count
    biases = [(2.0, 0.0)] * 8
    trainer, _, _ = scripted_trainer(
        tmp_path,
        biases=biases,
        early_stopping=EarlyStoppingConfig(patience=1),
        validate_every=2,
    )

    state = trainer.fit(*make_loaders(0))

    # counting non-validation epochs would have stopped after epoch 3 instead
    assert state.epoch == 5
    assert state.next_epoch == 6
    assert "early stopping" in state.stop_reason


def test_request_stop_from_extra_callback_ends_at_epoch_boundary(tmp_path: Path) -> None:
    result = run_experiment(
        run_config(tmp_path, epochs=4), command="pytest stop", extra_callbacks=[StopAtEpoch(1)]
    )

    assert result.state.status == "finished"
    assert result.state.epoch == 1
    assert result.state.next_epoch == 2
    assert result.state.stop_reason == "custom criterion met"
    assert (result.run_dir / "checkpoints" / "last.pt").is_file()


def test_resume_allows_a_new_patience_and_recovers(tmp_path: Path) -> None:
    # per-epoch val_loss gains stay below min_delta=1.0, so the strict run
    # counts epochs 1 and 2 as non-improving and stops after the second
    strict = run_config(
        tmp_path, epochs=8, early_stopping=EarlyStoppingConfig(patience=1, min_delta=1.0)
    )
    first = run_experiment(strict, command="pytest strict")
    assert first.state.next_epoch == 3
    assert "early stopping" in first.state.stop_reason

    relaxed = run_config(
        tmp_path, epochs=6, early_stopping=EarlyStoppingConfig(patience=5, min_delta=1.0)
    )
    resumed = run_experiment(relaxed, command="pytest relaxed", resume=first.run_dir)

    assert resumed.state.status == "finished"
    assert resumed.state.next_epoch == 6
    assert resumed.state.stop_reason is None


def test_stop_reason_round_trips_through_state_serialization() -> None:
    state = TrainerState(epoch=4, stop_reason="early stopping: val_loss plateaued")

    assert state.to_dict()["stop_reason"] == "early stopping: val_loss plateaued"
    assert TrainerState.from_dict(state.to_dict()).stop_reason == state.stop_reason
    # checkpoints saved before early stopping carry no stop_reason key
    with pytest.raises(ValueError, match="trainer state fields"):
        TrainerState.from_dict({"epoch": 3})


def test_trainer_rejects_unsupported_monitor(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="monitor must be one of: val_loss, val_accuracy"):
        scripted_trainer(tmp_path, biases=[(0.0, 0.0)], early_stopping=None, monitor="train_loss")


def test_early_stopping_warns_when_no_callback_advances_the_best_metric(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # without CheckpointCallback nothing advances best_metric, so every epoch
    # counts as an improvement, the run never stops, and one warning explains why
    model = nn.Linear(4, 2)
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        epochs=3,
        callbacks=[BiasSchedule([(0.0, 0.0)] * 3)],
        early_stopping=EarlyStoppingConfig(patience=1),
    )

    with caplog.at_level("WARNING", logger="trainlattice.plugins.trainer.supervised"):
        state = trainer.fit(*make_loaders(0))

    assert state.status == "finished"
    assert state.next_epoch == 3
    assert state.stop_reason is None
    warnings = [record for record in caplog.records if "no best metric" in record.message]
    assert len(warnings) == 1
