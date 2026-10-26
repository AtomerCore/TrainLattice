from pathlib import Path

import json
from math import cos, pi

import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trainlattice.contracts.callbacks import Callback
from trainlattice.builtin_plugins.callbacks_basic.callbacks import CheckpointCallback
from helpers import CheckpointManager
from trainlattice.contracts.config import DatasetConfig, ExperimentConfig, ExperimentSection, ModelConfig, OptimizerConfig, SchedulerConfig, TrainerConfig
from helpers import build_dataloaders
from trainlattice.errors import CheckpointError
from helpers import build_scheduler
from helpers import run_experiment
from helpers import Trainer
from trainlattice.contracts.state import TrainerState


class EventRecorder(Callback):
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_train_start(self, trainer: Trainer) -> None:
        self.events.append("train_start")

    def on_epoch_start(self, trainer: Trainer) -> None:
        self.events.append("epoch_start")

    def on_batch_end(self, trainer: Trainer) -> None:
        self.events.append("batch_end")

    def on_epoch_end(self, trainer: Trainer) -> None:
        self.events.append("epoch_end")

    def on_train_end(self, trainer: Trainer) -> None:
        self.events.append("train_end")


class RaiseOnBatch(Callback):
    def on_batch_end(self, trainer: Trainer) -> None:
        raise RuntimeError("simulated interruption")


def loader() -> DataLoader:
    generator = torch.Generator().manual_seed(7)
    inputs = torch.randn(12, 4, generator=generator)
    targets = torch.randint(0, 2, (12,), generator=generator)
    return DataLoader(TensorDataset(inputs, targets), batch_size=3, shuffle=False)


def test_trainer_lifecycle_and_checkpoint_round_trip(tmp_path: Path) -> None:
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    manager = CheckpointManager(tmp_path / "checkpoints")
    recorder = EventRecorder()
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        epochs=2,
        callbacks=[recorder, CheckpointCallback(manager, {"name": "test"})],
    )

    state = trainer.fit(loader(), loader())

    assert state.status == "finished"
    assert state.next_epoch == 2
    assert state.global_step == 8
    assert recorder.events[0] == "train_start"
    assert recorder.events[-1] == "train_end"
    assert (tmp_path / "checkpoints" / "last.pt").is_file()

    restored_model = nn.Linear(4, 2)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1, momentum=0.9)
    restored_state = manager.load(
        tmp_path / "checkpoints" / "last.pt",
        model=restored_model,
        optimizer=restored_optimizer,
    )

    assert restored_state.next_epoch == 2
    assert restored_state.global_step == 8
    for original, restored in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(original, restored)


def test_resume_continues_epoch_and_step(tmp_path: Path) -> None:
    model = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    manager = CheckpointManager(tmp_path / "checkpoints")
    first = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        epochs=1,
        callbacks=[CheckpointCallback(manager, {})],
    )
    first.fit(loader())

    resumed_model = nn.Linear(4, 2)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=0.01)
    state = manager.load(manager.directory / "last.pt", model=resumed_model, optimizer=resumed_optimizer)
    resumed = Trainer(
        model=resumed_model,
        optimizer=resumed_optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        epochs=2,
        state=state,
    )

    final = resumed.fit(loader())
    assert final.next_epoch == 2
    assert final.global_step == 8


def test_exception_creates_diagnostic_emergency_checkpoint(tmp_path: Path) -> None:
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(tmp_path / "checkpoints")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        epochs=2,
        callbacks=[RaiseOnBatch(), CheckpointCallback(manager, {})],
    )

    with pytest.raises(RuntimeError, match="simulated interruption"):
        trainer.fit(loader())

    assert trainer.state.status == "failed"
    emergency = tmp_path / "checkpoints" / "emergency.pt"
    assert emergency.is_file()
    assert not (tmp_path / "checkpoints" / "last.pt").exists()
    with pytest.raises(CheckpointError, match="diagnostic emergency snapshot"):
        manager.load(emergency, model=model, optimizer=optimizer)


def test_resume_matches_uninterrupted_training_exactly(tmp_path: Path) -> None:
    dataset_config = DatasetConfig(
        name="synthetic",
        batch_size=3,
        options={"samples": 12, "val_samples": 6, "input_shape": [4], "num_classes": 2},
    )

    torch.manual_seed(31)
    uninterrupted_model = nn.Linear(4, 2)
    uninterrupted_optimizer = torch.optim.SGD(uninterrupted_model.parameters(), lr=0.05, momentum=0.9)
    uninterrupted = Trainer(
        model=uninterrupted_model,
        optimizer=uninterrupted_optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        epochs=2,
    )
    uninterrupted.fit(build_dataloaders(dataset_config, seed=9).train)

    torch.manual_seed(31)
    resumed_model = nn.Linear(4, 2)
    resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.05, momentum=0.9)
    manager = CheckpointManager(tmp_path / "exact")
    first_stage = Trainer(
        model=resumed_model,
        optimizer=resumed_optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        epochs=1,
        callbacks=[CheckpointCallback(manager, {"experiment": {"name": "exact"}})],
    )
    first_stage.fit(build_dataloaders(dataset_config, seed=9).train)

    reloaded_model = nn.Linear(4, 2)
    reloaded_optimizer = torch.optim.SGD(reloaded_model.parameters(), lr=0.05, momentum=0.9)
    state = manager.load(
        manager.directory / "last.pt", model=reloaded_model, optimizer=reloaded_optimizer
    )
    second_stage = Trainer(
        model=reloaded_model,
        optimizer=reloaded_optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        epochs=2,
        state=state,
    )
    second_stage.fit(build_dataloaders(dataset_config, seed=9).train)

    for expected, actual in zip(uninterrupted_model.parameters(), reloaded_model.parameters()):
        assert torch.equal(expected, actual)


def test_checkpoint_rejects_incompatible_recovery_config(tmp_path: Path) -> None:
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(tmp_path / "compatibility")
    saved_config = {"model": {"name": "mlp", "input_dim": 4}}
    manager.save(
        "last.pt",
        model=model,
        optimizer=optimizer,
        trainer_state=TrainerState(),
        config=saved_config,
    )

    with pytest.raises(CheckpointError, match="incompatible"):
        manager.load(
            "last.pt",
            model=model,
            optimizer=optimizer,
            expected_config={"model": {"name": "mlp", "input_dim": 8}},
        )


def test_checkpoint_accepts_runtime_only_config_changes(tmp_path: Path) -> None:
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(tmp_path / "runtime-only")
    saved_config = {
        "experiment": {"name": "unit", "seed": 3, "runs_dir": "runs"},
        "trainer": {"epochs": 1, "device": "cpu", "validate_every": 1, "log_every": 10},
    }
    manager.save(
        "last.pt",
        model=model,
        optimizer=optimizer,
        trainer_state=TrainerState(),
        config=saved_config,
    )

    resumed = manager.load(
        "last.pt",
        model=model,
        optimizer=optimizer,
        expected_config={
            "experiment": {"name": "unit", "seed": 3, "runs_dir": "elsewhere"},
            "trainer": {"epochs": 5, "device": "cuda", "validate_every": 1, "log_every": 1},
        },
    )

    assert resumed.next_epoch == 0


@pytest.mark.parametrize("version", [3, 4])
def test_checkpoint_rejects_unsupported_format_version(tmp_path: Path, version: int) -> None:
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(tmp_path / "legacy")
    current = manager.save(
        "current.pt",
        model=model,
        optimizer=optimizer,
        trainer_state=TrainerState(global_step=7),
        config={},
    )
    payload = torch.load(current, weights_only=False)
    assert payload["format_version"] == 5
    assert payload["scheduler"] is None
    payload["format_version"] = version
    legacy = manager.directory / f"legacy-{version}.pt"
    torch.save(payload, legacy)

    with pytest.raises(CheckpointError, match="unsupported checkpoint format version"):
        manager.load(legacy, model=model, optimizer=optimizer)


def scheduler_run_config(tmp_path: Path, epochs: int) -> ExperimentConfig:
    """A minimal runnable configuration with a warmup-cosine scheduler.

    ``total_steps`` is explicit so the schedule stays fixed while the
    runtime-only ``trainer.epochs`` differs between the stages of a resume.
    """
    return ExperimentConfig(
        experiment=ExperimentSection(name="scheduler-checkpoint", seed=3, runs_dir=str(tmp_path / "runs")),
        model=ModelConfig("mlp", {"input_dim": 4, "hidden_dims": [8], "num_classes": 2}),
        dataset=DatasetConfig(
            name="synthetic",
            batch_size=4,
            options={"samples": 12, "val_samples": 8, "input_shape": [4], "num_classes": 2},
        ),
        optimizer=OptimizerConfig(lr=0.05),
        trainer=TrainerConfig(epochs=epochs, device="cpu", validate_every=1, log_every=1),
        scheduler=SchedulerConfig("warmup_cosine", {"warmup_steps": 1, "min_lr": 0.005, "total_steps": 6}),
    )


def logged_learning_rates(run_dir: Path) -> dict[int, float]:
    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    return {record["step"]: record["lr"] for record in records if record["phase"] == "train"}


def test_scheduler_state_round_trips_through_format_v5(tmp_path: Path) -> None:
    uninterrupted = run_experiment(scheduler_run_config(tmp_path, epochs=2), command="pytest uninterrupted")
    first = run_experiment(scheduler_run_config(tmp_path, epochs=1), command="pytest first stage")

    payload = torch.load(first.run_dir / "checkpoints" / "last.pt", weights_only=False)
    assert payload["format_version"] == 5
    assert payload["scheduler"]["last_epoch"] == 3
    assert payload["scheduler_budget"] == 6

    resumed = run_experiment(
        scheduler_run_config(tmp_path, epochs=2), command="pytest resume", resume=first.run_dir
    )

    reference = logged_learning_rates(uninterrupted.run_dir)
    resumed_rates = logged_learning_rates(resumed.run_dir)
    assert sorted(resumed_rates) == [4, 5, 6]
    for step, rate in resumed_rates.items():
        assert rate == pytest.approx(reference[step])


def test_checkpoint_requires_scheduler_state(tmp_path: Path) -> None:
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(tmp_path / "missing-state")
    path = manager.save("last.pt", model=model, optimizer=optimizer, trainer_state=TrainerState(), config={})
    scheduler = build_scheduler(SchedulerConfig("step_decay", {"step_size": 2}), optimizer)
    with pytest.raises(CheckpointError, match="missing scheduler state"):
        manager.load(path, model=model, optimizer=optimizer, scheduler=scheduler)


class GradNormRecorder(Callback):
    def __init__(self) -> None:
        self.norms: list[float] = []

    def on_batch_end(self, trainer: Trainer) -> None:
        total = 0.0
        for parameter in trainer.model.parameters():
            if parameter.grad is not None:
                total += float(parameter.grad.norm() ** 2)
        self.norms.append(total**0.5)


def test_max_grad_norm_bounds_gradient_total_norm() -> None:
    torch.manual_seed(5)
    model = nn.Linear(12, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    generator = torch.Generator().manual_seed(6)
    inputs = torch.randn(12, 12, generator=generator) * 100
    targets = torch.randint(0, 3, (12,), generator=generator)
    loader_ = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=False)
    recorder = GradNormRecorder()
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        epochs=1,
        callbacks=[recorder],
        max_grad_norm=0.5,
    )

    trainer.fit(loader_)

    assert recorder.norms
    assert all(norm <= 0.5 * (1 + 1e-4) for norm in recorder.norms)


def test_best_checkpoint_tracks_monitored_metric_direction(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / "checkpoints")
    loss_callback = CheckpointCallback(manager, {}, monitor="val_loss")
    accuracy_callback = CheckpointCallback(manager, {}, monitor="val_accuracy")

    assert loss_callback._improved(0.5, None)
    assert loss_callback._improved(0.4, 0.5)
    assert not loss_callback._improved(0.6, 0.5)
    assert accuracy_callback._improved(0.6, 0.5)
    assert not accuracy_callback._improved(0.4, 0.5)


def test_best_checkpoint_written_on_improvement(tmp_path: Path) -> None:
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(tmp_path / "checkpoints")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        epochs=1,
        callbacks=[CheckpointCallback(manager, {})],
    )

    trainer.fit(loader(), loader())

    best = tmp_path / "checkpoints" / "best.pt"
    assert best.is_file()
    assert trainer.state.best_metric is not None
    assert trainer.state.best_metric == trainer.state.latest_metrics["val_loss"]
    restored_model = nn.Linear(4, 2)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1)
    restored_state = manager.load(best, model=restored_model, optimizer=restored_optimizer)
    assert restored_state.best_metric == trainer.state.best_metric


def test_last_checkpoint_carries_the_fresh_best_metric(tmp_path: Path) -> None:
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(tmp_path / "checkpoints")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        epochs=2,
        callbacks=[CheckpointCallback(manager, {})],
    )

    trainer.fit(loader(), loader())

    payload = torch.load(tmp_path / "checkpoints" / "last.pt", weights_only=False)
    assert payload["trainer_state"]["best_metric"] == trainer.state.best_metric


def test_nan_metric_never_becomes_the_best(tmp_path: Path) -> None:
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(tmp_path / "checkpoints")
    callback = CheckpointCallback(manager, {})
    state = TrainerState(latest_metrics={"val_loss": float("nan")})
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        epochs=1,
        callbacks=[callback],
        state=state,
    )

    callback.on_epoch_end(trainer)

    assert state.best_metric is None
    assert not (tmp_path / "checkpoints" / "best.pt").exists()
