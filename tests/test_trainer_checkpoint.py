from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trainlattice.callbacks import Callback, CheckpointCallback
from trainlattice.checkpoint import CheckpointManager
from trainlattice.config.schema import DatasetConfig
from trainlattice.datasets import build_dataloaders
from trainlattice.errors import CheckpointError
from trainlattice.trainer import Trainer, TrainerState


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


def test_checkpoint_rejects_unsupported_format_version(tmp_path: Path) -> None:
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
    payload["format_version"] = 2
    legacy = manager.directory / "legacy.pt"
    torch.save(payload, legacy)

    with pytest.raises(CheckpointError, match="unsupported checkpoint format version"):
        manager.load(legacy, model=model, optimizer=optimizer)
