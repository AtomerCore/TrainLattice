"""Diagnostic logging: CLI handler configuration and library log points."""

import logging
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trainlattice.contracts.callbacks import Callback
from helpers import CheckpointManager
from trainlattice.cli.main import CLI_HANDLER_ATTR, _configure_logging, main
from trainlattice.contracts.config import DatasetConfig, ExperimentConfig, ExperimentSection, ModelConfig, OptimizerConfig, TrainerConfig
from trainlattice.builtin_plugins.logging_basic.loggers import ConsoleLogger
from helpers import run_experiment
from helpers import Trainer
from trainlattice.contracts.state import TrainerState


def tiny_config(tmp_path: Path, epochs: int = 1) -> ExperimentConfig:
    return ExperimentConfig(
        experiment=ExperimentSection(name="diagnostics-test", seed=3, runs_dir=str(tmp_path / "runs")),
        model=ModelConfig("mlp", {"input_dim": 4, "hidden_dims": [8], "num_classes": 2}),
        dataset=DatasetConfig(
            name="synthetic",
            batch_size=4,
            options={"samples": 12, "val_samples": 8, "input_shape": [4], "num_classes": 2},
        ),
        optimizer=OptimizerConfig(lr=0.05),
        trainer=TrainerConfig(epochs=epochs, device="cpu", validate_every=1, log_every=100),
    )


def loader() -> DataLoader:
    generator = torch.Generator().manual_seed(7)
    inputs = torch.randn(12, 4, generator=generator)
    targets = torch.randint(0, 2, (12,), generator=generator)
    return DataLoader(TensorDataset(inputs, targets), batch_size=3, shuffle=False)


def test_run_directory_creation_is_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="trainlattice"):
        result = run_experiment(tiny_config(tmp_path), command="pytest diagnostics")

    record = next(r for r in caplog.records if "run directory" in r.getMessage())
    assert record.name == "trainlattice.plugins.experiment.default"
    assert record.levelno == logging.INFO
    assert str(result.run_dir) in record.getMessage()


def test_checkpoint_resume_logs_source_and_next_epoch(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    first = run_experiment(tiny_config(tmp_path), command="pytest first")

    with caplog.at_level(logging.INFO, logger="trainlattice"):
        run_experiment(tiny_config(tmp_path, epochs=2), command="pytest resume", resume=first.run_dir)

    record = next(r for r in caplog.records if "next epoch" in r.getMessage())
    assert record.name == "trainlattice.plugins.experiment.default"
    assert record.levelno == logging.INFO
    assert str(first.run_dir / "checkpoints" / "last.pt") in record.getMessage()
    assert f"next epoch {first.state.next_epoch}" in record.getMessage()


def test_checkpoint_manager_logs_save_and_load(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(tmp_path / "checkpoints")

    with caplog.at_level(logging.INFO, logger="trainlattice"):
        target = manager.save(
            "last.pt",
            model=model,
            optimizer=optimizer,
            trainer_state=TrainerState(),
            config={},
        )
        manager.load(target, model=model, optimizer=optimizer)

    messages = [(r.name, r.levelno, r.getMessage()) for r in caplog.records]
    assert ("trainlattice.plugins.checkpoint.local", logging.INFO, f"checkpoint saved to {target}") in messages
    assert ("trainlattice.plugins.checkpoint.local", logging.INFO, f"checkpoint restored from {target}") in messages


class FailEveryCallback(Callback):
    def on_batch_end(self, trainer: Trainer) -> None:
        raise RuntimeError("simulated interruption")

    def on_exception(self, trainer: Trainer, exc: BaseException) -> None:
        raise ValueError("handler failure")


def test_callback_exception_during_failure_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    model = nn.Linear(4, 2)
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        epochs=1,
        callbacks=[FailEveryCallback()],
    )

    with caplog.at_level(logging.WARNING, logger="trainlattice"):
        with pytest.raises(RuntimeError, match="simulated interruption"):
            trainer.fit(loader())

    record = next(r for r in caplog.records if r.name == "trainlattice.plugins.trainer.supervised")
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    assert "FailEveryCallback" in message
    assert "RuntimeError" in message
    assert "handler failure" in message


def test_cli_error_exits_2_through_error_channel(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = tmp_path / "invalid.yaml"
    config.write_text("profile: supervised-basic\ntrainr:\n  epochs: 2\n", encoding="utf-8")

    with caplog.at_level(logging.INFO, logger="trainlattice"):
        assert main(["train", str(config)]) == 2

    record = next(r for r in caplog.records if r.name == "trainlattice.cli.main")
    assert record.levelno == logging.ERROR
    assert "unknown configuration section" in record.getMessage()
    err = capsys.readouterr().err
    assert "[ERROR] trainlattice.cli.main:" in err
    assert "unknown configuration section" in err
    assert "TrainLattice error:" not in err


DEVICE_CONFIG = """
profile: supervised-basic
experiment:
  name: device-test
model:
dataset:
trainer:
  device: tpu
"""


def test_cli_translates_unsupported_device_through_error_channel(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "device.yaml"
    config.write_text(DEVICE_CONFIG, encoding="utf-8")

    assert main(["train", str(config)]) == 2

    err = capsys.readouterr().err
    assert "[ERROR] trainlattice.cli.main:" in err
    assert "trainer.device must be one of" in err


def test_cli_translates_runtime_failures_with_exit_code_2(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "runtime-failure.yaml"
    config.write_text("experiment:\n  name: runtime-failure\n", encoding="utf-8")

    def explode(path, *, resume=None, command="trainlattice train"):
        raise RuntimeError("simulated runtime failure")

    monkeypatch.setattr("trainlattice.application.run_from_file", explode)

    assert main(["train", str(config)]) == 2

    err = capsys.readouterr().err
    assert "[ERROR] trainlattice.cli.main:" in err
    assert "simulated runtime failure" in err


def test_configure_logging_is_idempotent_and_routes_by_severity(capsys: pytest.CaptureFixture[str]) -> None:
    _configure_logging(verbose=False)
    _configure_logging(verbose=False)

    root = logging.getLogger()
    marked = [h for h in root.handlers if getattr(h, CLI_HANDLER_ATTR, False)]
    assert len(marked) == 2

    probe = logging.getLogger("trainlattice.diagnostics.probe")
    probe.info("info message")
    probe.warning("warning message")
    captured = capsys.readouterr()
    assert "[INFO] trainlattice.diagnostics.probe: info message" in captured.out
    assert "warning message" not in captured.out
    assert "[WARNING] trainlattice.diagnostics.probe: warning message" in captured.err
    assert "info message" not in captured.err


def test_configure_logging_verbose_enables_debug(capsys: pytest.CaptureFixture[str]) -> None:
    _configure_logging(verbose=True)

    logging.getLogger("trainlattice.diagnostics.probe").debug("debug message")
    captured = capsys.readouterr()
    assert "[DEBUG] trainlattice.diagnostics.probe: debug message" in captured.out
    assert "debug message" not in captured.err


def test_console_logger_metric_format_is_unchanged(capsys: pytest.CaptureFixture[str]) -> None:
    ConsoleLogger().log({"train_loss": 0.5, "train_accuracy": 1.0, "epoch": 0}, 3)
    assert capsys.readouterr().out == "step=3 train_loss=0.5000 train_accuracy=1.0000 epoch=0\n"
