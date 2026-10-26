from pathlib import Path

import json

import pytest

pytest.importorskip("torch")

from trainlattice.cli.main import main


def test_train_help_documents_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["train", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "config" in out
    assert "--resume" in out


def test_unknown_command_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["snapshot"])
    assert excinfo.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_invalid_config_returns_domain_exit_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "invalid.yaml"
    config.write_text("profile: supervised-basic\ntrainr:\n  epochs: 2\n", encoding="utf-8")

    assert main(["train", str(config)]) == 2
    assert "unknown configuration section" in capsys.readouterr().err


def test_verbose_runs_record_the_flag_in_the_manifest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "verbose.yaml"
    config.write_text(
        f"""
profile: supervised-basic
experiment:
  name: cli-verbose
  runs_dir: "{(tmp_path / "runs").as_posix()}"
model:
  input_dim: 4
  hidden_dims: [8]
dataset:
  batch_size: 4
  samples: 8
  val_samples: 4
  input_shape: [4]
trainer:
  epochs: 1
  device: cpu
""",
        encoding="utf-8",
    )

    assert main(["train", str(config), "--verbose"]) == 0

    out = capsys.readouterr().out
    run_dir = Path(out.splitlines()[-1].split(": ", 1)[1])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["command"].endswith("--verbose")
