from pathlib import Path

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
    config.write_text("trainr:\n  epochs: 2\n", encoding="utf-8")

    assert main(["train", str(config)]) == 2
    assert "unknown configuration section" in capsys.readouterr().err
