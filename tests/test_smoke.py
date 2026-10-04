"""Exercise the installed package entry point from the repository root."""

from pathlib import Path
import subprocess
import sys

import pytest


def test_module_cli_from_repository_root() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "trainlattice", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "train" in result.stdout


@pytest.mark.parametrize("module", ["checkpoint", "trainer", "callbacks"])
def test_core_packages_import_independently(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import trainlattice.{module}"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
