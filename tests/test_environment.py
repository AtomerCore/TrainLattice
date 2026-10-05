"""Runtime environment capture and atomic JSON writes."""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from trainlattice.runtime.environment import write_json


def test_write_json_retries_transient_replace_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_json({"status": "created"}, tmp_path / "manifest.json")
    calls = {"count": 0}
    real_replace = Path.replace

    def flaky_replace(self: Path, target):
        calls["count"] += 1
        if calls["count"] == 1:
            raise PermissionError("target transiently held open")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    write_json({"status": "running"}, tmp_path / "manifest.json")

    assert calls["count"] == 2
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == {"status": "running"}


def test_write_json_cleans_temporary_after_persistent_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def locked_replace(self: Path, target):
        raise PermissionError("target held open")

    monkeypatch.setattr(Path, "replace", locked_replace)

    with pytest.raises(OSError):
        write_json({"status": "running"}, tmp_path / "manifest.json")

    assert not (tmp_path / "manifest.json.tmp").exists()
