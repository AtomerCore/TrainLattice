"""Source identity and metadata-only diagnostic commands."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from trainlattice.cli.main import main
from trainlattice.kernel.errors import PluginError
from trainlattice.kernel.lifecycle import PluginManager
from trainlattice.kernel.snapshot import PluginSnapshot
from test_discovery import write_plugin


def test_snapshot_tracks_implementation_content_and_allows_relocation(tmp_path: Path) -> None:
    root = write_plugin(tmp_path / "original")
    manager = PluginManager()
    first = PluginSnapshot.from_plan(manager.plan([str(root)]))
    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    assert PluginSnapshot.from_plan(manager.plan([str(relocated)])).identity() == first.identity()
    (root / "helper.py").write_text("answer = 43\n", encoding="utf-8")
    assert PluginSnapshot.from_plan(manager.plan([str(root)])).identity() != first.identity()


def test_activation_rejects_source_changes_since_planning(tmp_path: Path) -> None:
    root = write_plugin(tmp_path)
    manager = PluginManager()
    plan = manager.plan([str(root)])
    (root / "helper.py").write_text("answer = 43\n", encoding="utf-8")
    with pytest.raises(PluginError, match="source changed"):
        manager.activate(plan)


def test_diagnostic_commands_are_static_and_do_not_import_torch(tmp_path: Path, capsys) -> None:
    root = write_plugin(tmp_path / "plugin")
    (root / "plugin.py").write_text("raise AssertionError('entry executed')", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"plugins": [{"source": "./plugin"}]}), encoding="utf-8")
    assert main(["plugins", "inspect", str(root)]) == 0
    assert main(["plugins", "validate", str(root)]) == 0
    assert main(["plugins", "list", "--config", str(config)]) == 0
    assert main(["plugins", "graph", "--config", str(config)]) == 0
    assert "community.example.answer" in capsys.readouterr().out
    code = f"import sys; sys.modules['torch'] = None; from trainlattice.cli.main import main; raise SystemExit(main(['plugins', 'validate', {str(root)!r}]))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
