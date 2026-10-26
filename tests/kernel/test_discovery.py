"""Directory discovery, metadata preflight and isolated entry loading."""

from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from trainlattice.kernel.errors import PluginError
from trainlattice.kernel.loader import PluginLoader
from trainlattice.kernel.manifest import PluginManifest
from trainlattice.kernel.sources import PluginSourceResolver


def write_plugin(root: Path, **overrides) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    data = {"schema_version": 1, "id": "community.example.answer", "version": "0.1.0", "api_version": 1, "entry": "plugin.py:activate", "state": {"mode": "stateless"}}
    data.update(overrides)
    (root / "plugin.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    (root / "plugin.py").write_text("from .helper import answer\ndef activate(ctx):\n    return answer\n", encoding="utf-8")
    (root / "helper.py").write_text("answer = 42\n", encoding="utf-8")
    return root


def test_local_plugin_metadata_does_not_execute_entry_and_loader_supports_relative_imports(tmp_path: Path) -> None:
    root = write_plugin(tmp_path / "中文 plugin")
    source = PluginSourceResolver(tmp_path).resolve(root.name)
    before = tuple(sys.path)
    manifest = PluginManifest.read(source.root)
    assert not any(getattr(module, "__file__", None) == str(root / "plugin.py") for module in tuple(sys.modules.values()))
    assert PluginLoader().load(manifest)(None) == 42
    assert tuple(sys.path) == before


def test_manifest_rejects_unknown_fields_and_entries_outside_root(tmp_path: Path) -> None:
    root = write_plugin(tmp_path / "invalid", typo=True)
    with pytest.raises(PluginError, match=r"manifest.typo"):
        PluginManifest.read(root)
    write_plugin(root, entry="../outside.py:activate")
    with pytest.raises(PluginError, match="inside the plugin root"):
        PluginManifest.read(root)


def test_package_and_manifest_import_without_torch() -> None:
    code = "import sys; sys.modules['torch'] = None; import trainlattice; from trainlattice.kernel.manifest import PluginManifest; assert 'trainlattice.trainer' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
