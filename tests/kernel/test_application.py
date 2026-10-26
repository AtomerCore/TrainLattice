"""Native command bootstrap and ownership across success and failure."""

from pathlib import Path

import pytest
import yaml

from trainlattice.application import run_experiment, run_from_file
from trainlattice.errors import ConfigError
from test_discovery import write_plugin


def command_plugin(root: Path, code: str) -> Path:
    root = write_plugin(root, provides=[{"service": "trainlattice.command/train"}])
    (root / "plugin.py").write_text(code, encoding="utf-8")
    return root


def test_native_file_bootstrap_resolves_local_sources_and_plugin_private_config(tmp_path: Path) -> None:
    command_plugin(tmp_path / "command", "def activate(ctx):\n    ctx.provide('trainlattice.command/train', lambda request: ctx.config())\n")
    config = {"plugins": [{"source": "./command", "config": {"width": 4}}]}
    source = tmp_path / "native.yaml"
    source.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert run_from_file(source) == {"width": 4}
    assert run_experiment(config, base_dir=tmp_path) == {"width": 4}
    with pytest.raises(ConfigError, match="profile or"):
        run_experiment({"model": {"width": 4}})


def test_failed_command_releases_activated_plugin_resources(tmp_path: Path) -> None:
    marker = tmp_path / "open-resource"
    root = command_plugin(tmp_path / "command", f"from pathlib import Path\nfrom trainlattice.errors import ConfigError\ndef activate(ctx):\n    marker = Path({str(marker)!r})\n    marker.write_text('open')\n    ctx.effect(marker.unlink)\n    def execute(request):\n        raise ConfigError('model.width is invalid')\n    ctx.provide('trainlattice.command/train', execute)\n")
    with pytest.raises(ConfigError, match="model.width"):
        run_experiment({"plugins": [{"source": str(root)}]})
    assert not marker.exists()


def test_repeated_experiments_get_fresh_application_state(tmp_path: Path) -> None:
    root = command_plugin(tmp_path / "command", "def activate(ctx):\n    values = []\n    def execute(request):\n        values.append(1)\n        return len(values)\n    ctx.provide('trainlattice.command/train', execute)\n")
    config = {"plugins": [{"source": str(root)}]}
    assert run_experiment(config) == 1
    assert run_experiment(config) == 1
