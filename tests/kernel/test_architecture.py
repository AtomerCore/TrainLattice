"""Dependency boundaries and import-time plugin entry invariants."""

import ast
from pathlib import Path
import subprocess
import sys

from trainlattice.kernel.manifest import PluginManifest

ROOT = Path(__file__).resolve().parents[2] / "src" / "trainlattice"


def test_kernel_and_application_import_without_training_dependencies():
    allowed = set(sys.stdlib_module_names) | {"yaml", "trainlattice", "__future__"}
    for file in (ROOT / "kernel").glob("*.py"):
        for node in ast.walk(ast.parse(file.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                assert all(item.name.split(".")[0] in allowed for item in node.names), file
            elif isinstance(node, ast.ImportFrom) and not node.level:
                assert node.module.split(".")[0] in allowed, file
                assert not node.module.startswith("trainlattice.builtin_plugins"), file
    code = "import sys; sys.modules['torch']=None; import trainlattice.application; from trainlattice.cli.main import main; assert main(['plugins','inspect','builtin://trainer-supervised'])==0"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_plugin_entries_defer_implementation_imports_and_use_public_contracts():
    for folder in (ROOT / "builtin_plugins").iterdir():
        if not (folder / "plugin.yaml").is_file(): continue
        manifest = PluginManifest.read(folder)
        entry = ast.parse(manifest.entry_file.read_text(encoding="utf-8"))
        assert all(isinstance(node, (ast.Expr, ast.FunctionDef)) for node in entry.body), folder
        for file in folder.glob("*.py"):
            for node in ast.walk(ast.parse(file.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert not node.module.startswith("trainlattice.builtin_plugins."), file


def test_training_implementation_has_one_plugin_owner():
    for name in ("registry", "config", "modeling", "datasets", "optim", "trainer", "checkpoint", "callbacks", "logging", "runtime"):
        assert not list((ROOT / name).rglob("*.py")), name
    for name in ("trainer.py", "architectures.py", "schedulers.py", "store.py"):
        assert len(list((ROOT / "builtin_plugins").rglob(name))) == 1, name
