"""Dependency preflight and transactional plugin lifetime behavior."""

from pathlib import Path

import pytest

from trainlattice.kernel.effects import EffectStack
from trainlattice.kernel.errors import CleanupError, PluginError
from trainlattice.kernel.lifecycle import PluginManager
from trainlattice.kernel.sources import PluginSourceResolver
from test_discovery import write_plugin


def test_dependency_preflight_rejects_missing_service_without_executing_code(tmp_path: Path) -> None:
    root = write_plugin(tmp_path / "missing", requires={"answer": {"service": "example.widget/answer"}})
    (root / "plugin.py").write_text("raise AssertionError('entry executed')", encoding="utf-8")
    with pytest.raises(PluginError, match="missing service example.widget/answer"):
        PluginManager().plan([str(root)])


def test_dependency_cycle_reports_the_entire_chain(tmp_path: Path) -> None:
    a = write_plugin(tmp_path / "a", id="example.plugin.a", provides=[{"service": "example.widget/a"}], requires={"other": {"service": "example.widget/b"}})
    b = write_plugin(tmp_path / "b", id="example.plugin.b", provides=[{"service": "example.widget/b"}], requires={"other": {"service": "example.widget/a"}})
    with pytest.raises(PluginError, match=r"example.plugin.a -> example.plugin.b -> example.plugin.a"):
        PluginManager().plan([str(a), str(b)])


def test_dependency_versions_and_duplicate_providers_are_preflight_errors(tmp_path: Path) -> None:
    a = write_plugin(tmp_path / "a", id="example.plugin.a", provides=[{"service": "example.widget/a"}])
    b = write_plugin(tmp_path / "b", id="example.plugin.b", requires={"other": {"service": "example.widget/a", "api_version": 2}})
    manager = PluginManager()
    with pytest.raises(PluginError, match="API mismatch"):
        manager.plan([str(a), str(b)])
    with pytest.raises(PluginError, match="duplicate plugin ID"):
        manager.plan([str(a), str(a)])
    write_plugin(b, id="example.plugin.b", provides=[{"service": "example.widget/a"}])
    with pytest.raises(PluginError, match="service conflict"):
        manager.plan([str(a), str(b)])


def test_activation_failure_revokes_current_and_preceding_plugin_effects(tmp_path: Path) -> None:
    a = write_plugin(tmp_path / "a", id="example.plugin.a", provides=[{"service": "example.widget/a"}])
    b = write_plugin(tmp_path / "b", id="example.plugin.b", requires={"other": {"service": "example.widget/a"}})
    marker = tmp_path / "resource"
    (a / "plugin.py").write_text(f"from pathlib import Path\ndef activate(ctx):\n    marker = Path({str(marker)!r})\n    marker.write_text('open')\n    ctx.effect(marker.unlink)\n    ctx.provide('example.widget/a', 42)\n", encoding="utf-8")
    (b / "plugin.py").write_text("def activate(ctx):\n    assert ctx.require('other') == 42\n    raise RuntimeError('second plugin failed')\n", encoding="utf-8")
    manager = PluginManager()
    with pytest.raises(PluginError, match="second plugin failed"):
        manager.activate(manager.plan([str(b), str(a)]))
    assert not marker.exists()
    with manager.activate(manager.plan([str(a)])) as app:
        assert app.require("example.widget/a") == 42
    assert not marker.exists()
    with pytest.raises(PluginError, match="closed"):
        app.require("example.widget/a")


def test_cleanup_continues_after_errors_and_repeated_close_is_idempotent() -> None:
    stack = EffectStack()
    calls = []
    stack.add(lambda: calls.append("first"))
    stack.add(lambda: (_ for _ in ()).throw(ValueError("cleanup failure")))
    stack.add(lambda: calls.append("last"))
    with pytest.raises(CleanupError, match="cleanup failure"):
        stack.close()
    stack.close()
    assert calls == ["last", "first"]


def test_application_run_scope_disposes_resources_and_rejects_reuse(tmp_path: Path) -> None:
    root = write_plugin(tmp_path / "run", provides=[{"service": "example.widget/run", "scope": "run"}])
    (root / "plugin.py").write_text("def activate(ctx):\n    ctx.provide('example.widget/run', 42)\n", encoding="utf-8")
    manager = PluginManager(PluginSourceResolver(tmp_path))
    with manager.activate(manager.plan([root.name])) as app:
        with app.run_scope():
            assert app.require("example.widget/run") == 42
        with pytest.raises(PluginError, match="unavailable"):
            app.require("example.widget/run")
        with pytest.raises(PluginError, match="already been used"):
            with app.run_scope():
                pass
    app.close()
