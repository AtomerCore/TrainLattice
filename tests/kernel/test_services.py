"""Declared service access, frozen registration and context disposal."""

from pathlib import Path

import pytest

from trainlattice.kernel.context import PluginContextImpl
from trainlattice.kernel.errors import PluginError
from trainlattice.kernel.manifest import PluginManifest
from trainlattice.kernel.services import ServiceRegistry
from test_discovery import write_plugin


def test_context_binds_declared_dependencies_and_copies_configuration(tmp_path: Path) -> None:
    root = write_plugin(tmp_path / "provider", provides=[{"service": "example.widget/answer"}])
    registry = ServiceRegistry()
    provider = PluginContextImpl(PluginManifest.read(root), registry)
    provider.provide("example.widget/answer", 42)
    consumer = PluginContextImpl(PluginManifest.read(write_plugin(tmp_path / "consumer", requires={"answer": {"service": "example.widget/default"}})), registry, {"answer": "example.widget/answer"}, {"nested": [1]})
    assert consumer.require("answer") == 42
    changed = consumer.config()
    changed["nested"].append(2)
    assert consumer.config() == {"nested": [1]}
    with pytest.raises(PluginError, match="undeclared dependency"):
        consumer.require("other")
    provider.close()
    with pytest.raises(PluginError, match="unavailable"):
        consumer.require("answer")


def test_service_conflicts_and_publication_after_activation_are_rejected(tmp_path: Path) -> None:
    manifest = PluginManifest.read(write_plugin(tmp_path, provides=[{"service": "example.widget/answer"}]))
    registry = ServiceRegistry()
    context = PluginContextImpl(manifest, registry)
    context.provide("example.widget/answer", 42)
    with pytest.raises(PluginError, match="service conflict"):
        context.provide("example.widget/answer", 43)
    context.seal()
    with pytest.raises(PluginError, match="frozen"):
        context.provide("example.widget/answer", 43)


def test_run_services_are_scoped_and_cleanup_is_idempotent(tmp_path: Path) -> None:
    manifest = PluginManifest.read(write_plugin(tmp_path, provides=[{"service": "example.widget/answer", "scope": "run"}]))
    registry = ServiceRegistry()
    context = PluginContextImpl(manifest, registry)
    context.provide("example.widget/answer", 42)
    with pytest.raises(PluginError, match="outside a run scope"):
        registry.resolve("example.widget/answer")
    registry.run_active = True
    assert registry.resolve("example.widget/answer") == 42
    context.close("run")
    context.close()
    context.close()
    assert registry.owned(manifest.id) == set()
    assert ServiceRegistry().owned(manifest.id) == set()
