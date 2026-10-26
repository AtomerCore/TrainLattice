"""Load trusted plugin entry files in unique package namespaces."""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType
from typing import Any, Callable
from uuid import uuid4

from .errors import PluginError
from .manifest import PluginManifest


class PluginLoader:
    """Load entries with relative imports and no changes to sys.path."""

    def load(self, manifest: PluginManifest) -> Callable[[Any], Any]:
        """Import the declared entry and return its activation callable."""
        namespace = "_trainlattice_plugin_" + uuid4().hex
        relative = manifest.entry_file.relative_to(manifest.root)
        package_name = namespace
        package_path = manifest.root
        package = ModuleType(package_name)
        package.__path__ = [str(package_path)]
        package.__package__ = package_name
        sys.modules[package_name] = package
        for part in relative.parts[:-1]:
            package_name += "." + part
            package_path /= part
            package = ModuleType(package_name)
            package.__path__ = [str(package_path)]
            package.__package__ = package_name
            sys.modules[package_name] = package
        name = package_name + "." + manifest.entry_file.stem
        spec = importlib.util.spec_from_file_location(name, manifest.entry_file)
        if spec is None or spec.loader is None:
            raise PluginError(f"{manifest.id}: cannot load {manifest.entry_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            entry = getattr(module, manifest.entry_callable, None)
            if not callable(entry):
                raise PluginError(f"{manifest.id}: entry {manifest.entry_callable!r} is not callable")
            return entry
        except BaseException:
            for loaded in tuple(sys.modules):
                if loaded == namespace or loaded.startswith(namespace + "."):
                    sys.modules.pop(loaded, None)
            raise
