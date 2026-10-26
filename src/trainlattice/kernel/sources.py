"""Resolve local directories and installed built-in plugin resources."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
import re

from .errors import PluginError


@dataclass(frozen=True, slots=True)
class PluginSource:
    """A locator and its resolved on-disk directory."""

    locator: str
    root: Path


class PluginSourceResolver:
    """Resolve sources without downloading code or modifying import paths."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = (base_dir or Path.cwd()).resolve()

    def resolve(self, locator: str) -> PluginSource:
        """Resolve a builtin URI or a local directory relative to the config."""
        if not isinstance(locator, str) or not locator.strip():
            raise PluginError("plugin source must be a non-empty string")
        if locator.startswith("builtin://"):
            name = locator.removeprefix("builtin://")
            if not re.fullmatch(r"[a-z][a-z0-9-]*", name):
                raise PluginError(f"invalid built-in source {locator!r}")
            root = Path(str(resources.files("trainlattice.builtin_plugins").joinpath(name.replace("-", "_"))))
        elif "://" in locator:
            raise PluginError(f"unsupported plugin source {locator!r}; use a local directory")
        else:
            root = Path(locator)
            if not root.is_absolute():
                root = self.base_dir / root
        root = root.resolve()
        if not root.is_dir():
            raise PluginError(f"plugin directory not found: {locator} ({root})")
        return PluginSource(locator, root)
