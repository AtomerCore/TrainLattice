"""Strict declarative plugin manifests and entry-path validation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from types import MappingProxyType
from typing import Any

import yaml

from .errors import PluginError

_ID = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+")
_SERVICE = re.compile(r"[a-z][a-z0-9_.]*/[a-z][a-z0-9_-]*")


class _UniqueLoader(yaml.SafeLoader):
    """Reject duplicate mapping keys instead of replacing metadata."""


def _unique_mapping(loader: _UniqueLoader, node: Any) -> dict[str, Any]:
    result = {}
    for key, value in loader.construct_pairs(node, deep=True):
        if not isinstance(key, str) or key in result:
            raise PluginError(f"manifest has a non-string or duplicate key: {key!r}")
        result[key] = value
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _mapping(value: Any, path: str, allowed: set[str], required: set[str] = frozenset()) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PluginError(f"{path} must be a mapping")
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown or missing:
        details = []
        if unknown:
            details.append("unknown field(s): " + ", ".join(f"{path}.{key}" for key in sorted(unknown)))
        if missing:
            details.append("required field(s): " + ", ".join(f"{path}.{key}" for key in sorted(missing)))
        raise PluginError("; ".join(details))
    return value


def _version(value: Any, path: str) -> int:
    if type(value) is not int or value < 1:
        raise PluginError(f"{path} must be a positive integer")
    return value


def _service(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _SERVICE.fullmatch(value):
        raise PluginError(f"{path} must be a service key such as trainlattice.model/mlp")
    return value


@dataclass(frozen=True, slots=True)
class ProvidedService:
    """A declared service identity and lifetime."""

    service: str
    api_version: int = 1
    scope: str = "application"


@dataclass(frozen=True, slots=True)
class RequiredService:
    """A named dependency with an optional default provider."""

    service: str
    api_version: int = 1
    optional: bool = False


@dataclass(frozen=True, slots=True)
class PluginManifest:
    """Validated plugin metadata, read without importing entry modules."""

    root: Path
    id: str
    version: str
    entry_file: Path
    entry_callable: str
    provides: tuple[ProvidedService, ...]
    requires: Mapping[str, RequiredService]
    state_mode: str
    api_version: int = 1
    description: str = ""

    @classmethod
    def read(cls, root: Path) -> PluginManifest:
        """Read and validate a plugin directory and its resolved entry path."""
        root = root.resolve()
        source = root / "plugin.yaml"
        try:
            raw = yaml.load(source.read_text(encoding="utf-8"), Loader=_UniqueLoader)
            return cls._parse(root, raw)
        except (OSError, yaml.YAMLError, PluginError) as exc:
            raise PluginError(f"{source}: {exc}") from exc

    @classmethod
    def _parse(cls, root: Path, raw: Any) -> PluginManifest:
        required = {"schema_version", "id", "version", "api_version", "entry", "state"}
        data = _mapping(raw, "manifest", required | {"provides", "requires", "description"}, required)
        for field in ("schema_version", "api_version"):
            if _version(data[field], f"manifest.{field}") != 1:
                raise PluginError(f"manifest.{field}: unsupported version {data[field]!r}")
        plugin_id = data["id"]
        if not isinstance(plugin_id, str) or not _ID.fullmatch(plugin_id):
            raise PluginError("manifest.id must be a lowercase dotted namespace")
        if not isinstance(data["version"], str) or not data["version"].strip():
            raise PluginError("manifest.version must be a non-empty string")
        entry = data["entry"]
        if not isinstance(entry, str) or entry.count(":") != 1:
            raise PluginError("manifest.entry must be relative_file.py:callable")
        filename, function = entry.split(":")
        relative = Path(filename.replace("\\", "/"))
        if relative.is_absolute() or PureWindowsPath(filename).is_absolute() or ".." in relative.parts:
            raise PluginError("manifest.entry must stay inside the plugin root")
        entry_file = (root / relative).resolve()
        if not entry_file.is_relative_to(root) or entry_file.suffix != ".py" or not entry_file.is_file():
            raise PluginError("manifest.entry must name an existing Python file inside the plugin root")
        if not function.isidentifier():
            raise PluginError("manifest.entry callable must be a Python identifier")
        provides_raw = data.get("provides", [])
        if not isinstance(provides_raw, list):
            raise PluginError("manifest.provides must be a list")
        provides = []
        seen = set()
        for index, value in enumerate(provides_raw):
            path = f"manifest.provides.{index}"
            item = _mapping(value, path, {"service", "api_version", "scope"}, {"service"})
            key = _service(item["service"], f"{path}.service")
            if key in seen:
                raise PluginError(f"{path}.service: duplicate service {key}")
            seen.add(key)
            scope = item.get("scope", "application")
            if scope not in {"application", "run"}:
                raise PluginError(f"{path}.scope must be application or run")
            provides.append(ProvidedService(key, _version(item.get("api_version", 1), f"{path}.api_version"), scope))
        requires_raw = data.get("requires", {})
        if not isinstance(requires_raw, Mapping):
            raise PluginError("manifest.requires must be an alias mapping")
        requires = {}
        for alias, value in requires_raw.items():
            path = f"manifest.requires.{alias}"
            if not isinstance(alias, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", alias):
                raise PluginError(f"{path}: invalid dependency alias")
            item = _mapping(value, path, {"service", "api_version", "optional"}, {"service"})
            optional = item.get("optional", False)
            if type(optional) is not bool:
                raise PluginError(f"{path}.optional must be a boolean")
            requires[alias] = RequiredService(_service(item["service"], f"{path}.service"), _version(item.get("api_version", 1), f"{path}.api_version"), optional)
        state = _mapping(data["state"], "manifest.state", {"mode"}, {"mode"})
        if state["mode"] not in {"stateless", "stateful"}:
            raise PluginError("manifest.state.mode must be stateless or stateful")
        description = data.get("description", "")
        if not isinstance(description, str):
            raise PluginError("manifest.description must be a string")
        return cls(root, plugin_id, data["version"], entry_file, function, tuple(provides), MappingProxyType(requires), state["mode"], description=description)
