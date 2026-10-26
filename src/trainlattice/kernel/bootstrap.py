"""Read only plugin locations, private configs and explicit service bindings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
import re
from typing import Any

import yaml

from trainlattice.errors import ConfigError
from .sources import PluginRequest


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping with a source-specific error."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ConfigError(f"{path}: configuration root must be a string-keyed mapping")
    return raw


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    """Training-independent plugin requests and binding overrides."""

    plugins: tuple[PluginRequest, ...]
    bindings: dict[str, dict[str, str | None]]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> BootstrapConfig:
        """Expand a built-in profile and append explicitly requested plugins."""
        if not isinstance(raw, Mapping):
            raise ConfigError("configuration root must be a mapping")
        profile = raw.get("profile")
        profile_raw = {}
        if profile is not None:
            if not isinstance(profile, str) or not re.fullmatch(r"[a-z][a-z0-9-]*", profile):
                raise ConfigError("profile must name a built-in profile")
            root = Path(str(resources.files("trainlattice.profiles")))
            profile_raw = read_yaml(root / f"{profile}.yaml")
        explicit = raw.get("plugins", [])
        if not isinstance(explicit, list):
            raise ConfigError("plugins must be a list")
        entries = [*profile_raw.get("plugins", []), *explicit]
        if not entries:
            raise ConfigError("profile or a non-empty plugins list is required")
        plugins = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping) or set(entry) - {"source", "config"}:
                raise ConfigError(f"plugins.{index} must contain source and optional config")
            source, config = entry.get("source"), entry.get("config", {})
            if not isinstance(source, str) or not source:
                raise ConfigError(f"plugins.{index}.source must be a non-empty string")
            if not isinstance(config, Mapping) or any(not isinstance(key, str) for key in config):
                raise ConfigError(f"plugins.{index}.config must be a string-keyed mapping")
            plugins.append(PluginRequest(source, dict(config)))
        bindings = {}
        for mapping in (profile_raw.get("bindings", {}), raw.get("bindings", {})):
            if not isinstance(mapping, Mapping):
                raise ConfigError("bindings must be a mapping")
            for plugin_id, values in mapping.items():
                if not isinstance(plugin_id, str) or not isinstance(values, Mapping):
                    raise ConfigError("bindings must map plugin IDs to dependency mappings")
                if any(not isinstance(alias, str) or (value is not None and not isinstance(value, str)) for alias, value in values.items()):
                    raise ConfigError(f"bindings.{plugin_id} must map aliases to service keys or null")
                bindings.setdefault(plugin_id, {}).update(values)
        return cls(tuple(plugins), bindings)
