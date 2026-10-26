"""Stable source fingerprints and portable plugin assembly snapshots."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .errors import PluginError

if TYPE_CHECKING:
    from .dependencies import PluginPlan


def content_fingerprint(root: Path) -> str:
    """Hash sorted implementation and metadata files relative to their root."""
    digest = hashlib.sha256()
    ignored = {"__pycache__", ".git", ".venv", ".pytest_cache"}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or ignored.intersection(relative.parts) or path.suffix not in {".py", ".yaml", ".yml", ".json", ".toml"}:
            continue
        if not path.resolve().is_relative_to(root.resolve()):
            raise PluginError(f"fingerprint source escapes plugin root: {path}")
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def contracts_fingerprint() -> str:
    """Fingerprint shared domain contracts without importing training code."""
    digest = hashlib.sha256()
    digest.update(content_fingerprint(Path(__file__).resolve().parents[1] / "contracts").encode())
    digest.update(Path(__file__).with_name("api.py").read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PluginSnapshot:
    """Traceable graph data with a location-independent recovery identity."""

    document: dict[str, Any]

    @classmethod
    def from_plan(cls, plan: PluginPlan) -> PluginSnapshot:
        """Record planned source identity, effective services and bindings."""
        plugins = []
        for plugin in plan.plugins:
            manifest = plugin.manifest
            plugins.append({
                "id": manifest.id, "version": manifest.version, "api_version": manifest.api_version,
                "source": plugin.source.locator, "path": str(manifest.root),
                "fingerprint": plan.fingerprints[manifest.id], "state_mode": manifest.state_mode,
                "services": [{"service": item.service, "api_version": item.api_version, "scope": item.scope} for item in manifest.provides],
                "bindings": dict(plugin.bindings), "config": deepcopy(dict(plugin.config)),
            })
        return cls({"schema_version": 1, "contracts_fingerprint": plan.contracts_fingerprint, "plugins": plugins})

    def to_dict(self) -> dict[str, Any]:
        """Return an independent JSON/YAML-compatible snapshot."""
        return deepcopy(self.document)

    def identity(self) -> dict[str, Any]:
        """Return recovery identity independent of directory relocation."""
        document = self.to_dict()
        for plugin in document["plugins"]:
            plugin.pop("source", None)
            plugin.pop("path", None)
        document["plugins"].sort(key=lambda plugin: plugin["id"])
        return document
