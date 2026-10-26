"""Static exact-version binding and dependency graph resolution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .errors import PluginError
from .manifest import PluginManifest, ProvidedService
from .sources import PluginRequest, PluginSource, PluginSourceResolver
from .snapshot import content_fingerprint, contracts_fingerprint


@dataclass(frozen=True, slots=True)
class PlannedPlugin:
    """One plugin's metadata, effective bindings and private configuration."""

    source: PluginSource
    manifest: PluginManifest
    bindings: Mapping[str, str | None]
    config: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PluginPlan:
    """A topologically ordered plugin assembly description."""

    plugins: tuple[PlannedPlugin, ...]
    fingerprints: Mapping[str, str]
    contracts_fingerprint: str


class DependencyResolver:
    """Resolve dependencies without importing executable plugin code."""

    def resolve(self, requests: Sequence[str | PluginRequest], source_resolver: PluginSourceResolver, bindings: Mapping[str, Mapping[str, str | None]] | None = None) -> PluginPlan:
        """Validate sources, providers and bindings, then topologically sort."""
        overrides = bindings or {}
        plugins = {}
        providers: dict[str, tuple[str, ProvidedService]] = {}
        for request in requests:
            request = PluginRequest(request) if isinstance(request, str) else request
            source = source_resolver.resolve(request.source)
            manifest = PluginManifest.read(source.root)
            if manifest.id in plugins:
                previous = plugins[manifest.id][0]
                raise PluginError(f"duplicate plugin ID {manifest.id}: {previous.root} and {source.root}")
            plugins[manifest.id] = (source, manifest, request.config)
            for service in manifest.provides:
                if service.service in providers:
                    raise PluginError(f"service conflict {service.service}: {providers[service.service][0]} and {manifest.id}")
                providers[service.service] = (manifest.id, service)
        unknown_plugins = set(overrides) - set(plugins)
        if unknown_plugins:
            raise PluginError("bindings reference unknown plugin(s): " + ", ".join(sorted(unknown_plugins)))
        planned = {}
        edges = {}
        for plugin_id, (source, manifest, config) in plugins.items():
            selected = overrides.get(plugin_id, {})
            if not isinstance(selected, Mapping):
                raise PluginError(f"bindings.{plugin_id} must be a mapping")
            unknown_aliases = set(selected) - set(manifest.requires)
            if unknown_aliases:
                raise PluginError(f"bindings.{plugin_id}: undeclared dependencies: {', '.join(sorted(unknown_aliases))}")
            effective = {}
            dependencies = []
            for alias, requirement in manifest.requires.items():
                target = selected.get(alias, requirement.service)
                path = f"{plugin_id}.{alias}"
                if target is not None and not isinstance(target, str):
                    raise PluginError(f"bindings.{path} must be a service key or null")
                if target is None:
                    if not requirement.optional:
                        raise PluginError(f"{path}: required dependency cannot be disabled")
                    effective[alias] = None
                    continue
                provider = providers.get(target)
                if provider is None:
                    if requirement.optional and alias not in selected:
                        effective[alias] = None
                        continue
                    raise PluginError(f"{path}: missing service {target} (API {requirement.api_version})")
                owner, declaration = provider
                if declaration.api_version != requirement.api_version:
                    raise PluginError(f"{path}: service API mismatch {target}: requires {requirement.api_version}, {owner} provides {declaration.api_version}")
                effective[alias] = target
                dependencies.append(owner)
            edges[plugin_id] = tuple(dict.fromkeys(dependencies))
            planned[plugin_id] = PlannedPlugin(source, manifest, MappingProxyType(effective), MappingProxyType(deepcopy(dict(config))))
        ordered = []
        active = []
        visited = set()

        def visit(plugin_id: str) -> None:
            if plugin_id in active:
                cycle = active[active.index(plugin_id):] + [plugin_id]
                raise PluginError("plugin dependency cycle: " + " -> ".join(cycle))
            if plugin_id in visited:
                return
            active.append(plugin_id)
            for dependency in edges[plugin_id]:
                visit(dependency)
            active.pop()
            visited.add(plugin_id)
            ordered.append(planned[plugin_id])

        for plugin_id in plugins:
            visit(plugin_id)
        return PluginPlan(tuple(ordered), MappingProxyType({item.manifest.id: content_fingerprint(item.manifest.root) for item in ordered}), contracts_fingerprint())
