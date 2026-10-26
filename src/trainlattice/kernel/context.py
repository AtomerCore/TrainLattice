"""Plugin-local configuration, declared access and tracked effects."""

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

from .effects import EffectStack
from .errors import PluginError
from .manifest import PluginManifest
from .services import ServiceRegistration, ServiceRegistry


class PluginContextImpl:
    """Enforce one manifest's bindings and resource ownership."""

    def __init__(self, manifest: PluginManifest, services: ServiceRegistry, bindings: Mapping[str, str | None] | None = None, config: Mapping[str, Any] | None = None) -> None:
        self.manifest = manifest
        self.services = services
        self.bindings = {alias: requirement.service for alias, requirement in manifest.requires.items()}
        self.bindings.update(bindings or {})
        self._config = deepcopy(dict(config or {}))
        self.effects = EffectStack()
        self._sealed = False

    @property
    def plugin_id(self) -> str:
        """Return the context's manifest ID."""
        return self.manifest.id

    def config(self) -> dict[str, Any]:
        """Return a configuration copy that cannot mutate another plugin."""
        return deepcopy(self._config)

    def require(self, alias: str) -> Any:
        """Resolve a declared dependency with its exact API contract."""
        requirement = self.manifest.requires.get(alias)
        if requirement is None:
            raise PluginError(f"{self.plugin_id}: undeclared dependency {alias!r}")
        service = self.bindings[alias]
        if service is None:
            if requirement.optional:
                return None
            raise PluginError(f"{self.plugin_id}.{alias}: required dependency cannot be disabled")
        return self.services.resolve(service, requirement.api_version)

    def provide(self, service: str, implementation: Any, *, api_version: int = 1, scope: str | None = None) -> None:
        """Publish a manifest-declared service and track its revocation."""
        if self._sealed:
            raise PluginError(f"{self.plugin_id}: the active plugin graph is frozen")
        declaration = next((item for item in self.manifest.provides if item.service == service), None)
        if declaration is None:
            raise PluginError(f"{self.plugin_id}: undeclared provided service {service}")
        scope = declaration.scope if scope is None else scope
        if type(api_version) is not int or api_version != declaration.api_version or scope != declaration.scope:
            raise PluginError(f"{self.plugin_id}: service {service} differs from its manifest API or scope")
        self.services.register(ServiceRegistration(service, implementation, self.plugin_id, api_version, scope))
        self.effects.add(lambda: self.services.revoke(service, self.plugin_id), scope)

    def effect(self, cleanup: Callable[[], None], *, scope: str = "application") -> None:
        """Track resource cleanup in this plugin's owning scope."""
        if scope == "run" and not self.services.run_active:
            raise PluginError(f"{self.plugin_id}: run effects require an active run scope")
        self.effects.add(cleanup, scope)

    def seal(self) -> None:
        """Verify publication completeness and freeze service registration."""
        missing = {item.service for item in self.manifest.provides} - self.services.owned(self.plugin_id)
        if missing:
            raise PluginError(f"{self.plugin_id}: declared services were not provided: {', '.join(sorted(missing))}")
        self._sealed = True

    def close(self, scope: str | None = None) -> None:
        """Dispose owned resources and revoke services."""
        self._sealed = True
        self.effects.close(scope)
