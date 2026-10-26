"""Owned services with explicit scope and exact API matching."""

from dataclasses import dataclass
from typing import Any

from .errors import PluginError


@dataclass(frozen=True, slots=True)
class ServiceRegistration:
    """A published service and its declared owner."""

    service: str
    implementation: Any
    provider: str
    api_version: int
    scope: str


class ServiceRegistry:
    """An application-local service graph, frozen after activation."""

    def __init__(self) -> None:
        self._items: dict[str, ServiceRegistration] = {}
        self._frozen = False
        self.run_active = False

    def register(self, registration: ServiceRegistration) -> None:
        """Publish one unique service during graph activation."""
        if self._frozen:
            raise PluginError("the active plugin graph is frozen")
        if registration.service in self._items:
            previous = self._items[registration.service]
            raise PluginError(f"service conflict {registration.service}: {previous.provider} and {registration.provider}")
        self._items[registration.service] = registration

    def resolve(self, service: str, api_version: int = 1) -> Any:
        """Resolve an active service with the requested exact API version."""
        item = self._items.get(service)
        if item is None:
            raise PluginError(f"service unavailable: {service} (API {api_version})")
        if item.api_version != api_version:
            raise PluginError(f"service API mismatch {service}: requires {api_version}, provides {item.api_version}")
        if item.scope == "run" and not self.run_active:
            raise PluginError(f"run service {service} is unavailable outside a run scope")
        return item.implementation

    def revoke(self, service: str, provider: str) -> None:
        """Revoke a registration when its owning context is disposed."""
        item = self._items.get(service)
        if item is not None and item.provider == provider:
            del self._items[service]

    def freeze(self) -> None:
        """Prevent further service publication in the active application."""
        self._frozen = True

    def owned(self, provider: str) -> set[str]:
        """Return the service keys published by an owner for activation checks."""
        return {name for name, item in self._items.items() if item.provider == provider}
