"""Stable, training-independent plugin author contracts."""

from collections.abc import Callable
from typing import Any, Protocol


class StateProvider(Protocol):
    """Versioned plugin-owned state captured at an application safe boundary."""

    state_schema_version: int

    def snapshot_state(self) -> dict[str, Any]:
        """Return the plugin's serializable state."""
        ...

    def restore_state(self, payload: dict[str, Any]) -> None:
        """Restore a compatible plugin-owned state payload."""
        ...


class PluginContext(Protocol):
    """Declared service access and owned resource lifetime for one plugin."""

    @property
    def plugin_id(self) -> str:
        """Return the manifest identity of this plugin."""
        ...

    def require(self, alias: str) -> Any:
        """Resolve a declared and explicitly bound dependency alias."""
        ...

    def provide(self, service: str, implementation: Any, *, api_version: int = 1, scope: str | None = None) -> None:
        """Publish a declared service with automatic revocation."""
        ...

    def effect(self, cleanup: Callable[[], None], *, scope: str = "application") -> None:
        """Register disposal of an owned resource."""
        ...

    def config(self) -> dict[str, Any]:
        """Return an independent copy of this plugin's configuration."""
        ...
