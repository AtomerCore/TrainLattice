"""Atomic plugin activation, frozen services and deterministic teardown."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import logging
from types import MappingProxyType
from typing import Any

from .api import StateProvider
from .context import PluginContextImpl
from .dependencies import DependencyResolver, PluginPlan
from .errors import CleanupError, PluginError
from .loader import PluginLoader
from .services import ServiceRegistry
from .sources import PluginRequest, PluginSourceResolver

logger = logging.getLogger(__name__)


class Application:
    """A frozen graph with one run scope and owned plugin contexts."""

    def __init__(self, plan: PluginPlan, services: ServiceRegistry, contexts: list[PluginContextImpl], state_providers: dict[str, StateProvider]) -> None:
        self.plan = plan
        self._services = services
        self._contexts = contexts
        self.state_providers = MappingProxyType(state_providers)
        self._closed = False
        self._run_started = False

    def require(self, service: str, *, api_version: int = 1) -> Any:
        """Resolve a public service for the application shell."""
        if self._closed:
            raise PluginError("application is closed")
        return self._services.resolve(service, api_version)

    def _dispose(self, scope: str | None = None) -> None:
        errors = []
        for context in reversed(self._contexts):
            try:
                context.close(scope)
            except CleanupError as exc:
                errors.extend(exc.errors)
        if errors:
            raise CleanupError(errors)

    @contextmanager
    def run_scope(self) -> Iterator[Application]:
        """Enable one run's services and dispose its resources on every exit."""
        if self._closed or self._run_started:
            raise PluginError("application run scope is closed or has already been used")
        self._run_started = True
        self._services.run_active = True
        try:
            yield self
        except BaseException:
            try:
                self._dispose("run")
            except CleanupError as exc:
                logger.error("cleanup after run failure: %s", exc)
            raise
        else:
            self._dispose("run")
        finally:
            self._services.run_active = False

    def close(self) -> None:
        """Dispose contexts exactly once in reverse dependency order."""
        if self._closed:
            return
        self._closed = True
        self._services.run_active = False
        self._dispose()

    def __enter__(self) -> Application:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            self.close()
        except CleanupError:
            if exc is None:
                raise
            logger.exception("cleanup after application failure")


class PluginManager:
    """Plan metadata-only assemblies and activate them transactionally."""

    def __init__(self, source_resolver: PluginSourceResolver | None = None) -> None:
        self.source_resolver = source_resolver or PluginSourceResolver()

    def plan(self, sources: Sequence[str | PluginRequest], *, bindings: Mapping[str, Mapping[str, str | None]] | None = None) -> PluginPlan:
        """Resolve the complete graph without importing plugin entries."""
        return DependencyResolver().resolve(sources, self.source_resolver, bindings)

    def activate(self, plan: PluginPlan) -> Application:
        """Activate in dependency order or revoke every completed effect."""
        services = ServiceRegistry()
        contexts = []
        providers = {}
        current_id = "<assembly>"
        try:
            for plugin in plan.plugins:
                manifest = plugin.manifest
                current_id = manifest.id
                context = PluginContextImpl(manifest, services, plugin.bindings, plugin.config)
                contexts.append(context)
                provider = PluginLoader().load(manifest)(context)
                if manifest.state_mode == "stateful":
                    version = getattr(provider, "state_schema_version", None)
                    if type(version) is not int or version < 1 or not callable(getattr(provider, "snapshot_state", None)) or not callable(getattr(provider, "restore_state", None)):
                        raise PluginError(f"{manifest.id}: stateful plugin must return a versioned state provider")
                    providers[manifest.id] = provider
                elif provider is not None:
                    raise PluginError(f"{manifest.id}: stateless activation must return None")
                context.seal()
            services.freeze()
            return Application(plan, services, contexts, providers)
        except BaseException as exc:
            cleanup_errors = []
            for context in reversed(contexts):
                try:
                    context.close()
                except CleanupError as cleanup:
                    cleanup_errors.extend(cleanup.errors)
            if cleanup_errors:
                logger.error("rollback errors after %s: %s", current_id, CleanupError(cleanup_errors))
            if isinstance(exc, Exception):
                error = PluginError(f"{current_id}: plugin activation failed: {exc}")
                error.cleanup_errors = tuple(cleanup_errors)
                raise error from exc
            raise
