"""Scoped LIFO disposal that completes even when individual cleanups fail."""

from collections.abc import Callable

from .errors import CleanupError, PluginError


class EffectStack:
    """Own application/run resources and dispose every effect exactly once."""

    def __init__(self) -> None:
        self._items: list[tuple[str, Callable[[], None]]] = []
        self._closed: set[str] = set()

    def add(self, cleanup: Callable[[], None], scope: str = "application") -> None:
        """Track an effect in an open scope."""
        if scope not in {"application", "run"} or scope in self._closed:
            raise PluginError(f"effect scope {scope!r} is invalid or closed")
        if not callable(cleanup):
            raise PluginError("effect cleanup must be callable")
        self._items.append((scope, cleanup))

    def close(self, scope: str | None = None) -> None:
        """Dispose selected effects in reverse order, collecting failures."""
        selected = {scope} if scope is not None else {"application", "run"}
        if not selected <= {"application", "run"}:
            raise PluginError(f"invalid effect scope {scope!r}")
        self._closed.update(selected)
        pending = [item for item in self._items if item[0] in selected]
        self._items = [item for item in self._items if item[0] not in selected]
        errors = []
        for _, cleanup in reversed(pending):
            try:
                cleanup()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise CleanupError(errors)
