"""Generic named constructor registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

from trainlattice.errors import RegistryError

T = TypeVar("T")


class Registry(Generic[T]):
    """Map stable lowercase names to component constructors."""

    def __init__(self, category: str) -> None:
        self.category = category
        self._items: dict[str, Callable[..., T]] = {}

    @staticmethod
    def _normalize(name: str) -> str:
        normalized = name.strip().lower().replace("-", "_")
        if not normalized:
            raise RegistryError("registry names must not be empty")
        return normalized

    def register(self, name: str, target: Callable[..., T] | None = None):
        """Register a constructor directly or as a decorator."""
        key = self._normalize(name)

        def decorator(value: Callable[..., T]) -> Callable[..., T]:
            if key in self._items:
                raise RegistryError(f"{self.category} {key!r} is already registered")
            self._items[key] = value
            return value

        return decorator(target) if target is not None else decorator

    def get(self, name: str) -> Callable[..., T]:
        key = self._normalize(name)
        try:
            return self._items[key]
        except KeyError as exc:
            choices = ", ".join(self.names()) or "none"
            raise RegistryError(f"unknown {self.category} {name!r}; available: {choices}") from exc

    def build(self, name: str, /, *args: Any, **kwargs: Any) -> T:
        """Construct a registered component."""
        return self.get(name)(*args, **kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self._normalize(name) in self._items
