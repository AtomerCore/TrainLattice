"""Generic named constructor registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from trainlattice.errors import RegistryError

T = TypeVar("T")

OptionValidator = Callable[[str, Mapping[str, Any]], None]
OptionContract = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True, slots=True)
class Registration(Generic[T]):
    """A registered constructor plus its optional validate and contract declarations.

    ``validate`` receives a dotted-path prefix and the resolved options mapping and
    raises ``ConfigError`` for unsupported or invalid options. ``contract`` maps the
    resolved options to a comparable component contract.
    """

    constructor: Callable[..., T]
    validate: OptionValidator | None = None
    contract: OptionContract | None = None


class Registry(Generic[T]):
    """Map stable lowercase names to constructors and their declarations."""

    def __init__(self, category: str) -> None:
        self.category = category
        self._items: dict[str, Registration[T]] = {}

    @staticmethod
    def _normalize(name: str) -> str:
        normalized = name.strip().lower().replace("-", "_")
        if not normalized:
            raise RegistryError("registry names must not be empty")
        return normalized

    def register(
        self,
        name: str,
        target: Callable[..., T] | None = None,
        *,
        validate: OptionValidator | None = None,
        contract: OptionContract | None = None,
    ):
        """Register a constructor directly or as a decorator, with optional declarations."""
        key = self._normalize(name)

        def decorator(value: Callable[..., T]) -> Callable[..., T]:
            if key in self._items:
                raise RegistryError(f"{self.category} {key!r} is already registered")
            self._items[key] = Registration(value, validate=validate, contract=contract)
            return value

        return decorator(target) if target is not None else decorator

    def entry(self, name: str) -> Registration[T]:
        """Return the full registration for a name."""
        key = self._normalize(name)
        try:
            return self._items[key]
        except KeyError as exc:
            choices = ", ".join(self.names()) or "none"
            raise RegistryError(f"unknown {self.category} {name!r}; available: {choices}") from exc

    def get(self, name: str) -> Callable[..., T]:
        return self.entry(name).constructor

    def build(self, name: str, /, *args: Any, **kwargs: Any) -> T:
        """Construct a registered component."""
        return self.get(name)(*args, **kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self._normalize(name) in self._items
