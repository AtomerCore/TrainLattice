import pytest

from trainlattice.errors import RegistryError
from trainlattice.registry import Registry


def test_registry_registers_and_builds() -> None:
    registry = Registry("widget")

    @registry.register("answer")
    def build_answer(value: int = 42) -> int:
        return value

    assert registry.build("ANSWER", value=7) == 7
    assert registry.names() == ("answer",)


def test_registry_rejects_duplicates_and_reports_choices() -> None:
    registry = Registry("widget")
    registry.register("first", lambda: 1)

    with pytest.raises(RegistryError, match="already registered"):
        registry.register("first", lambda: 2)
    with pytest.raises(RegistryError, match="available: first"):
        registry.get("missing")
    with pytest.raises(RegistryError, match="available: first"):
        registry.entry("missing")


def test_registry_carries_declarations_alongside_constructors() -> None:
    registry = Registry("widget")

    def check_options(path: str, options: dict) -> None:
        if "bad" in options:
            raise ValueError(f"{path}.bad")

    @registry.register("declared", validate=check_options, contract=lambda options: options["size"])
    def build(**kwargs):
        return object()

    entry = registry.entry("declared")
    assert entry.constructor is build
    assert entry.validate is check_options
    assert entry.contract is not None and entry.contract({"size": 3}) == 3
    assert registry.get("declared") is build
    assert registry.build("declared") is not None


def test_registry_entries_without_declarations_default_to_none() -> None:
    registry = Registry("widget")
    registry.register("plain", dict)

    entry = registry.entry("plain")
    assert entry.validate is None
    assert entry.contract is None


