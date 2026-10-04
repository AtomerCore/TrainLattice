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

