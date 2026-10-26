"""Shared test fixtures."""

from __future__ import annotations

import logging
import sys

import pytest


@pytest.fixture(autouse=True)
def _reset_cli_logging() -> None:
    """Detach CLI diagnostic handlers so logging state never leaks between tests."""
    yield
    cli = sys.modules.get("trainlattice.cli.main")
    if cli is None:
        return
    CLI_HANDLER_ATTR = cli.CLI_HANDLER_ATTR
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, CLI_HANDLER_ATTR, False):
            root.removeHandler(handler)
            handler.close()
    logging.getLogger("trainlattice").setLevel(logging.NOTSET)


@pytest.fixture(autouse=True)
def _isolate_component_registries(request):
    """Restore the shared component registries after tests that extend them."""
    if request.node.path.parent.name == "kernel":
        yield
        return
    try:
        from trainlattice.datasets import DATASETS
        from trainlattice.modeling import ARCHITECTURES
        from trainlattice.optim import OPTIMIZERS, SCHEDULERS
        from trainlattice.trainer.losses import LOSS_TASKS, LOSSES
    except Exception:
        yield
        return
    saved = {
        DATASETS: dict(DATASETS._items),
        ARCHITECTURES: dict(ARCHITECTURES._items),
        OPTIMIZERS: dict(OPTIMIZERS._items),
        SCHEDULERS: dict(SCHEDULERS._items),
        LOSSES: dict(LOSSES._items),
    }
    saved_tasks = dict(LOSS_TASKS)
    yield
    for registry, items in saved.items():
        registry._items.clear()
        registry._items.update(items)
    LOSS_TASKS.clear()
    LOSS_TASKS.update(saved_tasks)
