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
def _small_training_workloads():
    """Keep tiny CPU fixtures on one thread without importing torch for kernel tests."""
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(1)
