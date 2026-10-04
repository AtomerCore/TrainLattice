"""Shared test fixtures."""

from __future__ import annotations

import logging

import pytest


@pytest.fixture(autouse=True)
def _reset_cli_logging() -> None:
    """Detach CLI diagnostic handlers so logging state never leaks between tests."""
    yield
    try:
        from trainlattice.cli.main import CLI_HANDLER_ATTR
    except Exception:
        return
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, CLI_HANDLER_ATTR, False):
            root.removeHandler(handler)
            handler.close()
    logging.getLogger("trainlattice").setLevel(logging.NOTSET)
