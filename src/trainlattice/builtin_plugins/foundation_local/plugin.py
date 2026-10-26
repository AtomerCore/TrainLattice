"""Publish local runtime facilities on activation."""


def activate(ctx):
    """Register the local runtime without initializing a device or RNG."""
    from .runtime import LocalRuntime

    ctx.provide("trainlattice.runtime/local", LocalRuntime())
