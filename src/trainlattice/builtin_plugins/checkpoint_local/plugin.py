"""Activate local checkpoint persistence."""


def activate(ctx):
    """Publish the checkpoint v5 factory."""
    from .store import CheckpointFactory
    ctx.provide("trainlattice.checkpoint/local", CheckpointFactory())
