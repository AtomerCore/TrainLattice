"""Activate local checkpoint persistence."""


def activate(ctx):
    """Publish the checkpoint v5 factory."""
    from trainlattice.contracts.config import validate_plugin_options
    validate_plugin_options(ctx.config(), ctx.plugin_id)
    from .store import CheckpointFactory
    ctx.provide("trainlattice.checkpoint/local", CheckpointFactory())
