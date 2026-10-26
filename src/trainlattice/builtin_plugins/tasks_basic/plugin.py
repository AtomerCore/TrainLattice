"""Publish task declarations for supported supervised objectives."""


def activate(ctx):
    """Register classification and regression task contracts."""
    from trainlattice.contracts.config import validate_plugin_options
    validate_plugin_options(ctx.config(), ctx.plugin_id)
    from trainlattice.contracts.components import Task

    ctx.provide("trainlattice.task/classification", Task("classification"))
    ctx.provide("trainlattice.task/regression", Task("regression"))
