"""Publish the train command through the experiment service."""


def activate(ctx):
    """Bind command dispatch without owning experiment implementation."""
    from trainlattice.contracts.config import validate_plugin_options
    validate_plugin_options(ctx.config(), ctx.plugin_id)
    ctx.provide("trainlattice.command/train", ctx.require("experiment"))
