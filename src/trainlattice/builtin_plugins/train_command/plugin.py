"""Publish the train command through the experiment service."""


def activate(ctx):
    """Bind command dispatch without owning experiment implementation."""
    ctx.provide("trainlattice.command/train", ctx.require("experiment"))
