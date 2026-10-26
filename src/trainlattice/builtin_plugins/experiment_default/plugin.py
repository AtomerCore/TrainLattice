"""Publish the native experiment orchestration service."""


def activate(ctx):
    """Resolve component factories once during plugin assembly."""
    from trainlattice.contracts.config import validate_plugin_options
    validate_plugin_options(ctx.config(), ctx.plugin_id)
    from .experiment import execute
    services = {alias: ctx.require(alias) for alias in (
        "runtime", "model", "dataset", "optimizer", "loss", "scheduler", "trainer", "checkpoint", "logging", "callbacks")}
    ctx.provide("trainlattice.experiment/default", lambda request: execute(request, services))
