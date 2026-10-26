"""Publish the native experiment orchestration service."""


def activate(ctx):
    """Resolve component factories once during plugin assembly."""
    from .experiment import execute
    services = {alias: ctx.require(alias) for alias in (
        "runtime", "model", "dataset", "optimizer", "loss", "scheduler", "trainer", "checkpoint", "logging", "callbacks")}
    ctx.provide("trainlattice.experiment/default", lambda request: execute(request, services))
