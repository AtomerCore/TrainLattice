"""Publish ordered training lifecycle callbacks."""


def activate(ctx):
    """Build logging, user, and checkpoint observers in that order."""
    from .callbacks import LoggingCallback, CheckpointCallback

    def create(*, logger, checkpoint, config, log_every, monitor, scheduler_budget=None, extra=()):
        return [LoggingCallback(logger, log_every=log_every), *extra,
                CheckpointCallback(checkpoint, config, monitor=monitor, scheduler_budget=scheduler_budget)]

    ctx.provide("trainlattice.callbacks/basic", create)
