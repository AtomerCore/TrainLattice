"""Publish structured console and JSONL logging."""


def activate(ctx):
    """Create run-owned log sinks through the logging factory."""
    from .loggers import CompositeLogger, ConsoleLogger, JSONLLogger

    def create(directory):
        logger = CompositeLogger([ConsoleLogger(), JSONLLogger(directory / "metrics.jsonl")])
        ctx.effect(logger.close, scope="run")
        return logger

    ctx.provide("trainlattice.logging/basic", create)
