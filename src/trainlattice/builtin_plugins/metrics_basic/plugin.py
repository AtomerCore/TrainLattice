"""Publish task-aware metric accumulators."""


def activate(ctx):
    """Bind a factory for each supported task's sample-weighted totals."""
    from trainlattice.contracts.config import validate_plugin_options
    validate_plugin_options(ctx.config(), ctx.plugin_id)
    from .metrics import ClassificationTotals, LossTotals

    def create(task):
        if task not in {"classification", "regression"}:
            raise ValueError(f"unsupported metric task: {task}")
        return ClassificationTotals() if task == "classification" else LossTotals()

    ctx.provide("trainlattice.metrics/basic", create)
