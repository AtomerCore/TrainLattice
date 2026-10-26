"""Publish the supervised trainer factory."""


def activate(ctx):
    """Bind metrics once when composing the trainer factory."""
    from trainlattice.contracts.config import validate_plugin_options
    validate_plugin_options(ctx.config(), ctx.plugin_id)
    from .trainer import Trainer
    metrics = ctx.require("metrics")

    def create(**options):
        return Trainer(metrics_factory=metrics, **options)

    ctx.provide("trainlattice.trainer/supervised", create)
