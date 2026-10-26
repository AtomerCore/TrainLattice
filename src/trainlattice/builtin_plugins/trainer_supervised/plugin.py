"""Publish the supervised trainer factory."""


def activate(ctx):
    """Bind metrics once when composing the trainer factory."""
    from .trainer import Trainer
    metrics = ctx.require("metrics")

    def create(**options):
        return Trainer(metrics_factory=metrics, **options)

    ctx.provide("trainlattice.trainer/supervised", create)
