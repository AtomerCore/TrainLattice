"""Publish losses with explicitly injected task contracts."""


def activate(ctx):
    """Register cross entropy, mean squared error and absolute error."""
    from trainlattice.contracts.config import validate_plugin_options
    validate_plugin_options(ctx.config(), ctx.plugin_id)
    from torch import nn
    from trainlattice.contracts.components import LossFactory
    from .validation import _validate_cross_entropy, _validate_distance_loss

    classification, regression = ctx.require("classification"), ctx.require("regression")
    ctx.provide("trainlattice.loss/cross_entropy", LossFactory("cross_entropy", nn.CrossEntropyLoss, _validate_cross_entropy, task=classification))
    ctx.provide("trainlattice.loss/mse", LossFactory("mse", nn.MSELoss, _validate_distance_loss, task=regression))
    ctx.provide("trainlattice.loss/l1", LossFactory("l1", nn.L1Loss, _validate_distance_loss, task=regression))
