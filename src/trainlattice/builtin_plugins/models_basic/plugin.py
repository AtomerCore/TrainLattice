"""Publish configurable model factories."""


def activate(ctx):
    """Register MLP and CNN through the common plugin protocol."""
    from trainlattice.contracts.components import ComponentFactory
    from .architectures import CNN, MLP
    from .validation import _validate_mlp, _mlp_contract, _validate_cnn, _cnn_contract

    ctx.provide("trainlattice.model/mlp", ComponentFactory("mlp", MLP, _validate_mlp, _mlp_contract))
    ctx.provide("trainlattice.model/cnn", ComponentFactory("cnn", CNN, _validate_cnn, _cnn_contract))
