"""Publish deterministic synthetic and local tensor datasets."""


def activate(ctx):
    """Register dataset factories with their loader and shape contracts."""
    from trainlattice.contracts.config import validate_plugin_options
    validate_plugin_options(ctx.config(), ctx.plugin_id)
    from trainlattice.contracts.components import DatasetFactory
    from .loaders import build_dataloaders
    from .synthetic import build_synthetic, _validate_synthetic, _synthetic_contract
    from .tensor import build_tensor, _validate_tensor, _tensor_contract

    ctx.provide("trainlattice.dataset/synthetic", DatasetFactory("synthetic", build_synthetic, _validate_synthetic, _synthetic_contract, build_dataloaders))
    ctx.provide("trainlattice.dataset/tensor", DatasetFactory("tensor", build_tensor, _validate_tensor, _tensor_contract, build_dataloaders))
    ctx.provide("trainlattice.data/loaders", build_dataloaders)
