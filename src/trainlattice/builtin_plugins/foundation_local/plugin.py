"""Publish local runtime facilities on activation."""


def activate(ctx):
    """Register the local runtime without initializing a device or RNG."""
    from trainlattice.contracts.config import validate_plugin_options
    validate_plugin_options(ctx.config(), ctx.plugin_id)
    try:
        from .runtime import LocalRuntime
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            from trainlattice.errors import ConfigError
            raise ConfigError("training requires PyTorch; install 'trainlattice[train]'") from exc
        raise

    ctx.provide("trainlattice.runtime/local", LocalRuntime())
