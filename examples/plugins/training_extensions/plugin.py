"""Activate local training extensions using public domain contracts."""


def activate(ctx):
    """Publish components, counted callbacks, and versioned epoch state."""
    from trainlattice.contracts.components import ComponentFactory, DatasetFactory
    from trainlattice.errors import ConfigError
    from torch import nn
    from torch.optim import SGD
    from .components import linear_contract, sine, sine_contract, validate_linear, validate_sine, validate_sgd, ExponentialFactory
    from .metrics import PredictionTotals
    from .state import EpochCounter

    config = ctx.config()
    if set(config) - {"metric_scale"}:
        raise ConfigError("plugins.example.training.extensions.config contains unknown fields")
    scale = config.get("metric_scale", 1.0)
    if type(scale) not in (int, float) or not scale > 0:
        raise ConfigError("plugins.example.training.extensions.config.metric_scale must be positive")
    base_metrics, base_callbacks = ctx.require("metrics"), ctx.require("callbacks")
    counter = EpochCounter()

    def callbacks(**options):
        options["extra"] = [*options.get("extra", ()), counter]
        return base_callbacks(**options)

    ctx.provide("example.optimizer/momentum_sgd", ComponentFactory("momentum_sgd", SGD, validate_sgd))
    ctx.provide("example.model/linear", ComponentFactory("linear", lambda input_dim=4, num_classes=2: nn.Linear(input_dim, num_classes), validate_linear, linear_contract))
    ctx.provide("example.dataset/sine", DatasetFactory("sine", sine, validate_sine, sine_contract, ctx.require("loaders")))
    ctx.provide("example.scheduler/exponential", ExponentialFactory())
    ctx.provide("example.metrics/predictions", lambda task: PredictionTotals(base_metrics(task), scale))
    ctx.provide("example.callbacks/counted", callbacks)
    return counter
