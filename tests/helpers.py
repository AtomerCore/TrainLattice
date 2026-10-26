"""Small test adapters that consume native plugin services."""

from trainlattice.application import build_application


def get_component(category, name):
    """Resolve a component from the current native built-in composition."""
    with build_application({"profile": "supervised-basic"}) as app:
        return app.require(f"trainlattice.{category}/{name}")


def validate_model_config(config):
    """Preflight a model's declared options."""
    get_component("model", config.name).validate("model", config.options)


def build_model(config):
    """Build a model through its selected service."""
    factory = get_component("model", config.name)
    factory.validate("model", config.options)
    return factory.build(**config.options)


def build_dataloaders(config, *, seed, factory=None):
    """Build loaders using a native or explicitly supplied dataset factory."""
    if factory is not None:
        from trainlattice.builtin_plugins.datasets_basic.loaders import build_dataloaders as assemble
        return assemble(config, seed=seed, factory=factory)
    return get_component("dataset", config.name).build_loaders(config, seed=seed)
