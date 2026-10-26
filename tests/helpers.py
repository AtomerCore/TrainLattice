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


def validate_optimizer_config(config):
    """Validate the selected native optimizer."""
    get_component("optimizer", config.name).validate("optimizer", config.options)


def build_optimizer(config, params):
    """Construct an optimizer with its declared options."""
    factory = get_component("optimizer", config.name)
    factory.validate("optimizer", config.options)
    return factory.build(params, lr=config.lr, **config.options)


def validate_loss_config(config):
    """Validate the selected native loss."""
    get_component("loss", config.name).validate("loss", config.options)


def build_loss(config):
    """Construct a task-declaring loss service."""
    factory = get_component("loss", config.name)
    factory.validate("loss", config.options)
    return factory.build(**config.options)


def loss_task(name):
    """Read the task kind of a native loss service."""
    return get_component("loss", name).task.kind


def validate_scheduler_config(config):
    """Validate the selected native scheduler."""
    get_component("scheduler", config.name).validate("scheduler", config.options)


def build_scheduler(config, optimizer, total_steps=None):
    """Construct a native scheduler with a resolved budget."""
    return get_component("scheduler", config.name).create(config, optimizer, total_steps)
