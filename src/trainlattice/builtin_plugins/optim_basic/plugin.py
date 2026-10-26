"""Publish optimizer and learning-rate scheduler factories."""


def activate(ctx):
    """Register the built-in optimization services."""
    from torch.optim import Adam, AdamW, SGD
    from trainlattice.contracts.components import ComponentFactory
    from .optimizers import _validate_adam_family, _validate_sgd
    from .schedulers import SchedulerFactory, _build_warmup_cosine, _build_warmup_linear, _build_step_decay, _validate_warmup, _validate_step_decay

    for name, constructor, validator in (("adamw", AdamW, _validate_adam_family), ("adam", Adam, _validate_adam_family), ("sgd", SGD, _validate_sgd)):
        ctx.provide(f"trainlattice.optimizer/{name}", ComponentFactory(name, constructor, validator))
    for name, constructor, validator in (("warmup_cosine", _build_warmup_cosine, _validate_warmup), ("warmup_linear", _build_warmup_linear, _validate_warmup), ("step_decay", _build_step_decay, _validate_step_decay)):
        ctx.provide(f"trainlattice.scheduler/{name}", SchedulerFactory(name, constructor, validator))
