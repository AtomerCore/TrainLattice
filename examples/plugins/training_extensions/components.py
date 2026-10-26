"""Locally owned model, data, optimizer, and scheduler declarations."""

import math
from torch.utils.data import TensorDataset
import torch

from trainlattice.contracts.components import ComponentContract
from trainlattice.errors import ConfigError


def reject_unknown(path, options, allowed):
    """Report unsupported options at their domain paths."""
    unknown = set(options) - allowed
    if unknown:
        raise ConfigError("unknown configuration field(s): " + ", ".join(f"{path}.{key}" for key in sorted(unknown)))


def positive(value, path):
    """Require a positive dimension or sample count."""
    if type(value) is not int or value < 1:
        raise ConfigError(f"{path} must be a positive integer")


def validate_linear(path, options):
    """Validate the local linear model's shape."""
    reject_unknown(path, options, {"input_dim", "num_classes"})
    for key, default in (("input_dim", 4), ("num_classes", 2)):
        positive(options.get(key, default), f"{path}.{key}")


def linear_contract(options):
    """Expose the linear model's input and output widths."""
    return ComponentContract(options.get("input_dim", 4), options.get("num_classes", 2))


def validate_sine(path, options):
    """Validate deterministic regression split sizes."""
    reject_unknown(path, options, {"samples", "val_samples"})
    for key, default in (("samples", 16), ("val_samples", 8)):
        positive(options.get(key, default), f"{path}.{key}")


def sine_contract(options):
    """Declare scalar inputs and regression targets."""
    return ComponentContract(1, 1, (1,))


def sine(*, train, split=None, samples=16, val_samples=8, seed=42):
    """Generate independent deterministic sine splits with float targets."""
    training = (split == "train") if split is not None else train
    generator = torch.Generator().manual_seed(seed + (0 if training else 1))
    inputs = torch.rand(samples if training else val_samples, 1, generator=generator) * 2 * math.pi
    return TensorDataset(inputs, torch.sin(inputs))


def validate_sgd(path, options):
    """Validate momentum SGD options before parameter construction."""
    reject_unknown(path, options, {"momentum", "weight_decay"})
    for key, value in options.items():
        if type(value) not in (float, int) or not value >= 0:
            raise ConfigError(f"{path}.{key} must be non-negative")


class ExponentialFactory:
    """An external scheduler factory implementing the scheduler service API."""

    name = "exponential"

    def validate(self, path, options):
        """Validate the exponential decay multiplier."""
        reject_unknown(path, options, {"gamma"})
        value = options.get("gamma", 0.9)
        if type(value) not in (float, int) or not value > 0:
            raise ConfigError(f"{path}.gamma must be positive")

    def create(self, config, optimizer, total_steps=None):
        """Construct a per-step exponential schedule."""
        self.validate("scheduler", config.options)
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.options.get("gamma", 0.9))
