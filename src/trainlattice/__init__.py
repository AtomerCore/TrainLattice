"""TrainLattice public package interface.

Custom architectures, datasets, optimizers, losses, and learning-rate
schedulers register through ``ARCHITECTURES``, ``DATASETS``, ``OPTIMIZERS``,
``LOSSES``, and ``SCHEDULERS`` before an experiment starts; experiments execute
through ``run_experiment`` or ``run_from_file``.
"""

from trainlattice.callbacks import Callback
from trainlattice.config import (
    ComponentContract,
    ExperimentConfig,
    LossConfig,
    load_config,
    save_config,
)
from trainlattice.datasets import DATASETS
from trainlattice.errors import ConfigError, RegistryError, TrainLatticeError
from trainlattice.modeling import ARCHITECTURES
from trainlattice.optim import OPTIMIZERS, SCHEDULERS
from trainlattice.runtime import run_experiment, run_from_file
from trainlattice.trainer import LOSSES, register_loss

__all__ = [
    "ARCHITECTURES",
    "DATASETS",
    "LOSSES",
    "OPTIMIZERS",
    "SCHEDULERS",
    "Callback",
    "ComponentContract",
    "ConfigError",
    "ExperimentConfig",
    "LossConfig",
    "RegistryError",
    "TrainLatticeError",
    "load_config",
    "register_loss",
    "run_experiment",
    "run_from_file",
    "save_config",
]

__version__ = "0.4.0"
