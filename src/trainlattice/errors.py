"""Domain exceptions surfaced by TrainLattice."""


class TrainLatticeError(Exception):
    """Base class for user-facing TrainLattice failures."""


class ConfigError(TrainLatticeError):
    """Raised when an experiment configuration is invalid."""


class RegistryError(TrainLatticeError):
    """Raised for invalid registry operations."""


class CheckpointError(TrainLatticeError):
    """Raised when a checkpoint cannot be saved or restored."""

