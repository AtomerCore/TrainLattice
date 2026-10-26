"""Plugin discovery, contract and lifecycle diagnostics."""

from trainlattice.errors import TrainLatticeError


class PluginError(TrainLatticeError):
    """A plugin manifest, dependency or execution contract failed."""


class CleanupError(PluginError):
    """All cleanup actions ran, but at least one failed."""

    def __init__(self, errors: list[BaseException]) -> None:
        self.errors = errors
        super().__init__("plugin cleanup failed: " + "; ".join(str(error) for error in errors))
