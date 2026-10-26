"""An epoch observer with checkpointed extension state."""

from trainlattice.contracts.callbacks import Callback


class EpochCounter(Callback):
    """Count epochs across interruption and native checkpoint recovery."""

    state_schema_version = 1

    def __init__(self):
        self.epochs = 0

    def on_epoch_end(self, trainer):
        """Advance before the checkpoint observer snapshots plugin state."""
        self.epochs += 1

    def snapshot_state(self):
        """Serialize the extension's current epoch count."""
        return {"epochs": self.epochs}

    def restore_state(self, payload):
        """Restore the extension's versioned epoch count."""
        self.epochs = payload["epochs"]
