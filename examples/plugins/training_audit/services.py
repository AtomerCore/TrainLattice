"""Alternative service behavior expressed through the public structural APIs."""


class AuditState:
    """Versioned evidence that selected trainer and persistence services ran."""

    state_schema_version = 1

    def __init__(self):
        self.counts = {name: 0 for name in ("trainer_create", "fit", "store_create", "preflight", "save", "restore")}

    def snapshot_state(self):
        """Return an independent record of observed service calls."""
        return dict(self.counts)

    def restore_state(self, payload):
        """Restore counters before recording the new restore operation."""
        self.counts = dict(payload)


class AuditedTrainer:
    """A trainer service that records execution around a delegated lifecycle."""

    def __init__(self, trainer, audit):
        self.trainer = trainer
        self.audit = audit

    @property
    def state(self):
        """Expose the trainer's mutable recovery state."""
        return self.trainer.state

    @state.setter
    def state(self, value):
        self.trainer.state = value

    @property
    def scaler(self):
        """Expose mixed-precision state for checkpoint restoration."""
        return self.trainer.scaler

    def fit(self, train_loader, validation_loader):
        """Record the selected lifecycle service's invocation."""
        self.audit.counts["fit"] += 1
        return self.trainer.fit(train_loader, validation_loader)


class AuditedStore:
    """A checkpoint service recording preflight, persistence, and recovery."""

    def __init__(self, store, audit):
        self.store, self.audit = store, audit

    def preflight(self, path, **options):
        """Inspect a recovery source through the selected persistence service."""
        self.audit.counts["preflight"] += 1
        return self.store.preflight(path, **options)

    def save(self, name, **state):
        """Record and atomically persist a snapshot."""
        self.audit.counts["save"] += 1
        return self.store.save(name, **state)

    def restore(self, payload, **objects):
        """Restore the graph's state before advancing its restore counter."""
        state = self.store.restore(payload, **objects)
        self.audit.counts["restore"] += 1
        return state


class AuditedCheckpointFactory:
    """Bind audited stores to the application's frozen graph."""

    def __init__(self, factory, audit):
        self.factory, self.audit = factory, audit

    def create(self, directory, **options):
        """Construct the selected store and record its creation."""
        self.audit.counts["store_create"] += 1
        return AuditedStore(self.factory.create(directory, **options), self.audit)
