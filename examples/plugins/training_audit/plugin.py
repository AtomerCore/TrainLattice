"""Activate alternate trainer and persistence service implementations."""


def activate(ctx):
    """Decorate declared services and checkpoint their call counters."""
    from .services import AuditState, AuditedTrainer, AuditedCheckpointFactory
    state = AuditState()
    trainer_factory = ctx.require("trainer")

    def create_trainer(**options):
        state.counts["trainer_create"] += 1
        return AuditedTrainer(trainer_factory(**options), state)

    ctx.provide("example.trainer/audited", create_trainer)
    ctx.provide("example.checkpoint/audited", AuditedCheckpointFactory(ctx.require("checkpoint"), state))
    return state
