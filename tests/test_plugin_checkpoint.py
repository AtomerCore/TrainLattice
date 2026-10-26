"""Checkpoint graph compatibility, plugin state, and atomic recovery."""

import random
from pathlib import Path

import pytest
torch = pytest.importorskip("torch")

from trainlattice.application import build_application
from trainlattice.contracts.state import TrainerState
from trainlattice.errors import CheckpointError
from trainlattice.kernel.snapshot import PluginSnapshot


class Counter:
    state_schema_version = 1

    def __init__(self, value=0):
        self.value = value

    def snapshot_state(self):
        return {"value": self.value}

    def restore_state(self, payload):
        self.value = payload["value"]
        random.random()
        torch.rand(1)


def make_store(tmp_path, providers=None):
    app = build_application({"profile": "supervised-basic"})
    with app:
        return app.require("trainlattice.checkpoint/local").create(
            tmp_path / "checkpoints", snapshot=app.snapshot, state_providers=providers or {})


def objects():
    model = torch.nn.Linear(2, 2)
    return model, torch.optim.Adam(model.parameters(), lr=0.1)


def test_checkpoint_restores_plugin_state_and_rng_last(tmp_path: Path):
    provider = Counter(8)
    store = make_store(tmp_path, {"counter": provider})
    model, optimizer = objects()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    path = store.save("last.pt", model=model, optimizer=optimizer,
                      trainer_state=TrainerState(next_epoch=2), config={})
    expected = (random.random(), torch.rand(2))
    provider.value = 0
    restored_model, restored_optimizer = objects()
    state = store.load(path, model=restored_model, optimizer=restored_optimizer)
    assert state.next_epoch == 2 and provider.value == 8
    assert random.random() == expected[0] and torch.equal(torch.rand(2), expected[1])
    assert restored_optimizer.state_dict()["state"]
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), restored_model.parameters()))
    assert not list(store.directory.glob("*.tmp"))


def test_checkpoint_preflight_rejects_incomplete_or_incompatible_state(tmp_path: Path):
    provider = Counter()
    store = make_store(tmp_path, {"counter": provider})
    model, optimizer = objects()
    path = store.save("last.pt", model=model, optimizer=optimizer, trainer_state=TrainerState(), config={})
    payload = torch.load(path, weights_only=True)
    for key, changed, message in [
        ("format_version", 4, "format version"),
        ("resume_safe", False, "diagnostic emergency"),
        ("plugin_states", {}, "state providers"),
        ("plugin_states", {"counter": {"schema_version": 2, "payload": {}}}, "schema mismatch"),
        ("trainer_state", {}, "trainer state fields"),
    ]:
        torch.save({**payload, key: changed}, path)
        with pytest.raises(CheckpointError, match=message):
            store.preflight(path)
    changed_snapshot = store.snapshot.to_dict()
    changed_snapshot["plugins"][0]["fingerprint"] = "changed"
    torch.save({**payload, "plugins_snapshot": changed_snapshot}, path)
    with pytest.raises(CheckpointError, match="fingerprint mismatch"):
        store.preflight(path)


def test_checkpoint_identity_supports_relocation_and_atomic_save_failure(tmp_path: Path, monkeypatch):
    store = make_store(tmp_path)
    assert not store.directory.exists()
    model, optimizer = objects()
    path = store.save("last.pt", model=model, optimizer=optimizer, trainer_state=TrainerState(), config={})
    original = path.read_bytes()
    relocated = store.snapshot.to_dict()
    for plugin in relocated["plugins"]:
        plugin["path"] = "other/root"
        plugin["source"] = "other/source"
    store.snapshot = PluginSnapshot(relocated)
    store.preflight(path)
    def fail_save(payload, temporary):
        Path(temporary).write_bytes(b"partial")
        raise OSError("disk full")
    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(CheckpointError, match="disk full"):
        store.save("last.pt", model=model, optimizer=optimizer, trainer_state=TrainerState(), config={})
    assert path.read_bytes() == original
    assert list(store.directory.iterdir()) == [path]
