"""Independent metrics, logging ownership, and lifecycle service contracts."""

from types import SimpleNamespace
import json

import pytest
torch = pytest.importorskip("torch")

from trainlattice.application import build_application
from trainlattice.contracts.state import TrainerState
from trainlattice.builtin_plugins.logging_basic.loggers import CompositeLogger
from trainlattice.kernel.errors import CleanupError


def test_metric_services_weight_uneven_batches_by_samples():
    with build_application({"profile": "supervised-basic"}) as app:
        create = app.require("trainlattice.metrics/basic")
        totals = create("classification")
        totals.update(torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([0, 1]), torch.tensor(1.))
        totals.update(torch.tensor([[0., 1.]]), torch.tensor([0]), torch.tensor(4.))
        assert totals.compute("training") == {"loss": 2., "accuracy": 2 / 3}
        regression = create("regression")
        regression.update(torch.ones(2, 1), torch.ones(2, 1), torch.tensor(3.))
        assert regression.compute("training") == {"loss": 3.}


def test_logger_services_close_resources_after_failure(tmp_path):
    with build_application({"profile": "supervised-basic"}) as app:
        with pytest.raises(RuntimeError, match="run failed"):
            with app.run_scope():
                logger = app.require("trainlattice.logging/basic")(tmp_path)
                logger.log({"loss": 1.}, 2)
                raise RuntimeError("run failed")
        assert logger.loggers[1]._stream.closed
        assert json.loads((tmp_path / "metrics.jsonl").read_text())["step"] == 2
    calls = []
    class Sink:
        def __init__(self, fail=False): self.fail = fail
        def close(self):
            calls.append(self.fail)
            if self.fail: raise OSError("close failed")
    composite = CompositeLogger([Sink(True), Sink()])
    with pytest.raises(CleanupError): composite.close()
    composite.close()
    assert calls == [True, False]


def test_callback_factory_orders_observers_and_saves_fresh_best_metric():
    events = []
    class Logger:
        def log(self, metrics, step): events.append("log")
    class Observer:
        def on_epoch_end(self, trainer): events.append("user")
    class Store:
        def save(self, name, **state):
            events.append((name, state["trainer_state"].best_metric))
    trainer = SimpleNamespace(state=TrainerState(latest_metrics={"val_loss": 2.}),
                              model=None, optimizer=None, scheduler=None, scaler=None)
    with build_application({"profile": "supervised-basic"}) as app:
        callbacks = app.require("trainlattice.callbacks/basic")(
            logger=Logger(), checkpoint=Store(), config={}, log_every=1, monitor="val_loss", extra=[Observer()])
        for callback in callbacks: callback.on_epoch_end(trainer)
    assert events == ["log", "user", ("last.pt", 2.), ("best.pt", 2.)]
