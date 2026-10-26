"""Real directory plugins, stateful recovery, and replaceable service proofs."""

from copy import deepcopy
from pathlib import Path
import shutil

import pytest
import yaml
torch = pytest.importorskip("torch")

from trainlattice.application import run_experiment
from trainlattice.contracts.callbacks import Callback
from trainlattice.errors import CheckpointError

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def configuration(tmp_path, name="custom.yaml"):
    raw = yaml.safe_load((EXAMPLES / name).read_text(encoding="utf-8"))
    raw["experiment"]["runs_dir"] = str(tmp_path / "runs")
    return raw


def payload(result):
    return torch.load(result.run_dir / "checkpoints" / "last.pt", weights_only=True)


def assert_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first: assert_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second): assert_equal(a, b)
    else:
        assert first == second


def test_external_components_metrics_and_plugin_state_match_epoch_resume(tmp_path):
    raw = configuration(tmp_path)
    class Stop(Callback):
        def on_epoch_end(self, trainer): trainer.request_stop("epoch boundary")
    continuous = run_experiment(raw, base_dir=EXAMPLES)
    interrupted = run_experiment(raw, base_dir=EXAMPLES, callbacks=[Stop()])
    resumed = run_experiment(raw, base_dir=EXAMPLES, resume=interrupted.run_dir)
    expected, actual = payload(continuous), payload(resumed)
    for key in ("model", "optimizer", "scheduler", "scaler", "rng_state", "plugin_states"):
        assert_equal(expected[key], actual[key])
    assert actual["plugin_states"]["example.training.extensions"]["payload"] == {"epochs": 2}
    assert "train_mean_prediction" in resumed.state.latest_metrics
    assert resumed.state.global_step == continuous.state.global_step == 8


def test_alternative_trainer_and_checkpoint_services_execute_training_and_restore(tmp_path):
    raw = configuration(tmp_path, "regression.yaml")
    raw["plugins"].append({"source": "plugins/training_audit"})
    raw["bindings"]["org.trainlattice.experiment.default"].update(
        {"trainer": "example.trainer/audited", "checkpoint": "example.checkpoint/audited"})
    raw["trainer"]["epochs"] = 1
    first = run_experiment(raw, base_dir=EXAMPLES)
    counts = payload(first)["plugin_states"]["example.training.audit"]["payload"]
    assert counts["trainer_create"] == counts["fit"] == 1
    assert counts["store_create"] == 2 and counts["save"] > 0
    raw["trainer"]["epochs"] = 2
    resumed = run_experiment(raw, base_dir=EXAMPLES, resume=first.run_dir)
    counts = payload(resumed)["plugin_states"]["example.training.audit"]["payload"]
    assert counts["fit"] == 2 and counts["restore"] == 1
    assert "train_accuracy" not in resumed.state.latest_metrics


def test_external_implementation_changes_fail_recovery_before_run_artifacts(tmp_path):
    directory = tmp_path / "中文 external plugin"
    shutil.copytree(EXAMPLES / "plugins" / "training_extensions", directory, ignore=shutil.ignore_patterns("__pycache__"))
    raw = configuration(tmp_path)
    raw["plugins"][0]["source"] = str(directory)
    first = run_experiment(raw, base_dir=EXAMPLES)
    changed = deepcopy(raw)
    changed["experiment"]["runs_dir"] = str(tmp_path / "changed-runs")
    with (directory / "components.py").open("a", encoding="utf-8") as stream:
        stream.write("\n# revision\n")
    with pytest.raises(CheckpointError, match="fingerprint mismatch.*example.training.extensions"):
        run_experiment(changed, base_dir=EXAMPLES, resume=first.run_dir)
    assert not (tmp_path / "changed-runs").exists()
