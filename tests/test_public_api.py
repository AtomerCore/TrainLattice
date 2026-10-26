"""Native application entries, component selection, and external model services."""

from copy import deepcopy
import json

import pytest
import yaml

pytest.importorskip("torch")

from trainlattice.application import run_experiment, run_from_file
from trainlattice.contracts.callbacks import Callback
from trainlattice.errors import ConfigError


def config(tmp_path):
    return {"profile": "supervised-basic", "experiment": {"name": "public-api", "seed": 3, "runs_dir": str(tmp_path / "runs")},
            "model": {"input_dim": 4, "hidden_dims": [8], "num_classes": 2},
            "dataset": {"input_shape": [4], "num_classes": 2, "batch_size": 4, "samples": 12, "val_samples": 6},
            "trainer": {"device": "cpu", "epochs": 1}}


def test_application_entries_preserve_callbacks_and_reloadable_artifacts(tmp_path):
    events = []
    class Observer(Callback):
        def on_epoch_end(self, trainer): events.append((trainer.state.epoch, trainer.state.best_metric))
    first = run_experiment(config(tmp_path), callbacks=[Observer()])
    assert events == [(0, None)]
    resolved = yaml.safe_load((first.run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert resolved["plugins"] and "name" not in resolved["model"]
    resolved["trainer"]["epochs"] = 2
    source = tmp_path / "resume.yaml"
    source.write_text(yaml.safe_dump(resolved), encoding="utf-8")
    resumed = run_from_file(source, resume=first.run_dir)
    assert resumed.state.next_epoch == 2
    manifest = json.loads((first.run_dir / "manifest.json").read_text())
    assert "architectures.py" in manifest["model_source"]
    assert "synthetic.py" in manifest["dataset_source"]


def test_native_options_require_bindings_and_preflight_before_artifacts(tmp_path):
    original = config(tmp_path)
    for section, key, value, message in [("model", "name", "mlp", "model.name"),
                                         ("model", "layerz", 1, "model.layerz"),
                                         ("scheduler", "total_steps", 3, "scheduler requires a service")]:
        raw = deepcopy(original)
        raw.setdefault(section, {})[key] = value
        with pytest.raises(ConfigError, match=message): run_experiment(raw)
        assert not (tmp_path / "runs").exists()


def test_local_model_service_trains_validates_and_resumes(tmp_path):
    directory = tmp_path / "本地 model"
    directory.mkdir()
    (directory / "plugin.yaml").write_text(yaml.safe_dump({"schema_version": 1, "id": "example.model", "version": "1", "api_version": 1,
        "entry": "plugin.py:activate", "provides": [{"service": "example.model/linear"}], "requires": {}, "state": {"mode": "stateless"}}), encoding="utf-8")
    (directory / "plugin.py").write_text('''def activate(ctx):
    from torch import nn
    from trainlattice.contracts.components import ComponentFactory, ComponentContract
    from trainlattice.errors import ConfigError
    def validate(path, options):
        if options: raise ConfigError(f"{path}.unexpected option")
    ctx.provide("example.model/linear", ComponentFactory("linear_custom", lambda: nn.Linear(4, 2), validate, lambda options: ComponentContract(4, 2)))
''', encoding="utf-8")
    raw = config(tmp_path)
    raw["plugins"] = [{"source": str(directory)}]
    raw["bindings"] = {"org.trainlattice.experiment.default": {"model": "example.model/linear"}}
    raw["model"] = {}
    first = run_experiment(raw)
    raw["trainer"]["epochs"] = 2
    assert run_experiment(raw, resume=first.run_dir).state.next_epoch == 2
    raw["model"]["layerz"] = 1
    with pytest.raises(ConfigError, match="model.unexpected"): run_experiment(raw)
