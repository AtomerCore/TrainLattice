from pathlib import Path

import pytest

from trainlattice.contracts.components import ComponentContract, compare_contracts
from trainlattice.contracts.config import ExperimentConfig
from trainlattice.builtin_plugins.checkpoint_local.config import configs_compatible
from helpers import load_config
from trainlattice.errors import ConfigError


BASE_CONFIG = """
experiment:
  name: unit-test
model:
  name: mlp
  input_dim: 4
dataset:
  name: synthetic
  batch_size: 2
  input_shape: [4]
optimizer:
  lr: 0.01
trainer:
  epochs: 2
"""


def write_config(tmp_path: Path, text: str = BASE_CONFIG) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_yaml_into_typed_sections(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))

    assert config.experiment.name == "unit-test"
    assert config.model.name == "mlp"
    assert config.model.options["input_dim"] == 4
    assert config.dataset.name == "synthetic"
    assert config.optimizer.lr == 0.01
    assert config.trainer.epochs == 2
    assert config.trainer.accumulate == 1


def test_rejects_invalid_batch_size(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="dataset.batch_size"):
        load_config(write_config(tmp_path, BASE_CONFIG.replace("batch_size: 2", "batch_size: 0")))


def test_rejects_non_positive_learning_rate(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="optimizer.lr must be a positive number"):
        load_config(write_config(tmp_path, BASE_CONFIG.replace("lr: 0.01", "lr: 0")))


def test_rejects_unknown_sections_and_fixed_fields(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown configuration section.*trainr"):
        load_config(write_config(tmp_path, BASE_CONFIG + "\ntrainr:\n  epochs: 9\n"))
    with pytest.raises(ConfigError, match="trainer.epochz"):
        load_config(write_config(tmp_path, BASE_CONFIG + "\ntrainer:\n  epochz: 9\n"))
    with pytest.raises(ConfigError, match="optimizer.name must be a string"):
        load_config(write_config(tmp_path, BASE_CONFIG.replace("lr: 0.01", "name: [adam]\n  lr: 0.01")))


def test_loads_optimizer_and_loss_sections(tmp_path: Path) -> None:
    config = load_config(
        write_config(
            tmp_path,
            BASE_CONFIG.replace("lr: 0.01", "name: sgd\n  lr: 0.01\n  momentum: 0.9")
            + "loss:\n  name: mse\n",
        )
    )

    assert config.optimizer.name == "sgd"
    assert config.optimizer.options == {"momentum": 0.9}
    assert config.loss.name == "mse"
    assert config.loss.options == {}

    defaulted = load_config(write_config(tmp_path))
    assert defaulted.optimizer.name == "adamw"
    assert defaulted.loss.name == "cross_entropy"

    resolved = config.to_dict()
    assert resolved["optimizer"] == {"name": "sgd", "lr": 0.01, "momentum": 0.9}
    assert resolved["loss"] == {"name": "mse"}
    assert defaulted.to_dict()["loss"] == {"name": "cross_entropy"}


def test_rejects_invalid_scalar_types(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="trainer.epochs must be an integer"):
        load_config(write_config(tmp_path, BASE_CONFIG.replace("epochs: 2", 'epochs: "two"')))


def test_rejects_device_outside_supported_domain(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="trainer.device must be one of"):
        load_config(write_config(tmp_path, BASE_CONFIG.replace("epochs: 2", "epochs: 2\n  device: tpu")))


def test_rejects_invalid_accumulate(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="trainer.accumulate must be an integer"):
        load_config(write_config(tmp_path, BASE_CONFIG.replace("epochs: 2", "epochs: 2\n  accumulate: 1.5")))
    with pytest.raises(ConfigError, match="trainer.accumulate must be an integer"):
        load_config(write_config(tmp_path, BASE_CONFIG.replace("epochs: 2", "epochs: 2\n  accumulate: true")))
    with pytest.raises(ConfigError, match="trainer.accumulate must be at least 1"):
        load_config(write_config(tmp_path, BASE_CONFIG.replace("epochs: 2", "epochs: 2\n  accumulate: 0")))


def test_rejects_invalid_monitor_and_grad_norm(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="trainer.monitor must be one of"):
        load_config(
            write_config(tmp_path, BASE_CONFIG.replace("epochs: 2", "epochs: 2\n  monitor: train_loss"))
        )
    with pytest.raises(ConfigError, match="trainer.monitor must be one of"):
        load_config(
            write_config(tmp_path, BASE_CONFIG.replace("epochs: 2", "epochs: 2\n  monitor: [val_loss]"))
        )
    with pytest.raises(ConfigError, match="trainer.max_grad_norm must be a positive number"):
        load_config(
            write_config(tmp_path, BASE_CONFIG.replace("epochs: 2", "epochs: 2\n  max_grad_norm: 0"))
        )
    with pytest.raises(ConfigError, match="trainer.max_grad_norm must be a positive number"):
        load_config(
            write_config(tmp_path, BASE_CONFIG.replace("epochs: 2", "epochs: 2\n  max_grad_norm: .nan"))
        )
    with pytest.raises(ConfigError, match="optimizer.lr must be a positive number"):
        load_config(write_config(tmp_path, BASE_CONFIG.replace("lr: 0.01", "lr: .nan")))


def test_reports_directory_paths_explicitly(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="configuration path is a directory, not a file"):
        load_config(tmp_path)


def test_reports_empty_files_explicitly(tmp_path: Path) -> None:
    config = tmp_path / "empty.yaml"
    config.write_text("", encoding="utf-8")

    with pytest.raises(ConfigError, match="configuration file is empty"):
        load_config(config)


def test_contract_comparison_reports_both_mismatch_kinds() -> None:
    model = ComponentContract(input_dim=4, num_classes=2)
    dataset = ComponentContract(input_dim=5, num_classes=10)

    errors = compare_contracts(model, dataset)

    assert errors == [
        "model.input_dim (4) must match dataset.input_shape product (5)",
        "model.num_classes (2) must match dataset.num_classes (10)",
    ]


def test_contract_comparison_accepts_matching_contracts() -> None:
    assert compare_contracts(ComponentContract(784, 10), ComponentContract(784, 10)) == []


def test_contract_comparison_skips_unstated_input_dim() -> None:
    stated = ComponentContract(input_dim=None, num_classes=3)

    assert compare_contracts(ComponentContract(4, 3), stated) == []


def test_contract_comparison_reports_structured_shape_mismatch() -> None:
    model = ComponentContract(input_dim=None, num_classes=10, input_shape=(1, 28, 28))
    dataset = ComponentContract(input_dim=3072, num_classes=10, input_shape=(3, 32, 32))

    errors = compare_contracts(model, dataset)

    assert errors == [
        "model.input_shape ([1, 28, 28]) must match dataset.input_shape ([3, 32, 32])"
    ]


def test_contract_comparison_crosses_input_dim_against_shape_product() -> None:
    model = ComponentContract(input_dim=100, num_classes=10)
    dataset = ComponentContract(input_dim=None, num_classes=10, input_shape=(1, 28, 28))

    errors = compare_contracts(model, dataset)

    assert errors == [
        "model.input_dim (100) must match dataset.input_shape product (784)"
    ]


def test_contract_comparison_accepts_matching_structured_shapes() -> None:
    model = ComponentContract(input_dim=None, num_classes=10, input_shape=(1, 28, 28))
    dataset = ComponentContract(input_dim=None, num_classes=10, input_shape=(1, 28, 28))

    assert compare_contracts(model, dataset) == []


def test_recovery_comparison_allows_runtime_only_changes(tmp_path: Path) -> None:
    first = load_config(write_config(tmp_path))
    second = load_config(
        write_config(
            tmp_path,
            BASE_CONFIG.replace("epochs: 2", "epochs: 10")
            .replace("trainer:\n", "trainer:\n  device: cpu\n  log_every: 99\n")
            .replace("name: unit-test", "name: unit-test\n  runs_dir: elsewhere"),
        )
    )
    changed_model = load_config(
        write_config(tmp_path, BASE_CONFIG.replace("input_dim: 4", "input_dim: 4\n  hidden_dims: [16]"))
    )
    changed_lr = load_config(write_config(tmp_path, BASE_CONFIG.replace("lr: 0.01", "lr: 0.02")))

    assert configs_compatible(first, second)
    assert not configs_compatible(first, changed_model)
    assert not configs_compatible(first, changed_lr)


def test_loads_nested_early_stopping_section(tmp_path: Path) -> None:
    config = load_config(
        write_config(
            tmp_path,
            BASE_CONFIG.replace(
                "epochs: 2",
                "epochs: 2\n  early_stopping:\n    patience: 3\n    min_delta: 0.05",
            ),
        )
    )

    assert config.trainer.early_stopping is not None
    assert config.trainer.early_stopping.patience == 3
    assert config.trainer.early_stopping.min_delta == 0.05
    assert config.to_dict()["trainer"]["early_stopping"] == {"patience": 3, "min_delta": 0.05}
    assert ExperimentConfig.from_dict(config.to_dict()) == config

    defaulted = load_config(
        write_config(tmp_path, BASE_CONFIG.replace("epochs: 2", "epochs: 2\n  early_stopping:\n    patience: 2"))
    )
    assert defaulted.trainer.early_stopping is not None
    assert defaulted.trainer.early_stopping.min_delta == 0.0

    disabled = load_config(
        write_config(tmp_path, BASE_CONFIG.replace("epochs: 2", "epochs: 2\n  early_stopping: null"))
    )
    assert disabled.trainer.early_stopping is None


def test_rejects_invalid_early_stopping(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"trainer\.early_stopping\.patiencez"):
        load_config(
            write_config(tmp_path, BASE_CONFIG.replace("epochs: 2", "epochs: 2\n  early_stopping:\n    patiencez: 2"))
        )
    with pytest.raises(ConfigError, match="trainer.early_stopping.patience must be an integer"):
        load_config(
            write_config(tmp_path, BASE_CONFIG.replace("epochs: 2", 'epochs: 2\n  early_stopping:\n    patience: "two"'))
        )
    with pytest.raises(ConfigError, match="trainer.early_stopping.patience must be at least 1"):
        load_config(
            write_config(tmp_path, BASE_CONFIG.replace("epochs: 2", "epochs: 2\n  early_stopping:\n    patience: 0"))
        )
    with pytest.raises(ConfigError, match="trainer.early_stopping.min_delta must be a non-negative number"):
        load_config(
            write_config(tmp_path, BASE_CONFIG.replace("epochs: 2", "epochs: 2\n  early_stopping:\n    patience: 2\n    min_delta: -0.1"))
        )
    with pytest.raises(ConfigError, match="trainer.early_stopping.min_delta must be a non-negative number"):
        load_config(
            write_config(tmp_path, BASE_CONFIG.replace("epochs: 2", 'epochs: 2\n  early_stopping:\n    patience: 2\n    min_delta: slow'))
        )
    with pytest.raises(ConfigError, match="trainer.early_stopping must be a mapping"):
        load_config(write_config(tmp_path, BASE_CONFIG.replace("epochs: 2", "epochs: 2\n  early_stopping: 3")))
    with pytest.raises(ConfigError, match="invalid trainer.early_stopping configuration"):
        load_config(
            write_config(tmp_path, BASE_CONFIG.replace("epochs: 2", "epochs: 2\n  early_stopping:\n    min_delta: 0.1"))
        )


def test_recovery_comparison_allows_early_stopping_changes(tmp_path: Path) -> None:
    strict = load_config(
        write_config(
            tmp_path,
            BASE_CONFIG.replace("epochs: 2", "epochs: 2\n  early_stopping:\n    patience: 2\n    min_delta: 0.1"),
        )
    )
    relaxed = load_config(
        write_config(tmp_path, BASE_CONFIG.replace("epochs: 2", "epochs: 2\n  early_stopping:\n    patience: 9"))
    )
    without = load_config(write_config(tmp_path))
    changed_monitor = load_config(
        write_config(
            tmp_path,
            BASE_CONFIG.replace(
                "epochs: 2", "epochs: 2\n  early_stopping:\n    patience: 2\n  monitor: val_accuracy"
            ),
        )
    )

    assert configs_compatible(strict, relaxed)
    assert configs_compatible(strict, without)
    assert not configs_compatible(strict, changed_monitor)


def test_recovery_comparison_requires_resolved_training_fields() -> None:
    resolved = {"trainer": {"precision": "fp32", "accumulate": 1}, "loss": {"name": "cross_entropy"}}
    assert configs_compatible(resolved, resolved)
    assert not configs_compatible({}, resolved)
    assert not configs_compatible(resolved, {**resolved, "loss": {"name": "mse"}})
