from pathlib import Path

import pytest

from trainlattice.config import ComponentContract, compare_contracts, configs_compatible, load_config
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
    with pytest.raises(ConfigError, match="optimizer.name"):
        load_config(write_config(tmp_path, BASE_CONFIG.replace("lr: 0.01", "name: adamw\n  lr: 0.01")))


def test_rejects_invalid_scalar_types(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="trainer.epochs must be an integer"):
        load_config(write_config(tmp_path, BASE_CONFIG.replace("epochs: 2", 'epochs: "two"')))


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
