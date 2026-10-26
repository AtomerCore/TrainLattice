from typing import Any

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import TensorDataset

from trainlattice.contracts.config import DatasetConfig
from trainlattice.errors import ConfigError
from trainlattice.builtin_plugins.datasets_basic.loaders import EpochShuffleSampler
from trainlattice.contracts.components import ComponentFactory
from helpers import get_component, build_dataloaders


def test_synthetic_loaders_expose_distinct_splits() -> None:
    config = DatasetConfig(
        name="synthetic",
        batch_size=2,
        options={
            "samples": 8,
            "val_samples": 4,
            "input_shape": [3],
            "num_classes": 2,
        },
    )

    loaders = build_dataloaders(config, seed=7)

    assert len(loaders.train.dataset) == 8
    assert loaders.validation is not None and len(loaders.validation.dataset) == 4


def test_synthetic_rejects_removed_test_split_option() -> None:
    config = DatasetConfig(
        name="synthetic",
        batch_size=2,
        options={"test_samples": 6},
    )

    with pytest.raises(ConfigError, match="dataset.test_samples"):
        build_dataloaders(config, seed=7)


def test_epoch_sampler_order_depends_only_on_seed_and_epoch() -> None:
    dataset = TensorDataset(torch.arange(12))
    first = EpochShuffleSampler(dataset, seed=3)
    resumed = EpochShuffleSampler(dataset, seed=3)

    first.set_epoch(1)
    resumed.set_epoch(1)

    assert list(first) == list(resumed)
    first.set_epoch(0)
    assert list(first) != list(resumed)


def test_dataset_factory_builds_train_and_validation() -> None:
    name = "unit_split_dataset"

    def build_unit_dataset(*, train: bool, split: str | None = None, **_: Any) -> TensorDataset:
        sizes = {"train": 8, "validation": 2}
        values = torch.arange(sizes[split or "train"])
        return TensorDataset(values.float().unsqueeze(1), values)

    loaders = build_dataloaders(DatasetConfig(name=name, batch_size=2), seed=5, factory=ComponentFactory(name, build_unit_dataset))

    assert len(loaders.train.dataset) == 8
    assert loaders.validation is not None and len(loaders.validation.dataset) == 2


def _write_tensor_split(directory, name: str, count: int, num_classes: int = 3) -> None:
    generator = torch.Generator().manual_seed(11)
    torch.save(
        {
            "features": torch.randn(count, 1, 4, 4, generator=generator),
            "labels": torch.randint(0, num_classes, (count,), generator=generator),
        },
        directory / name,
    )


def test_tensor_loaders_read_split_files(tmp_path) -> None:
    _write_tensor_split(tmp_path, "train.pt", 8)
    _write_tensor_split(tmp_path, "validation.pt", 4)
    config = DatasetConfig(name="tensor", batch_size=2, options={"root": str(tmp_path), "num_classes": 3})

    loaders = build_dataloaders(config, seed=7)

    assert len(loaders.train.dataset) == 8
    assert loaders.validation is not None and len(loaders.validation.dataset) == 4


def test_tensor_contract_states_on_disk_shape(tmp_path) -> None:
    _write_tensor_split(tmp_path, "train.pt", 8)
    _write_tensor_split(tmp_path, "validation.pt", 4)
    contract = get_component("dataset", "tensor").contract

    assert contract is not None
    declared = contract({"root": str(tmp_path), "num_classes": 3})
    assert declared.input_shape == (1, 4, 4)
    assert declared.input_dim is None
    assert declared.num_classes == 3


def test_tensor_reports_missing_root_and_files(tmp_path) -> None:
    config = DatasetConfig(name="tensor", batch_size=2, options={"root": str(tmp_path / "missing")})
    with pytest.raises(ConfigError, match="dataset.root directory not found"):
        build_dataloaders(config, seed=7)

    empty_root = tmp_path / "data"
    empty_root.mkdir()
    with pytest.raises(ConfigError, match="dataset.train_file file not found"):
        get_component("dataset", "tensor").contract({"root": str(empty_root)})


def test_tensor_rejects_unreadable_and_inconsistent_files(tmp_path) -> None:
    (tmp_path / "train.pt").write_text("not a tensor file", encoding="utf-8")
    contract = get_component("dataset", "tensor").contract

    assert contract is not None
    with pytest.raises(ConfigError, match="dataset.train_file file could not be read"):
        contract({"root": str(tmp_path)})

    torch.save({"features": torch.randn(4, 3), "labels": torch.zeros(3, dtype=torch.long)}, tmp_path / "train.pt")
    with pytest.raises(ConfigError, match="dataset.train_file tensors are inconsistent"):
        contract({"root": str(tmp_path)})


def test_tensor_rejects_labels_outside_class_range(tmp_path) -> None:
    torch.save(
        {"features": torch.randn(4, 3), "labels": torch.tensor([0, 1, 2, 9])},
        tmp_path / "train.pt",
    )
    config = DatasetConfig(name="tensor", batch_size=2, options={"root": str(tmp_path), "num_classes": 3})

    with pytest.raises(ConfigError, match="dataset.num_classes must exceed the maximum label"):
        build_dataloaders(config, seed=7)


def test_tensor_rejects_unknown_options(tmp_path) -> None:
    config = DatasetConfig(name="tensor", batch_size=2, options={"root": str(tmp_path), "samples": 5})

    with pytest.raises(ConfigError, match="dataset.samples"):
        build_dataloaders(config, seed=7)


def test_tensor_rejects_empty_split_files(tmp_path) -> None:
    torch.save({"features": torch.randn(0, 3), "labels": torch.zeros(0, dtype=torch.long)}, tmp_path / "train.pt")

    with pytest.raises(ConfigError, match="dataset.train_file file must contain at least one sample"):
        get_component("dataset", "tensor").contract({"root": str(tmp_path)})


def test_tensor_rejects_non_long_and_negative_labels(tmp_path) -> None:
    torch.save(
        {"features": torch.randn(4, 3), "labels": torch.tensor([0.0, 1.0, 2.0, 1.0])},
        tmp_path / "train.pt",
    )
    with pytest.raises(ConfigError, match="dataset.train_file labels must be a Long"):
        get_component("dataset", "tensor").contract({"root": str(tmp_path)})

    torch.save(
        {"features": torch.randn(4, 3), "labels": torch.tensor([-1, 1, 2, 1], dtype=torch.long)},
        tmp_path / "train.pt",
    )
    with pytest.raises(ConfigError, match="dataset.train_file labels must be non-negative"):
        get_component("dataset", "tensor").contract({"root": str(tmp_path)})


def test_tensor_rejects_train_validation_shape_disagreement(tmp_path) -> None:
    torch.save(
        {"features": torch.randn(8, 1, 4, 4), "labels": torch.randint(0, 3, (8,))},
        tmp_path / "train.pt",
    )
    torch.save(
        {"features": torch.randn(4, 1, 8, 8), "labels": torch.randint(0, 3, (4,))},
        tmp_path / "validation.pt",
    )

    with pytest.raises(ConfigError, match=r"dataset.val_file feature shape \[1, 8, 8\] must match"):
        get_component("dataset", "tensor").contract({"root": str(tmp_path)})
