from typing import Any

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import TensorDataset

from trainlattice.config.schema import DatasetConfig
from trainlattice.errors import ConfigError
from trainlattice.datasets import DATASETS, EpochShuffleSampler, build_dataloaders


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


def test_registered_dataset_extension_point_builds_train_and_validation() -> None:
    name = "unit_split_dataset"

    @DATASETS.register(name)
    def build_unit_dataset(*, train: bool, split: str | None = None, **_: Any) -> TensorDataset:
        sizes = {"train": 8, "validation": 2}
        values = torch.arange(sizes[split or "train"])
        return TensorDataset(values.float().unsqueeze(1), values)

    loaders = build_dataloaders(DatasetConfig(name=name, batch_size=2), seed=5)

    assert len(loaders.train.dataset) == 8
    assert loaders.validation is not None and len(loaders.validation.dataset) == 2
