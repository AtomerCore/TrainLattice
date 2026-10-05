"""Run a regression experiment: a custom dataset with the built-in `mse` loss.

This module registers the ``sine_wave`` dataset on import and then starts
the experiment, demonstrating that component code can live in the entry
script itself. Regression targets are float tensors shaped like the model
outputs, and the loss section selects ``mse`` — the trainer then reports
loss only, without accuracy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, TensorDataset

from trainlattice import DATASETS, ComponentContract, ConfigError, run_from_file

CONFIG = Path(__file__).parent / "regression.yaml"


def build_sine_wave(
    *,
    train: bool,
    split: str | None = None,
    samples: int = 128,
    val_samples: int = 32,
    seed: int | None = None,
    **_: Any,
) -> Dataset[Any]:
    """Build one split of noisy sine regression data.

    Targets carry the model's output shape ``(N, 1)`` as float tensors,
    which is what ``MSELoss`` compares against.
    """
    split_name = split or ("train" if train else "validation")
    counts = {"train": samples, "validation": val_samples}
    if split_name not in counts:
        raise ConfigError(f"unsupported sine_wave dataset split: {split_name}")
    offset = {"train": 0, "validation": 1}[split_name]
    generator = torch.Generator().manual_seed((seed if seed is not None else 42) + offset)
    inputs = torch.rand(int(counts[split_name]), 1, generator=generator) * 2 * math.pi
    targets = torch.sin(inputs) + torch.randn(inputs.shape, generator=generator) * 0.1
    return TensorDataset(inputs, targets)


def _validate_sine_wave(path: str, options: Mapping[str, Any]) -> None:
    """Reject unknown options and invalid sizes with dotted paths."""
    unknown = sorted(set(options) - {"samples", "val_samples"})
    if unknown:
        keys = ", ".join(f"{path}.{key}" for key in unknown)
        raise ConfigError(f"unknown configuration field(s): {keys}")
    for key, default in (("samples", 128), ("val_samples", 32)):
        value = options.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ConfigError(f"{path}.{key} must be a positive integer")


def _sine_wave_contract(options: Mapping[str, Any]) -> ComponentContract:
    """State the input width and the regression target width."""
    return ComponentContract(input_dim=1, num_classes=1, input_shape=(1,))


DATASETS.register(
    "sine_wave",
    build_sine_wave,
    validate=_validate_sine_wave,
    contract=_sine_wave_contract,
)


if __name__ == "__main__":
    result = run_from_file(CONFIG, command="python examples/regression_experiment.py")
    print(f"finished run: {result.run_dir}")
    # Resume after an interruption, continuing from the last checkpoint:
    # run_from_file(
    #     CONFIG,
    #     resume=result.run_dir / "checkpoints" / "last.pt",
    #     command="python examples/regression_experiment.py --resume",
    # )
