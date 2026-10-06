# TrainLattice

English | [简体中文](README.zh-CN.md)

Minimal infrastructure for deep learning training.

**Build. Run. Recover. Reproduce.**

TrainLattice is a lightweight, transparent PyTorch training runtime for studying deep learning infrastructure from first principles. One strictly validated YAML file becomes a reproducible training run: a registered architecture (the built-in `mlp` or `cnn`, or a custom one registered from Python) on a registered dataset (the built-in `synthetic` or `tensor`, or a custom one), a registered optimizer (the built-in `adamw`, `adam`, or `sgd`) and loss (the built-in `cross_entropy`, `mse`, or `l1`), lifecycle callbacks, structured metrics, atomic checkpoints, explicit resume, and self-contained run artifacts.

## Features

- Strict YAML experiment configuration: unknown sections and fields are rejected with dotted-path errors before a run directory is created
- Registry-based component construction; the `mlp` and `cnn` architecture registries and the `synthetic` and `tensor` dataset registries remain visible extension points
- Selectable optimizers (`adamw`, `adam`, `sgd` with their hyperparameters) and losses (`cross_entropy`, `mse`, `l1`) through the same registry mechanism; losses declare a task kind, so regression runs report loss without accuracy
- Custom architectures and datasets through the same registries: importing a module registers its components, experiments start from Python via `run_experiment`/`run_from_file`, custom callbacks observe the lifecycle through `extra_callbacks`, and run manifests record where each component's code lives
- Configurable MLP with `input_dim`, `hidden_dims`, and `num_classes`; GELU activations and 0.1 dropout
- Configurable CNN with `input_shape`, `channels`, and `num_classes`; `Conv2d -> GELU -> MaxPool` blocks with a linear head
- Deterministic synthetic train and validation splits plus epoch-aware shuffle sampling
- File-backed `tensor` dataset reading `train.pt`/`validation.pt` tensor pairs from a local directory; nothing is downloaded
- Explicit `Trainer` lifecycle and serializable `TrainerState`
- Console and JSONL structured metric logging with sample-weighted loss and accuracy
- Structured diagnostic logging in `[LEVEL] logger.name: message` format: INFO and below go to stdout, WARNING and above go to stderr, Python warnings are captured into the same format, and `--verbose` enables debug-level diagnostics
- Atomic checkpoints (format v3) holding model, optimizer, trainer state, resolved configuration, and RNG state: `last.pt` after every epoch, `best.pt` whenever the monitored validation metric improves, plus a diagnostic `emergency.pt` when training fails
- Optional gradient clipping through `trainer.max_grad_norm`
- Explicit `--resume PATH` recovery from the exact completed-epoch boundary with configuration compatibility checks
- Self-contained run artifacts: `config.yaml`, `environment.json`, `manifest.json`, `metrics.jsonl`, and `checkpoints/`
- CPU and single-CUDA-device execution with process-wide seeding

## Install

TrainLattice requires Python 3.10 or newer.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

Core dependencies are PyYAML and PyTorch. The synthetic dataset is generated in memory; nothing is downloaded.

## Quick Start

Run the example experiment:

```powershell
python -m trainlattice train configs/synthetic_mlp.yaml
```

The same flow runs the convolutional example, `configs/synthetic_cnn.yaml`, unchanged. Metrics stream to the console and to `metrics.jsonl` in the new run directory. When the run finishes, the CLI prints the directory, for example `Run finished: runs/20261004-120000_synthetic-mlp-smoke`.

Diagnostic logging is a separate channel from the metric stream. Records use the `[LEVEL] logger.name: message` format: INFO and below share stdout with the metric and result output, while WARNING and above (including Python warnings captured into the same format) go to stderr, so stderr can be archived separately. Add `--verbose` to enable debug-level diagnostics:

```powershell
python -m trainlattice train configs/synthetic_mlp.yaml --verbose
```

Resume that run from its latest completed epoch:

```powershell
python -m trainlattice train configs/synthetic_mlp.yaml --resume runs/20261004-120000_synthetic-mlp-smoke
```

`--resume` accepts a run directory (its `checkpoints/last.pt` is used, or a `last.pt` directly inside it) or a checkpoint file path. Each invocation creates a fresh run directory and records the resume source in its manifest.

## Runs and Recovery

Each invocation creates an isolated, collision-safe directory under `runs/`:

```text
runs/20261004-120000_synthetic-mlp-smoke/
  config.yaml
  manifest.json
  environment.json
  metrics.jsonl
  checkpoints/
    last.pt
    best.pt
    emergency.pt
```

- `config.yaml` — the resolved configuration snapshot for the run.
- `manifest.json` — command, Git commit, seed, status, timestamps, epoch, global step, resume source, and the module/file that defined the model, dataset, optimizer, and loss (`model_source`, `dataset_source`, `optimizer_source`, `loss_source`); updated atomically through `created`, `running`, `finished`, and `failed` states.
- `environment.json` — Python, PyTorch, CUDA, platform, and selected device information.
- `metrics.jsonl` — append-only metric records, one JSON object per line.
- `checkpoints/last.pt` — written atomically after every completed epoch; the resume source.
- `checkpoints/best.pt` — written whenever the monitored validation metric (`trainer.monitor`, default `val_loss`) improves on the best value seen so far, including across resumes.
- `checkpoints/emergency.pt` — diagnostic snapshot written when an exception interrupts training; rejected for resume.

Resume restores the model, optimizer, trainer state, and available RNG state (Python, Torch, CUDA, NumPy when present), then continues at the next epoch boundary with the same sample order as an uninterrupted run. Recovery-sensitive configuration values must match the checkpoint; runtime-only settings (`trainer.device`, `trainer.log_every`, `trainer.epochs`, and `experiment.runs_dir`) may change.

## Configuration

`configs/synthetic_mlp.yaml` shows the complete supported surface:

```yaml
experiment:
  name: synthetic-mlp-smoke
  seed: 42
  runs_dir: runs

model:
  name: mlp
  input_dim: 784
  hidden_dims: [64, 32]
  num_classes: 10

dataset:
  name: synthetic
  batch_size: 32
  num_workers: 0
  samples: 128
  val_samples: 64
  input_shape: [1, 28, 28]
  num_classes: 10

optimizer:
  lr: 0.001

trainer:
  epochs: 2
  device: auto
  validate_every: 1
  log_every: 2
```

Sections and fields:

- `experiment` — `name` (default `experiment`), `seed` (default `42`), `runs_dir` (default `runs`).
- `model` — `name` selects the registered architecture: the built-ins `mlp` and `cnn`, or a custom one (see [Custom Components](#custom-components)). The `mlp` options are `input_dim` (default `784`), `hidden_dims` (default `[512, 256, 128]`), and `num_classes` (default `10`); hidden layers use GELU and 0.1 dropout. The `cnn` options are `input_shape` (default `[1, 28, 28]`), `channels` (default `[32, 64]`), and `num_classes` (default `10`); each channel block is `Conv2d(kernel 3, padding 1) -> GELU -> MaxPool(2)` followed by a linear head, and the spatial dimensions must stay divisible by the total pooling factor.
- `dataset` — `name` selects the registered dataset: the built-ins `synthetic` and `tensor`, or a custom one; `batch_size` (default `64`), `num_workers` (default `0`). The `synthetic` options are `samples` (default `256`), `val_samples` (default `64`), `input_shape` (default `[1, 28, 28]`), and `num_classes` (default `10`). The `tensor` options are `root` (required), `train_file` (default `train.pt`), `val_file` (default `validation.pt`), and `num_classes` (default `10`); each file must hold a `{"features": Tensor, "labels": Tensor}` pair with matching lengths and labels below `num_classes`.
- `optimizer` — `name` (default `adamw`) selects the registered optimizer: `adamw` or `adam` with options `betas`, `eps`, `weight_decay`, `amsgrad`; `sgd` with options `momentum`, `dampening`, `weight_decay`, `nesterov`; `lr` (default `0.001`) applies to every optimizer. Omitting the section keeps AdamW with the configured learning rate.
- `loss` — `name` (default `cross_entropy`) selects the registered loss: `cross_entropy` (classification; options `label_smoothing`, `ignore_index`, `reduction`) or `mse`/`l1` (regression; option `reduction`). Metrics aggregate the per-sample mean regardless of `reduction`; `sum` keeps its larger gradient scale. A regression loss makes the trainer report loss only — `trainer.monitor` must then stay `val_loss` — and dataset targets must be float tensors shaped like the model outputs. Omitting the section keeps cross entropy.
- `trainer` — `epochs` (default `1`), `device` (`auto`, `cpu`, `cuda`, or `cuda:N`), `validate_every` (default `1`), `log_every` (default `10`), `monitor` (`val_loss` or `val_accuracy`, default `val_loss`; drives `best.pt`), and `max_grad_norm` (default `null`, disabled; clips gradients by total norm before the optimizer step).

Two cross-field contracts are enforced between the model and dataset: the input dimensions must agree — an exact `input_shape` match when both sides state one, otherwise the flattened product — and the model's output width must equal the dataset's declared width. For classification this width is the class count (`num_classes` on both sides); for regression it is the target width (a scalar target declares `num_classes: 1` on both sides). Unknown sections, fields, and component options are rejected with dotted paths such as `optimizer.weight_decay` or `model.kernel_size`.

The `tensor` dataset reads plain local files. Generate an example pair once with PyTorch and train on it:

```python
from pathlib import Path

import torch

root = Path("data/tensor")
root.mkdir(parents=True, exist_ok=True)
generator = torch.Generator().manual_seed(0)
for name, count in (("train.pt", 256), ("validation.pt", 64)):
    torch.save(
        {
            "features": torch.randn(count, 1, 28, 28, generator=generator),
            "labels": torch.randint(0, 10, (count,), generator=generator),
        },
        root / name,
    )
```

```powershell
python -m trainlattice train configs/tensor_mlp.yaml
```

## Custom Components

Architectures, datasets, optimizers, and losses all register through the same registries the built-ins use — importing a module is what registers its components, and experiments start from Python. The smallest custom experiment:

```python
from pathlib import Path

import my_model  # defines Logistic and registers "logistic"

from trainlattice import run_from_file

result = run_from_file(Path("my.yaml"), command="python my_experiment.py")
```

with `my.yaml` naming the components exactly like the built-ins:

```yaml
model:
  name: logistic       # your registered architecture
  input_dim: 2
  num_classes: 2
dataset:
  name: two_blobs      # your registered dataset
  batch_size: 16
loss:
  name: cross_entropy  # or mse / l1, or your own
```

Custom components receive identical validation, contract comparison, checkpointing, and resume semantics, mix freely with the built-ins, and run manifests record where each component's code lives. Custom callbacks extend the lifecycle through `extra_callbacks` the same way. The complete reference — constructor signatures, `validate`/`contract` declarations, task-aware losses, callbacks, and the error-message catalog — lives in [CUSTOMIZING.md](CUSTOMIZING.md), and runnable examples ship in `examples/`:

```powershell
python examples/custom_experiment.py      # custom architecture + dataset, classification
python examples/regression_experiment.py  # custom dataset with the built-in mse loss and sgd optimizer
```

## Development

```powershell
python -m pytest
python -m trainlattice train configs/synthetic_mlp.yaml
```

Tests use deterministic synthetic data and run offline. Engineering boundaries and contribution rules are documented in [ARCHITECTURE.md](ARCHITECTURE.md) and [AGENTS.md](AGENTS.md).

## License

TrainLattice is released under the MIT License.
