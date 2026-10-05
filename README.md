# TrainLattice

English | [简体中文](README.zh-CN.md)

Minimal infrastructure for deep learning training.

**Build. Run. Recover. Reproduce.**

TrainLattice is a lightweight, transparent PyTorch training runtime for studying deep learning infrastructure from first principles. One strictly validated YAML file becomes a reproducible training run: a registered MLP architecture on the registered synthetic dataset, the fixed AdamW optimizer, lifecycle callbacks, structured metrics, atomic checkpoints, explicit resume, and self-contained run artifacts.

## Features

- Strict YAML experiment configuration: unknown sections and fields are rejected with dotted-path errors before a run directory is created
- Registry-based component construction; the `mlp` architecture and `synthetic` dataset registries remain visible extension points
- Configurable MLP with `input_dim`, `hidden_dims`, and `num_classes`; GELU activations and 0.1 dropout
- Deterministic synthetic train and validation splits plus epoch-aware shuffle sampling
- Explicit `Trainer` lifecycle and serializable `TrainerState`
- Console and JSONL structured metric logging with sample-weighted loss and accuracy
- Structured diagnostic logging in `[LEVEL] logger.name: message` format: INFO and below go to stdout, WARNING and above go to stderr, Python warnings are captured into the same format, and `--verbose` enables debug-level diagnostics
- Atomic checkpoints (format v3) holding model, optimizer, trainer state, resolved configuration, and RNG state: `last.pt` after every epoch, plus a diagnostic `emergency.pt` when training fails
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

Metrics stream to the console and to `metrics.jsonl` in the new run directory. When the run finishes, the CLI prints the directory, for example `Run finished: runs/20261004-120000_synthetic-mlp-smoke`.

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
    emergency.pt
```

- `config.yaml` — the resolved configuration snapshot for the run.
- `manifest.json` — command, Git commit, seed, status, timestamps, epoch, global step, and resume source; updated atomically through `created`, `running`, `finished`, and `failed` states.
- `environment.json` — Python, PyTorch, CUDA, platform, and selected device information.
- `metrics.jsonl` — append-only metric records, one JSON object per line.
- `checkpoints/last.pt` — written atomically after every completed epoch; the resume source.
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
- `model` — `name` must be `mlp`; options are `input_dim` (default `784`), `hidden_dims` (default `[512, 256, 128]`), and `num_classes` (default `10`). Hidden layers use GELU and 0.1 dropout.
- `dataset` — `name` must be `synthetic`; `batch_size` (default `64`), `num_workers` (default `0`); options are `samples` (default `256`), `val_samples` (default `64`), `input_shape` (default `[1, 28, 28]`), and `num_classes` (default `10`).
- `optimizer` — `lr` (default `0.001`) for the fixed AdamW optimizer.
- `trainer` — `epochs` (default `1`), `device` (`auto`, `cpu`, `cuda`, or `cuda:N`), `validate_every` (default `1`), `log_every` (default `10`).

Two cross-field contracts are enforced: `model.input_dim` must equal the product of `dataset.input_shape`, and `model.num_classes` must equal `dataset.num_classes`. Unknown sections, fields, and component options are rejected with dotted paths such as `optimizer.weight_decay` or `dataset.root`.

## Development

```powershell
python -m pytest
python -m trainlattice train configs/synthetic_mlp.yaml
```

Tests use deterministic synthetic data and run offline. Engineering boundaries and contribution rules are documented in [ARCHITECTURE.md](ARCHITECTURE.md) and [AGENTS.md](AGENTS.md).

## License

TrainLattice is released under the MIT License.
