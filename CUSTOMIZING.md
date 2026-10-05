# Customizing TrainLattice

English | [简体中文](CUSTOMIZING.zh-CN.md)

Everything the runtime builds is a registered component. Architectures, datasets, optimizers, and losses all live in registries, and any Python module that imports TrainLattice and calls `register` extends them — there is no plugin loading mechanism, no entry-point discovery, and no configuration for registration itself. This document is the complete reference for writing your own components.

## The Model

Four registries are the extension points, exported from the package root:

| Registry | Builds | Built-ins | Constructor signature |
|---|---|---|---|
| `ARCHITECTURES` | models | `mlp`, `cnn` | `(*, **options) -> nn.Module` |
| `DATASETS` | datasets | `synthetic`, `tensor` | `(*, train, split=None, **options) -> Dataset` |
| `OPTIMIZERS` | optimizers | `adamw`, `adam`, `sgd` | `(params, lr=..., **options) -> Optimizer` |
| `LOSSES` | losses | `cross_entropy`, `mse`, `l1` | `(**options) -> nn.Module`, via `register_loss` |

Importing a module is what registers its components. Your experiment script imports the component modules, then starts the run through `run_experiment` or `run_from_file`; the YAML configuration only names the components. The same registration calls the built-ins use are the ones you call — custom components get identical validation, contract comparison, checkpointing, and resume semantics.

## Registering Each Component Kind

### Architectures

```python
import torch.nn as nn

from trainlattice import ARCHITECTURES, ComponentContract, ConfigError


class Logistic(nn.Module):
    def __init__(self, *, input_dim: int = 2, num_classes: int = 2) -> None:
        super().__init__()
        self.network = nn.Linear(input_dim, num_classes)

    def forward(self, inputs):
        return self.network(inputs)


def _validate(path, options):
    unknown = sorted(set(options) - {"input_dim", "num_classes"})
    if unknown:
        keys = ", ".join(f"{path}.{key}" for key in unknown)
        raise ConfigError(f"unknown configuration field(s): {keys}")


def _contract(options):
    return ComponentContract(
        input_dim=options.get("input_dim", 2),
        num_classes=options.get("num_classes", 2),
    )


ARCHITECTURES.register("logistic", Logistic, validate=_validate, contract=_contract)
```

Conventions:

- The constructor takes flat keyword options mirroring the YAML `model` keys (everything except `name`) and returns an `nn.Module`.
- `validate(path, options)` receives the section path (`"model"`) and the resolved options. Rejecting unknown keys here is what turns typos into dotted-path errors before anything is built.
- `contract(options)` restates the input width/shape and output width from the options alone — the runtime never probes data to learn shapes. `ComponentContract(input_dim=..., num_classes=..., input_shape=...)`: `input_dim` is the flattened input width, `input_shape` the structured shape (state one or both; `None` skips that comparison track), and `num_classes` the output width — the class count for classification, the target width for regression.

### Datasets

```python
from torch.utils.data import Dataset, TensorDataset

from trainlattice import DATASETS


def build_two_blobs(
    *, train, split=None, input_dim=2, samples=128, val_samples=32, seed=None, **_
) -> Dataset:
    ...  # build one Gaussian blob per class, return TensorDataset(features, labels)


DATASETS.register("two_blobs", build_two_blobs, validate=..., contract=...)
```

Conventions:

- The constructor is called **once per split** — `train=True, split="train"` and `train=False, split="validation"` — and receives a `seed` injected by the runtime from `experiment.seed`. Derive per-split generators from it so both splits are deterministic and disjoint in randomness.
- Samples are `(input, target)` pairs: `long` targets for classification losses, float targets **shaped like the model outputs** for regression losses.
- Unknown extra keyword arguments should be tolerated (`**_`) — the runtime injects `seed` on top of your YAML options.
- Data augmentation belongs to the dataset constructor; the runtime applies no hidden transforms.

### Optimizers

```python
import torch

from trainlattice import OPTIMIZERS

OPTIMIZERS.register("rmsprop", torch.optim.RMSprop, validate=...)
```

Conventions:

- The constructor receives the model parameters **positionally**, then the learning rate and options: `constructor(params, lr=..., **options)` — exactly the `torch.optim` signature, so optimizer classes register directly.
- YAML: `optimizer: {name: rmsprop, lr: 0.001, alpha: 0.99}` — every key except `name` and `lr` is passed through as an option.
- There is no contract declaration for optimizers; `validate` is the only optional declaration.

### Losses

```python
import torch

from trainlattice import register_loss

register_loss("huber", torch.nn.SmoothL1Loss, task="regression", validate=...)
```

Conventions:

- Losses register through `register_loss`, which additionally requires a **task kind**: `classification` or `regression`.
- The task drives metrics: classification losses report loss and accuracy; regression losses report loss only. It also constrains `trainer.monitor` — `val_accuracy` requires a classification loss.
- YAML: `loss: {name: huber}` with loss-specific options passed through (`reduction`, and for `cross_entropy` also `label_smoothing`, `ignore_index`).

## A Complete Custom Experiment

See `examples/` for runnable versions: `custom_model.py`, `custom_dataset.py`, `custom_experiment.py`, `custom.yaml` (classification), and `regression_experiment.py` + `regression.yaml` (a custom dataset with the built-in `mse` loss and `sgd` optimizer). The entry script is the whole story:

```python
from pathlib import Path

import custom_dataset  # registers two_blobs on import
import custom_model  # registers logistic on import

from trainlattice import run_from_file

CONFIG = Path(__file__).parent / "custom.yaml"

result = run_from_file(CONFIG, command="python my_experiment.py")
# Resume after an interruption:
# run_from_file(CONFIG, resume="runs/<run>/checkpoints/last.pt", command="python my_experiment.py")
```

Pass the real launch command to `command=` — the run manifest records it as the reproduction entry, alongside `model_source`, `dataset_source`, `optimizer_source`, and `loss_source` fields pointing at the module and file that defined each component.

The YAML is the same shape as any built-in configuration; only the names differ:

```yaml
model:
  name: logistic
  input_dim: 2
  num_classes: 2

dataset:
  name: two_blobs
  batch_size: 16
  input_dim: 2

loss:
  name: cross_entropy
```

Custom and built-in components mix freely in any combination.

## Custom Callbacks

Lifecycle behavior extends through `Callback` subclasses passed to `extra_callbacks`:

```python
from time import perf_counter

from trainlattice import Callback, run_from_file


class EpochTimeCallback(Callback):
    def on_epoch_start(self, trainer):
        self._started = perf_counter()

    def on_epoch_end(self, trainer):
        print(f"epoch {trainer.state.epoch} took {perf_counter() - self._started:.3f}s")


run_from_file(CONFIG, command="python my_experiment.py", extra_callbacks=[EpochTimeCallback()])
```

Hooks receive the trainer — `on_train_start`, `on_epoch_start`, `on_batch_start`, `on_batch_end`, `on_epoch_end`, `on_train_end`, `on_exception` — so a callback can read the model, optimizer, and `TrainerState` at every boundary. Injected callbacks run after metric logging and before checkpointing within each event. Callbacks are observers: they cannot interrupt training (control-flow hooks such as early stopping are a future extension).

## Validation and Error Messages

All failures below happen before a run directory is created:

| Mistake | Error message |
|---|---|
| Misspelled component name | `unsupported model.name 'logstic'; available: cnn, logistic, mlp` |
| Unknown component option | `unknown configuration field(s): model.kernel_size` |
| Invalid option value | `optimizer.momentum must be a non-negative number` |
| Shape mismatch | `model.input_dim (4) must match dataset.input_shape product (2)` |
| Structured-shape mismatch | `model.input_shape ([3, 4, 4]) must match dataset.input_shape ([4])` |
| Output-width mismatch | `model.num_classes (3) must match dataset.num_classes (2)` |
| Monitor/task mismatch | `trainer.monitor 'val_accuracy' requires a classification loss; loss.name 'mse' is regression` |

Registering an already-registered name raises immediately (`architecture 'mlp' is already registered`), so built-ins cannot be shadowed. Without a `validate` declaration, unknown options surface as construction errors instead of dotted-path messages; without a `contract`, that component skips its side of the shape comparison. Declaring both is recommended.

## Compatibility Notes

- **Pickling**: with `num_workers > 0`, worker processes are spawned (not forked) on Windows and macOS — closure- or lambda-built datasets fail there. Use `num_workers: 0` while iterating on a custom dataset.
- **Resume**: recovery-sensitive configuration must match the checkpoint, which includes the loss section, the optimizer name, and all component options. A resumed run re-imports your component modules and rebuilds identical components; if you change component code between runs, resume from the changed code at your own risk — only the configuration is compared, not the source.
- **Reproduction**: the run manifest records where each component's code lives, but the experiment script itself is part of the reproduction entry — archive it alongside the run directory.

## Why It Is Designed This Way

- **Import-time registration instead of dynamic loading**: the YAML could point at plugin files to import, but that adds a loader, path-resolution rules, and a second discovery mechanism — while `import` is the mechanism Python programmers already have. Registration stays visible in the script, and any debugger, profiler, or reader can follow it.
- **Declarations travel with the registration**: `validate` and `contract` are attached where the constructor is registered, so a component carries its own validation and shape semantics everywhere it is used — built-ins and custom components are indistinguishable to the runtime.
- **The task kind is declared, not sniffed**: the trainer could guess metrics from target dtypes at runtime, but a declaration in `register_loss` fails fast at validation time and reads as intent.
