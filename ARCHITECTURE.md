# TrainLattice Architecture

## Purpose and Scope

TrainLattice is a transparent, configuration-driven PyTorch runtime for local classification experiments. It turns one strictly validated YAML file into a reproducible training run: a registered architecture (`mlp` or `cnn`) on a registered dataset (`synthetic` or `tensor`), the fixed AdamW optimizer, lifecycle callbacks, structured metrics, complete atomic checkpoints, explicit resume, and inspectable artifacts.

The runtime targets one Python process and one selected device (`cpu`, `cuda`, `cuda:N`, or `auto`). Its only interface is the `trainlattice` CLI with the `train` command.

## System Context

```text
User command: trainlattice train CONFIG [--resume PATH]
    |
    v
CLI -> Config loader -> Validated ExperimentConfig
                           |
                           v
                        Runner
          +----------------+----------------+
          |                |                |
          v                v                v
      Registries       Run context      Environment
 (mlp, cnn, synthetic, tensor) and manifest  capture
          |
          v
     Model / Dataset -> Trainer <-> TrainerState
                          |
                          v
                     Events -> Callbacks -> JSONL/console logs / checkpoints
```

The runtime is deliberately layered. The CLI translates user intent into one application call. The runner composes components. The trainer executes the lifecycle. Focused services own persistence, observability, and environment metadata.

## Package Boundaries

| Package | Responsibility | May depend on |
|---|---|---|
| `cli` | Argument parsing, dispatch, human-readable output | runtime runner |
| `config` | Typed schema, YAML parsing, component contracts, resume-compatibility comparison | standard library, PyYAML, `errors` |
| `registry` | Generic named-constructor registry with optional per-entry declarations | `errors` |
| `modeling` | MLP architecture, registration, validation, construction | PyTorch, `config`, `registry`, `errors` |
| `datasets` | Synthetic dataset registration and validation, split construction, epoch sampler, data loaders | PyTorch, `config`, `registry`, `errors` |
| `logging` | Console and JSONL metric sinks | standard library |
| `checkpoint` | Atomic save/load, RNG capture and restore, resume safety | PyTorch, `config`, `trainer` state, `errors` |
| `callbacks` | Logging and checkpoint reactions to lifecycle events | `logging`, `checkpoint`, `trainer` |
| `trainer` | Fit/validate lifecycle, state transitions, sample-weighted metrics | PyTorch, `callbacks` |
| `runtime` | Seeding, device selection, environment capture, run directories, manifests, composition | all domain packages except `cli` |

Dependencies flow from `cli` and `runtime` orchestration toward focused domain modules; nothing imports the CLI. Cross-cutting behavior enters `Trainer` through explicit collaborators rather than package-global flags.

## Configuration Model

`ExperimentConfig` is the validated root schema. It contains exactly five sections:

- `experiment`: name, seed, and run root.
- `model`: architecture name (`mlp` or `cnn`) plus the architecture's options.
- `dataset`: registered name (`synthetic` or `tensor`), batch size, worker count, and the dataset's options.
- `optimizer`: the learning rate for the fixed AdamW optimizer.
- `trainer`: epochs, device, validation cadence, log cadence, the monitored validation metric, and the optional gradient-norm bound.

The loader parses YAML, materializes dataclasses, then validates types, ranges, and within-section constraints. Unknown sections and unknown fixed-schema fields are rejected with dotted paths. Model and dataset options and the contracts between them are preflighted by the runner through the registry declarations before any component is constructed or a run directory is created: the input dimensions must agree when both sides state comparable dimensions — an exact `input_shape` match, or the flattened product when the model states only a width — and `model.num_classes` must equal `dataset.num_classes`. The resolved configuration is the only configuration passed into component factories; it is written to every run directory and stored inside every checkpoint.

## Registry and Construction

`Registry` maps a normalized lowercase name to a `Registration`: a constructor plus optional `validate` and `contract` declarations. `validate` checks the component's supported options with dotted-path errors; `contract` maps the resolved options to a `ComponentContract` (structured input shape, flattened input width, and class count) that the runner compares between the registered model and dataset before construction. Entries without declarations pass options through and skip contract comparison. Duplicate registration is rejected and lookup failures report the available choices. Two registries are the visible extension points: `modeling.ARCHITECTURES` (registered: `mlp`, `cnn`) and `datasets.DATASETS` (registered: `synthetic`, `tensor`). Built-ins register when their owning component module is imported.

`runtime/runner.py` composes one experiment in this order:

```text
load and validate configuration
  -> seed every generator
  -> resolve the explicit resume checkpoint, when requested
  -> validate model and dataset options
  -> compare model and dataset contracts
  -> select device
  -> build the registered architecture
  -> construct AdamW with the configured learning rate
  -> build deterministic data loaders
  -> create run context (config, environment, manifest)
  -> build logger, callbacks, and checkpoint manager
  -> restore checkpoint state when resuming
  -> construct Trainer and fit
```

## Model Infrastructure

The architecture registry dispatches on `model.name`. The built-in `mlp` flattens its input, applies `Linear -> GELU -> Dropout(0.1)` for every width in `hidden_dims`, and ends with a `Linear` to `num_classes`; activation and dropout are fixed properties of the architecture, and `input_dim`, `hidden_dims`, and `num_classes` are the supported options. The built-in `cnn` applies a `Conv2d(kernel 3, padding 1) -> GELU -> MaxPool(2)` block for every width in `channels`, then flattens into a `Linear` head; its supported options are `input_shape`, `channels`, and `num_classes`, and validation requires the spatial dimensions to stay divisible by the total pooling factor. Validation runs before module construction for every architecture and rejects unknown options and non-positive dimensions.

## Data Infrastructure

The synthetic dataset is generated in memory from the experiment seed, with independent deterministic generators for the train and validation splits (`samples`, `val_samples`, `input_shape`, `num_classes`). Nothing is downloaded and no optional vision dependency exists.

The tensor dataset reads one tensor-pair file per split from a local `root` directory (`train_file`, `val_file`, `num_classes`); each file holds a `{"features": Tensor, "labels": Tensor}` mapping loaded with `weights_only=True`, and split files are validated for existence, consistent lengths, and labels below `num_classes` before any run directory is created.

Dataset registration and option validation live in `datasets/builtin.py` (synthetic) and `datasets/tensor.py` (tensor). Split construction, the epoch sampler, and loader assembly live in `datasets/loaders.py`.

Training loaders use an epoch-aware sampler whose order is derived from the experiment seed and the epoch index. A resumed epoch therefore receives the same sample order as the corresponding uninterrupted epoch without depending on a hidden DataLoader generator cursor.

## Training Lifecycle

`TrainerState` is the serializable runtime snapshot. It records the completed epoch index, next epoch, global step, latest metrics, status, and timestamps.

The normal event order is:

```text
on_train_start
  for each epoch from state.next_epoch:
    set deterministic sampler epoch
    on_epoch_start
      for each training batch:
        on_batch_start
        forward -> cross-entropy loss -> backward -> AdamW step
        update state
        on_batch_end
    validate when (epoch + 1) % validate_every == 0
    on_epoch_end
on_train_end
```

Training and validation metrics are sample-weighted loss and accuracy aggregated in `trainer/metrics.py`. Validation runs under evaluation mode and without gradient recording, then restores the previous model mode even on failure.

When `trainer.max_grad_norm` is set, gradients are clipped by total norm between `backward()` and the optimizer step. When the monitored validation metric improves on the best value seen so far, the checkpoint callback rewrites `checkpoints/best.pt` in the same epoch-end step as `last.pt`; the best value lives in `TrainerState` and therefore survives resume.

An exception sets the state to `failed`, emits `on_exception` so callbacks can close log streams and write a diagnostic emergency snapshot, updates the manifest, and is re-raised. Exceptions raised inside exception-handling callbacks are reported as WARNING-level diagnostic records without replacing the original training failure.

## Logging

Two independent channels leave the process: metric logging and diagnostic logging.

Metric logging persists training measurements. Loggers implement `log(metrics, step)` and `close()`. A composite logger fans out to the console sink and the JSONL sink appended to `metrics.jsonl`. Metric records carry a step, a phase (`train` for batch records, `epoch` for epoch records), the epoch index, and scalar values. `LoggingCallback` decides when trainer state becomes a log record, driven by `trainer.log_every` for batch records and by epoch completion for epoch records. The trainer does not write files or format terminal output.

Diagnostic logging uses the standard-library `logging` module to report lifecycle events: run directory creation and resume boundaries in the runner, checkpoint saves and restores in the checkpoint manager, callback failures during exception handling in the trainer, and translated errors in the CLI. Library modules never install handlers; they only obtain `logging.getLogger(__name__)` and emit records. Handler configuration belongs to the CLI alone: the `train` command attaches a stdout handler for records below WARNING and a stderr handler for WARNING and above to the root logger, both formatting records as `[LEVEL] logger.name: message`. Stdout therefore carries metric and result output alongside informational diagnostics, while stderr can be archived separately. Python warnings are captured into the same format through `logging.captureWarnings`. The `trainlattice` logger runs at INFO by default; `--verbose` lowers it to DEBUG.

## Checkpoints and Resume

The checkpoint manager writes to a temporary sibling file and then atomically replaces the target. A checkpoint (format version 3) contains:

- format version;
- resume-safety marker;
- model state;
- optimizer state;
- serialized trainer state;
- resolved configuration;
- Python, NumPy when available, CPU Torch, and CUDA RNG state when available.

`last.pt` is rewritten after every completed epoch and is the resume source. `best.pt` is rewritten whenever the monitored validation metric improves and holds the same complete payload. `emergency.pt` records the point of an uncaught mid-epoch exception for diagnosis, is marked not resume-safe, and is rejected at load time. Only format version 3 is accepted; any other version is rejected explicitly.

Resume is always explicit (`--resume` with a run directory or checkpoint file). Loading restores model, optimizer, and trainer state — including the best monitored metric — plus available RNG state, then training begins at `TrainerState.next_epoch`. A compatibility check compares recovery-sensitive configuration values field by field; runtime-only settings (`trainer.device`, `trainer.log_every`, `trainer.epochs`, and `experiment.runs_dir`) may change, while `trainer.monitor` and `trainer.max_grad_norm` must match because they shape the recovered trajectory and the best-checkpoint standard. Configuration keys introduced after a checkpoint was saved compare as their default values, so checkpoints from earlier feature releases remain resumable. A resumed run is a new run directory whose manifest records `resume_source`.

## Experiment Artifacts

Every invocation receives a timestamped, collision-safe directory:

```text
runs/<timestamp>_<experiment>/
  config.yaml
  manifest.json
  environment.json
  metrics.jsonl
  checkpoints/
    last.pt
    best.pt
    emergency.pt
```

The manifest is updated atomically across `created`, `running`, `finished`, and `failed` states. It records the command, Git commit when available, seed, component names, timestamps, epoch, global step, and resume source. `environment.json` records Python, PyTorch, CUDA, platform, and device information for later comparison.

## CLI Surface

- `trainlattice train CONFIG [--resume PATH] [--verbose]`: create a run and execute training, optionally resuming from an explicit checkpoint; `--verbose` lowers the diagnostic log level to DEBUG.
- `trainlattice --version`: print the package version.

A finished run returns exit code 0 and prints `Run <status>: <run directory>`. Invalid configuration, missing files, incompatible resume checkpoints, and runtime failures are translated into one `[ERROR] trainlattice.cli.main: <message>` diagnostic record on stderr with exit code 2.

## Reliability and Compatibility Invariants

1. A completed checkpoint is never partially visible at its final path.
2. A completed-epoch resume preserves `global_step`, optimizer state, trainer state, available RNG state, and the epoch's sample order.
3. Run configuration snapshots are resolved values and are not mutated after creation.
4. Training and validation metric aggregation is sample-weighted.
5. Validation runs under evaluation mode and without gradient recording, then restores the previous model mode even on failure.
6. CPU execution is fully supported; CUDA is selected only when requested and available.
7. Importing `trainlattice` downloads nothing; the synthetic dataset is generated in memory.
8. Persisted formats carry version fields, and checkpoints reject unsupported format versions at load time.

## Testing Strategy

The automated suite uses synthetic and tiny tensor data for deterministic coverage of:

- YAML parsing and strict unknown-section and unknown-field rejection with dotted paths;
- cross-field shape and class-count contracts, including structured-shape equality and flattened-product comparison;
- registry duplicate and lookup behavior, including dataset extension through registration;
- configurable MLP and CNN construction and option validation;
- deterministic train and validation split behavior, epoch-sampler ordering, and file-backed tensor dataset loading with preflight file errors;
- sample-weighted metrics and evaluation-mode restoration;
- trainer state transitions, callback event order, gradient clipping bounds, and best-checkpoint tracking;
- checkpoint save/load round trips, including RNG state, best-metric continuity, and format-version rejection;
- interrupted-run emergency snapshots, resumed epoch and step continuity, and resumed-versus-uninterrupted parameter equality;
- run artifact contents and CLI exit behavior;
- one end-to-end synthetic training and resume flow.

## Extension Points

Later releases can introduce additional architectures and datasets by registering constructors in `modeling` and `datasets`, and features such as automatic mixed precision, gradient accumulation, clipping, profiling, `torchrun`, DDP, and FSDP by adding explicit policies and strategies around the existing state and lifecycle boundaries. Checkpoint format changes require a versioned migration path.
