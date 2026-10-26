# TrainLattice Architecture

## Purpose and Scope

TrainLattice is a transparent, configuration-driven deep learning training infrastructure — single-process to start — for local training experiments. It turns one strictly validated YAML file into a reproducible training run: a registered architecture (the built-in `mlp` or `cnn`, or a custom one) on a registered dataset (the built-in `synthetic` or `tensor`, or a custom one), a registered optimizer (the built-in `adamw`, `adam`, or `sgd`), an optional registered learning-rate scheduler (the built-in `warmup_cosine`, `warmup_linear`, or `step_decay`), a registered loss (the built-in `cross_entropy`, `mse`, or `l1`), lifecycle callbacks, structured metrics, complete atomic checkpoints, explicit resume, and inspectable artifacts.

The infrastructure currently runs in one Python process on one selected device (`cpu`, `cuda`, `cuda:N`, or `auto`); distributed execution is a long-term roadmap direction and is not scheduled. Its interfaces are the `trainlattice` CLI with the `train` command and the Python entry points `run_experiment` and `run_from_file` exported from the package root; the Python entry points are how custom components, registered by importing their modules, reach the same validated pipeline.

## System Context

```text
User entry: trainlattice train CONFIG [--resume PATH]
        or: python script calling run_experiment / run_from_file
    |
    v
CLI / Python entry -> Config loader -> Validated ExperimentConfig
                           |
                           v
                        Runner
          +----------------+----------------+
          |                |                |
          v                v                v
      Registries       Run context      Environment
 (models, datasets,   and manifest     capture
  optimizers, losses:
  built-ins plus imported
  custom components)
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
| `optim` | Optimizer registration, validation, and construction | PyTorch, `config`, `registry`, `errors` |
| `logging` | Console and JSONL metric sinks | standard library |
| `checkpoint` | Atomic save/load, RNG capture and restore, resume safety | PyTorch, `config`, `trainer` state, `errors` |
| `callbacks` | Logging and checkpoint reactions to lifecycle events | `logging`, `checkpoint`, `trainer` |
| `trainer` | Fit/validate lifecycle, task-aware metrics, loss registration and task declaration, state transitions | PyTorch, `callbacks` |
| `runtime` | Seeding, device selection, environment capture, run directories, manifests, composition | all domain packages except `cli` |

Dependencies flow from `cli` and `runtime` orchestration toward focused domain modules; nothing imports the CLI. Cross-cutting behavior enters `Trainer` through explicit collaborators rather than package-global flags. The package root (`trainlattice/__init__.py`) is the public API surface: it re-exports the experiment entry points (`run_experiment`, `run_from_file`), the configuration types and loaders (`ExperimentConfig`, `LossConfig`, `load_config`, `save_config`, `ComponentContract`), the five component registries (`ARCHITECTURES`, `DATASETS`, `OPTIMIZERS`, `SCHEDULERS`, `LOSSES` with `register_loss`), `Callback`, and the error types.

## Configuration Model

`ExperimentConfig` is the validated root schema. It contains exactly seven sections:

- `experiment`: name, seed, and run root.
- `model`: architecture name (any registered architecture — the built-in `mlp` or `cnn`, or a custom one) plus the architecture's options.
- `dataset`: dataset name (any registered dataset — the built-in `synthetic` or `tensor`, or a custom one), batch size, worker count, and the dataset's options.
- `optimizer`: optimizer name (default `adamw`; built-ins `adamw`, `adam`, `sgd`), the learning rate, and the optimizer's options.
- `trainer`: epochs, device, validation cadence, log cadence, the monitored validation metric, the optional gradient-norm bound, the micro-batches per optimizer step (`accumulate`), the numeric mode (`precision`: `fp32`, `bf16`, `fp16`), and the optional stopping criteria (`early_stopping` with `patience` and `min_delta`).
- `loss`: loss name (default `cross_entropy`; built-ins `cross_entropy`, `mse`, `l1`) and the loss's options.
- `scheduler`: optional scheduler name plus the scheduler's options; an absent or explicitly null section keeps a constant learning rate.

The loader parses YAML, materializes dataclasses, then validates types, ranges, and within-section constraints. Unknown sections and unknown fixed-schema fields are rejected with dotted paths. Model and dataset options and the contracts between them are preflighted by the runner through the registry declarations before any component is constructed or a run directory is created: the input dimensions must agree when both sides state comparable dimensions — an exact `input_shape` match, or the flattened product when the model states only a width — and the model's output width must equal the dataset's declared width (the class count for classification, the target width for regression). The resolved configuration is the only configuration passed into component factories; it is written to every run directory and stored inside every checkpoint.

## Registry and Construction

`Registry` maps a normalized lowercase name to a `Registration`: a constructor plus optional `validate` and `contract` declarations. `validate` checks the component's supported options with dotted-path errors; `contract` maps the resolved options to a `ComponentContract` (structured input shape, flattened input width, and class count) that the runner compares between the registered model and dataset before construction. Entries without declarations pass options through and skip contract comparison. Duplicate registration is rejected and lookup failures report the available choices. Five registries are the visible extension points: `modeling.ARCHITECTURES` (built-ins: `mlp`, `cnn`), `datasets.DATASETS` (built-ins: `synthetic`, `tensor`), `optim.OPTIMIZERS` (built-ins: `adamw`, `adam`, `sgd`; constructors receive the model parameters positionally), `optim.SCHEDULERS` (built-ins: `warmup_cosine`, `warmup_linear`, `step_decay`; constructors receive the optimizer and the resolved `total_steps` positionally), and `trainer.LOSSES` (built-ins: `cross_entropy`, `mse`, `l1`). Losses register through `register_loss`, which additionally declares a task kind — `classification` or `regression`; the trainer computes accuracy alongside loss only for classification tasks, and `trainer.monitor: val_accuracy` requires a classification loss.

Custom architectures and datasets register through the same calls. Architecture constructors take flat keyword options mirroring the YAML `model` keys and return an `nn.Module`; dataset constructors use the signature `(*, train, split=None, **options)`, accept the runtime-injected `seed`, and return a `Dataset` of `(input, target)` pairs — `long` targets for classification losses, float targets shaped like the model outputs for regression losses. Importing a module that calls `register` extends the registries, so a user script imports its component modules and then starts the run through `run_experiment` or `run_from_file`; `examples/` ships a complete custom architecture, dataset, configuration, and entry script, plus a regression entry that pairs a custom dataset with the built-in `mse` loss and `sgd` optimizer. Custom components receive the same validation, contract comparison, checkpointing, and resume semantics as the built-ins, and the run manifest records where each component's code is defined.

Training behavior extends through the same Python entry points: both accept `extra_callbacks`, instances of `Callback` whose hooks observe the trainer within each lifecycle event, ordered after metric logging and before checkpointing. Callbacks remain observers — they can read the model, optimizer, and trainer state, and they can end the run cooperatively through `Trainer.request_stop(reason)`, which `fit` honors once the current epoch's callbacks (including the checkpoint save) have finished. Callbacks cannot interrupt a step or an epoch in progress; hard interruption stays outside the callback contract.

`runtime/runner.py` composes one experiment in this order:

```text
load and validate configuration
  -> seed every generator
  -> resolve the explicit resume checkpoint, when requested
  -> validate model, dataset, optimizer, loss, and scheduler options
  -> check the monitor matches the loss task
  -> compare model and dataset contracts
  -> select device
  -> build the registered architecture
  -> construct the registered optimizer with the configured learning rate
  -> build the registered loss
  -> build deterministic data loaders
  -> construct the registered scheduler when one is configured
     (its step budget derived from the loader and epochs unless stated)
  -> create run context (config, environment, manifest)
  -> build logger, callbacks, and checkpoint manager
  -> construct Trainer (ahead of the restore, so its fp16 scaler joins the checkpoint load)
  -> restore checkpoint state when resuming
  -> fit
```

## Model Infrastructure

The architecture registry dispatches on `model.name`. The built-in `mlp` flattens its input, applies `Linear -> GELU -> Dropout(0.1)` for every width in `hidden_dims`, and ends with a `Linear` to `num_classes`; activation and dropout are fixed properties of the architecture, and `input_dim`, `hidden_dims`, and `num_classes` are the supported options. The built-in `cnn` applies a `Conv2d(kernel 3, padding 1) -> GELU -> MaxPool(2)` block for every width in `channels`, then flattens into a `Linear` head; its supported options are `input_shape`, `channels`, and `num_classes`, and validation requires the spatial dimensions to stay divisible by the total pooling factor. Validation runs before module construction for every architecture and rejects unknown options and non-positive dimensions.

## Data Infrastructure

The synthetic dataset is generated in memory from the experiment seed, with independent deterministic generators for the train and validation splits (`samples`, `val_samples`, `input_shape`, `num_classes`). Nothing is downloaded and no optional vision dependency exists.

The tensor dataset reads one tensor-pair file per split from a local `root` directory (`train_file`, `val_file`, `num_classes`); each file holds a `{"features": Tensor, "labels": Tensor}` mapping loaded with `weights_only=True`, and split files are validated for existence, consistent lengths, and labels below `num_classes` before any run directory is created.

Dataset registration and option validation live in `datasets/builtin.py` (synthetic) and `datasets/tensor.py` (tensor). Split construction, the epoch sampler, and loader assembly live in `datasets/loaders.py`.

Training loaders use an epoch-aware sampler whose order is derived from the experiment seed and the epoch index. A resumed epoch therefore receives the same sample order as the corresponding uninterrupted epoch without depending on a hidden DataLoader generator cursor.

## Training Lifecycle

`TrainerState` is the serializable runtime snapshot. It records the completed epoch index, next epoch, global step, latest metrics, best monitored value, status, timestamps, and the recorded reason when a run ends before its final epoch.

The normal event order is:

```text
on_train_start
  for each epoch from state.next_epoch:
    set deterministic sampler epoch
    reset the cuda memory peak statistic (cuda devices)
    on_epoch_start
      for each micro-batch:
        fetch the batch through an explicit iterator (timed into data_seconds)
        on_batch_start
        zero grads when a window opens
        forward under autocast (bf16/fp16) -> loss / accumulate -> backward
        when the micro-batch closes a window of accumulate batches:
          unscale when fp16 -> clip when max_grad_norm is set -> optimizer step -> scheduler step (when configured)
          update state (batch_metrics, global_step)
        on_batch_end
      flush a partial end-of-epoch window as one optimizer step
    validate when (epoch + 1) % validate_every == 0
    evaluate early stopping (when configured)
    on_epoch_end
    break when a stop was requested (reason recorded in state)
on_train_end
```

Training and validation metrics are sample-weighted and task-aware, aggregated in `trainer/metrics.py`: classification losses accumulate loss and correct predictions; regression losses accumulate loss only, and their datasets provide float targets shaped like the model outputs. The trainer selects the aggregate from the loss's declared task kind. Validation runs under evaluation mode and without gradient recording, then restores the previous model mode even on failure.

Micro-batches are consumed in windows of `trainer.accumulate` batches: gradients are zeroed when a window opens, each micro-batch loss is scaled by `1 / accumulate` before backward — with a mean-reduced loss over equal-sized micro-batches the accumulated gradient equals the gradient of the full batch — and one optimizer step closes a window. A partial window at the epoch end is flushed as one final step with the same scaling. Hooks fire per micro-batch, so callbacks observe data batches, while `state.global_step` counts optimizer steps (`ceil(batches / accumulate)` per epoch) and `batch_metrics` describes the window the latest step consumed as a sample-weighted aggregate. When a micro-batch closes a window, its optimizer step — and the `global_step` and `batch_metrics` update — runs before that micro-batch's `on_batch_end`, so the hook observes post-step state (`LoggingCallback` emits one record per optimizer step through exactly this guarantee); a flushed partial window takes its step after the final micro-batch's `on_batch_end`.

When a scheduler is configured, the trainer advances it once after every optimizer step. Batch and epoch metric records carry `lr` under the same unprefixed name as `step` and `epoch`: batch records report the learning rate that optimizer step applied, read from the first parameter group before the step, and epoch records report the end-of-epoch rate, the one the next step would apply. The built-in `warmup_cosine` and `warmup_linear` schedules span a step budget resolved from the explicit `scheduler.total_steps` option first and the derived run budget (optimizer steps per epoch — `ceil(batches per epoch / trainer.accumulate)` — times `trainer.epochs`) second, with a linear warmup over `warmup_steps` and an absolute `min_lr` floor; `step_decay` wraps `StepLR` with `step_size` and `gamma`.

Throughput and memory measurements join the same records without touching any random-number generator: `train_epoch` times the epoch with `time.perf_counter` and fetches batches through an explicit iterator, so the wait for each `next` call accumulates into `data_seconds` while the remaining seconds count as compute. Batch records carry `steps_per_sec` and `samples_per_sec` — cumulative averages that divide the optimizer steps and samples completed since the epoch began by the elapsed wall-clock seconds, data waits included — and epoch records carry `epoch_seconds`, `data_seconds`, and the epoch-wide `train_samples_per_sec`, the sample rate under the `train_` prefix while the two duration keys stay unprefixed. On a cuda device each epoch resets the allocator's peak statistic before its first batch and the epoch record closes with `max_memory_mb`, `torch.cuda.max_memory_allocated` converted to megabytes; cpu records omit the key.

When `trainer.max_grad_norm` is set, gradients are clipped by total norm immediately before the optimizer step, after the window's gradients have accumulated. When the monitored validation metric improves on the best value seen so far, the checkpoint callback rewrites `checkpoints/best.pt` in the same epoch-end step as `last.pt`; the best value lives in `TrainerState` and therefore survives resume.

`trainer.early_stopping` arms convergence stopping on the same monitored metric that drives `best.pt`: `val_loss` improves downward, `val_accuracy` upward, and only movements beyond `min_delta` count. The evaluation runs after the epoch's metrics land and before the epoch-end callbacks fold the fresh value into `state.best_metric`, so the comparison always faces the prior best. Epochs whose metrics lack the monitored key — those without a validation pass — leave the patience counter untouched; once more than `patience` consecutive evaluations fail to improve, the trainer requests a stop. Any callback can request the same through `Trainer.request_stop(reason)`. The request is runtime-only trainer state: `fit` checks it after every epoch's callbacks have finished — so the stopping epoch's `last.pt` is already on disk — then ends the loop with status `finished` and the reason recorded in `state.stop_reason`. The patience counter is runtime state too and restarts from zero after a resume, while the `best_metric` reference carries over through the checkpoint; `trainer.early_stopping` itself is runtime-only for compatibility, so a resumed run may change or drop it.

`trainer.precision` selects the numeric mode. `fp32` (the default) computes directly in the model's dtype; `bf16` and `fp16` wrap training and validation forward passes in `torch.amp.autocast` with the matching dtype while backward runs outside the context. Only `fp16` adds a `GradScaler`, shared across an accumulation window: each micro-batch loss is backwarded in scaled form, and the window-closing step unscales the optimizer, clips when `max_grad_norm` is set — so the norm sees true magnitudes — applies the scaled update (skipped when the scaled gradients were non-finite), and adapts the scale once per optimizer step. `fp16` requires a cuda device, because cpu autocast supports bfloat16 only; the trainer rejects its construction with guidance toward `bf16`, which is the cpu mixed-precision mode and uses no scaler.

An exception sets the state to `failed`, emits `on_exception` so callbacks can close log streams and write a diagnostic emergency snapshot, updates the manifest, and is re-raised. Exceptions raised inside exception-handling callbacks are reported as WARNING-level diagnostic records without replacing the original training failure.

## Logging

Two independent channels leave the process: metric logging and diagnostic logging.

Metric logging persists training measurements. Loggers implement `log(metrics, step)` and `close()`. A composite logger fans out to the console sink and the JSONL sink appended to `metrics.jsonl`. Metric records carry a step, a phase (`train` for batch records, `epoch` for epoch records), the epoch index, and scalar values. `LoggingCallback` decides when trainer state becomes a log record, driven by `trainer.log_every` for batch records — one record per completed optimizer step, whose metrics aggregate that step's micro-batches — and by epoch completion for epoch records. The trainer does not write files or format terminal output.

Diagnostic logging uses the standard-library `logging` module to report lifecycle events: run directory creation and resume boundaries in the runner, checkpoint saves and restores in the checkpoint manager, callback failures during exception handling in the trainer, and translated errors in the CLI. Library modules never install handlers; they only obtain `logging.getLogger(__name__)` and emit records. Handler configuration belongs to the CLI alone: the `train` command attaches a stdout handler for records below WARNING and a stderr handler for WARNING and above to the root logger, both formatting records as `[LEVEL] logger.name: message`. Stdout therefore carries metric and result output alongside informational diagnostics, while stderr can be archived separately. Python warnings are captured into the same format through `logging.captureWarnings`. The `trainlattice` logger runs at INFO by default; `--verbose` lowers it to DEBUG.

## Checkpoints and Resume

The checkpoint manager writes to a temporary sibling file and then atomically replaces the target. A checkpoint (format version 4) contains:

- format version;
- resume-safety marker;
- model state;
- optimizer state;
- scheduler state and its resolved step budget when a scheduler is configured (omitted otherwise);
- fp16 GradScaler state when the trainer runs one (omitted otherwise);
- serialized trainer state;
- resolved configuration;
- Python, NumPy when available, CPU Torch, and CUDA RNG state when available.

`last.pt` is rewritten after every completed epoch and is the resume source. `best.pt` is rewritten whenever the monitored validation metric improves and holds the same complete payload. `emergency.pt` records the point of an uncaught mid-epoch exception for diagnosis, is marked not resume-safe, and is rejected at load time. Only format versions 3 and 4 are accepted; any other version is rejected explicitly.

Resume is always explicit (`--resume` with a run directory or checkpoint file). Loading restores model, optimizer, and trainer state — including the best monitored metric — plus available RNG state, then training begins at `TrainerState.next_epoch`. Scheduler state restores verbatim from a format-4 checkpoint together with the scheduler's resolved step budget, so the schedule a run started with survives a resume even when the runtime-only `trainer.epochs` changes; a budget that differs from the freshly derived one is noted in an INFO record. A checkpoint saved without a scheduler resumes without one — the compatibility check rejects adding a `scheduler` section. A scheduler handed directly to `CheckpointManager.load` for a stateless checkpoint is replayed from the base learning rate to the saved `global_step`, exact for closed-form and recursive schedules alike. The fp16 GradScaler restores verbatim from a checkpoint that carries scaler state; because `trainer.precision` must match, a scaler exists exactly when the checkpointed run trained in fp16, while a direct `load` caller resuming a stateless checkpoint starts from the scaler's initial scale. A compatibility check compares recovery-sensitive configuration values field by field; runtime-only settings (`trainer.device`, `trainer.log_every`, `trainer.epochs`, `trainer.early_stopping`, and `experiment.runs_dir`) may change, while `trainer.monitor`, `trainer.max_grad_norm`, `trainer.accumulate`, `trainer.precision`, and the scheduler section must match because they shape the recovered trajectory and the best-checkpoint standard. Configuration keys introduced after a checkpoint was saved compare as their default values, so checkpoints from earlier feature releases remain resumable. A resumed run is a new run directory whose manifest records `resume_source`.

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

The manifest is updated atomically across `created`, `running`, `finished`, and `failed` states. It records the command, Git commit when available, seed, component names, the module and file that defined the model, dataset, optimizer, and loss (`model_source`, `dataset_source`, `optimizer_source`, `loss_source`), timestamps, epoch, global step, and resume source. `environment.json` records Python, PyTorch, CUDA, platform, and device information for later comparison.

## CLI Surface

- `trainlattice train CONFIG [--resume PATH] [--verbose]`: create a run and execute training, optionally resuming from an explicit checkpoint; `--verbose` lowers the diagnostic log level to DEBUG.
- `trainlattice --version`: print the package version.

A finished run returns exit code 0 and prints `Run <status>: <run directory>`. Invalid configuration, missing files, incompatible resume checkpoints, and runtime failures are translated into one `[ERROR] trainlattice.cli.main: <message>` diagnostic record on stderr with exit code 2.

## Reliability and Compatibility Invariants

1. A completed checkpoint is never partially visible at its final path.
2. A completed-epoch resume preserves `global_step`, optimizer state, trainer state, scheduler state and step budget, available RNG state, and the epoch's sample order.
3. Run configuration snapshots are resolved values and are not mutated after creation.
4. Training and validation metric aggregation is sample-weighted.
5. Validation runs under evaluation mode and without gradient recording, then restores the previous model mode even on failure.
6. CPU execution is fully supported; CUDA is selected only when requested and available.
7. Importing `trainlattice` downloads nothing; the synthetic dataset is generated in memory.
8. Persisted formats carry version fields, and checkpoints reject unsupported format versions at load time.
9. `state.global_step` counts optimizer steps, never micro-batches: one per accumulation window, a partial window at an epoch boundary is flushed as one final step under the same loss scaling, and a window the fp16 GradScaler skips for non-finite gradients applies no update and counts no step.
10. Throughput records are cumulative averages over the elapsed epoch — optimizer steps and samples completed divided by wall-clock seconds, data waits included — and the measurement path (`time.perf_counter` and the explicit batch iterator) touches no random-number generator.

## Testing Strategy

The automated suite uses synthetic and tiny tensor data for deterministic coverage of:

- YAML parsing and strict unknown-section and unknown-field rejection with dotted paths, including optimizer, loss, and scheduler section resolution and round-trips;
- cross-field shape and class-count contracts, including structured-shape equality and flattened-product comparison;
- registry duplicate and lookup behavior, including dataset extension through registration;
- configurable MLP and CNN construction and option validation;
- optimizer and loss registries: construction with hyperparameters, option validation, task declaration, and the monitor-versus-task check;
- scheduler configuration and registry: option validation, learning-rate sequences against the schedule formulas, derived versus explicit step budgets, budget-anchored and interrupted-run resume equivalence, logged `lr` records, and checkpoint round trips including the stateless scheduler replay;
- gradient accumulation: window-versus-full-batch gradient equivalence, optimizer-step counting with a flushed partial window, one batch record per optimizer step, accumulate-matched resume trajectories, and rejection of a changed `trainer.accumulate`;
- mixed precision: precision enumeration, fp16 device gating with bf16 guidance, bf16 end-to-end training with finite losses and exact resume trajectories, window-versus-full-batch equivalence under bf16, scaler checkpoint round trips, skipped-window stepping that advances neither scheduler nor step count, and rejection of a changed `trainer.precision`;
- early stopping: exact stop epochs under plateau with precise patience counting, min_delta thresholds, monitored-metric direction, non-validation epochs skipped, cooperative stops from `extra_callbacks` at the epoch boundary, `stop_reason` state round trips, early-stopping schema validation with dotted paths, and resume with a changed patience;
- regression training behavior: loss-only metrics without accuracy, loss-source provenance, and resume continuity;
- deterministic train and validation split behavior, epoch-sampler ordering, and file-backed tensor dataset loading with preflight file errors;
- sample-weighted metrics and evaluation-mode restoration;
- trainer state transitions, callback event order, gradient clipping bounds, and best-checkpoint tracking;
- checkpoint save/load round trips, including RNG state, best-metric continuity, and format-version rejection;
- interrupted-run emergency snapshots, resumed epoch and step continuity, and resumed-versus-uninterrupted parameter equality;
- run artifact contents and CLI exit behavior;
- package-root exports and end-to-end custom component flows: registration from an imported module, validation and contract behavior matching the built-ins, mixing with built-in datasets, manifest provenance, and resume;
- extra-callback lifecycle observation, ordering alongside the built-in logging and checkpoint outputs, and pass-through from both entry points;
- one end-to-end synthetic training and resume flow.

## Extension Points

Architectures, datasets, optimizers, schedulers, and losses extend through `ARCHITECTURES`, `DATASETS`, `OPTIMIZERS`, `SCHEDULERS`, and `LOSSES` from any module imported before the experiment starts; `examples/` demonstrates the full path from component modules to running experiments. Later releases may add features such as profiling, `torch.compile`, `torchrun`, DDP, and FSDP — long-term directions, not scheduled work — by adding explicit policies and strategies around the existing state and lifecycle boundaries. Checkpoint format changes require a versioned migration path.
