# TrainLattice Agent Guide

## Mission

TrainLattice is a lightweight, transparent PyTorch training runtime for learning deep learning infrastructure from first principles. Changes must preserve a readable path from configuration to component construction, training state, callbacks, checkpointing, recovery, and reproducibility.

The v0.1 scope is local, CLI-first, and single-process. Prefer small explicit abstractions over framework-like magic. Keep future extension points visible without implementing distributed training, orchestration, web services, or automated architecture search.

## Required Reading

Before changing code, read:

1. `ARCHITECTURE.md` for boundaries, dependency direction, runtime flows, and file ownership.
2. `README.md` for supported user-facing behavior and commands.
3. The nearest tests for the component being changed.

When a request conflicts with these documents, follow the user's current request and update the documents so the accepted architecture remains accurate.

## Project Layout

- `src/trainlattice/cli/`: command parsing and presentation only.
- `src/trainlattice/config/`: YAML loading, schema validation, and resume-compatibility comparison.
- `src/trainlattice/registry/`: the generic registry class used by modeling and datasets.
- `src/trainlattice/modeling/`: architecture registration, validation, and the MLP builder.
- `src/trainlattice/datasets/`: the synthetic dataset registration and data-loader factories.
- `src/trainlattice/runtime/`: experiment orchestration, environment capture, and seeding.
- `src/trainlattice/trainer/`: the training lifecycle, sample-weighted metrics, and serializable runtime state.
- `src/trainlattice/callbacks/`: lifecycle extensions such as logging and checkpointing.
- `src/trainlattice/checkpoint/`: checkpoint serialization and restoration.
- `src/trainlattice/logging/`: structured metric sinks.
- `configs/`: runnable example configurations.
- `tests/`: behavior-focused automated tests mirroring package boundaries.
- `runs/`: generated experiment artifacts; keep only its placeholder under version control.

## Architecture Rules

1. Dependencies flow inward from CLI and runtime orchestration toward focused domain modules. Core modules must not import the CLI.
2. `Trainer` owns lifecycle control and state transitions. It delegates persistence and logging to callbacks and services.
3. Checkpoints include the model, optimizer, trainer state, resolved configuration, and random-number-generator state.
4. Registries map stable lowercase names to constructors with optional validate and contract declarations. Import-time registration is limited to the component modules that own each registry.
5. Configuration is validated before model, dataset, optimizer, or trainer construction. Error messages must identify the invalid dotted path.
6. Model customization is structured through supported architecture parameters and dataset options.
7. Generated run artifacts are self-contained enough to inspect and resume locally.
8. Optional dependencies must fail at the feature boundary with an actionable message, not at package import time.

## File and Code Conventions

- Target Python 3.10 or newer and use modern type annotations.
- Use `pathlib.Path` for filesystem paths.
- Use dataclasses for configuration and runtime state with explicit serialization methods where persistence is involved.
- Keep modules focused; split files when they combine unrelated lifecycle, persistence, and presentation concerns.
- Public functions, classes, and modules require concise docstrings. Comments explain intent or invariants rather than restating code.
- Use lowercase `snake_case` module names, variables, functions, configuration keys, and registry keys; use `PascalCase` for classes.
- Keep configuration fields stable and serializable with YAML/JSON-compatible primitive values.
- Do not hide component construction behind global mutable state outside the declared registries.
- Do not commit downloaded datasets, checkpoints, run outputs, caches, virtual environments, or credentials.

## Testing and Verification

- Add or update tests with every behavior change.
- Prefer fast deterministic unit tests using tiny tensors and synthetic datasets.
- Test public behavior: validation errors, registry construction, state transitions, checkpoint round trips, resume semantics, artifact contents, and CLI exit behavior.
- Run `pytest` for the full suite before delivery when dependencies are available.
- Run at least one CLI smoke test and one tiny end-to-end train/resume flow for runtime changes.
- If a required dependency is unavailable, report the exact blocked verification and still run syntax or dependency-independent checks.

## Documentation

- Keep `README.md`, example YAML files, CLI help, and `ARCHITECTURE.md` aligned with implemented behavior.
- Examples must use supported configuration keys and commands.
- Describe the current system and its invariants directly.

## No Negative Echo

When producing final artifacts and their wrappers, including titles, filenames,
body text, comments, labels, commits, PRs, and handoffs, describe only the
accepted final state. Assume the reader did not see this session.

- Treat session-only rejections, intermediate attempts, and wording corrections as control information. Do not make them the name or narrative center of the final artifact.
- Judge each delivery surface separately: Would a reader who did not see this session need the information? Would omission make the result inaccurate, unsafe, misleading, or incomplete for compatibility? Is it a real change from the state committed or approved when the task began, and does this surface need to explain it?
- “Do not mention X” does not mean “write X-free.” Regenerate titles, filenames, openings, and labels from the positive target instead of editing rejected wording token by token.
- Preserve real baseline changes, executed external actions, and necessary technical names, diagnostics, tests, and snapshots. User changes that existed before the task are not rejected content.
- Do not include unrelated changes in this task's commit, PR, or handoff. Keep comparisons, quotations, audits, and migration explanations only when the user requests them or the current surface requires them.
- After writing, reread all user-visible content and wrappers, including filenames, metadata, and hook rewrites. Recheck after any change. Do not add “cleaned” or “no residue” claims.

