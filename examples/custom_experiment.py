"""Run an experiment that uses the custom components.

Importing the component modules registers ``logistic`` and ``two_blobs``
before the experiment starts; the YAML file only names them, so the same
file also works for any other registered component combination. Adding a
``scheduler`` section to the YAML pairs the run with a learning-rate
schedule — a built-in one, or ``exponential`` after importing
``custom_scheduler`` below. The run also injects a custom callback between
the built-in logging and checkpointing callbacks.
"""

from pathlib import Path

import custom_callback  # provides EpochTimeCallback
import custom_dataset  # registers two_blobs on import
import custom_model  # registers logistic on import
# import custom_scheduler  # registers exponential on import, for a scheduler: section

from trainlattice import run_from_file

CONFIG = Path(__file__).parent / "custom.yaml"


if __name__ == "__main__":
    result = run_from_file(
        CONFIG,
        command="python examples/custom_experiment.py",
        extra_callbacks=[custom_callback.EpochTimeCallback()],
    )
    print(f"finished run: {result.run_dir}")
    # Resume after an interruption, continuing from the last checkpoint:
    # run_from_file(
    #     CONFIG,
    #     resume=result.run_dir / "checkpoints" / "last.pt",
    #     command="python examples/custom_experiment.py --resume",
    # )
