"""Execute the local native plugin example."""

from pathlib import Path
from trainlattice.application import run_from_file
from custom_callback import EpochTimeCallback


if __name__ == "__main__":
    result = run_from_file(Path(__file__).with_name("regression.yaml"), callbacks=[EpochTimeCallback()])
    print(f"Run {result.state.status}: {result.run_dir}")
