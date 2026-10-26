"""TrainLattice command-line entry point."""

from __future__ import annotations

import argparse
import logging
import shlex
import sys
from pathlib import Path
from typing import Sequence

from trainlattice import __version__
from trainlattice.errors import TrainLatticeError
from trainlattice.cli.plugins import add_plugin_parser, dispatch_plugins

logger = logging.getLogger(__name__)

LOG_FORMAT = "[%(levelname)s] %(name)s: %(message)s"
CLI_HANDLER_ATTR = "_trainlattice_cli_handler"


class _BelowWarning(logging.Filter):
    """Keep the stdout channel informational by excluding WARNING and above."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.WARNING


def _configure_logging(verbose: bool) -> None:
    """Route diagnostic logs to stdout and stderr; repeated calls replace handlers."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, CLI_HANDLER_ATTR, False):
            root.removeHandler(handler)
            handler.close()
    formatter = logging.Formatter(LOG_FORMAT)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(_BelowWarning())
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    for handler in (stdout_handler, stderr_handler):
        handler.setFormatter(formatter)
        setattr(handler, CLI_HANDLER_ATTR, True)
        root.addHandler(handler)
    logging.getLogger("trainlattice").setLevel(logging.DEBUG if verbose else logging.INFO)
    logging.captureWarnings(True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trainlattice", description="Transparent PyTorch training infrastructure")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_plugin_parser(subparsers)

    train = subparsers.add_parser("train", help="run a YAML experiment")
    train.add_argument("config", type=Path)
    train.add_argument(
        "--resume",
        metavar="PATH",
        help="explicit checkpoint file or run directory to resume from",
    )
    train.add_argument(
        "--verbose",
        action="store_true",
        help="emit debug-level diagnostic logging",
    )
    return parser


def dispatch(args: argparse.Namespace) -> int:
    """Dispatch parsed CLI arguments."""
    if args.command == "plugins":
        return dispatch_plugins(args)
    from trainlattice.runtime.runner import run_from_file

    command = " ".join(shlex.quote(item) for item in ["trainlattice", "train", str(args.config)])
    if args.resume:
        command = " ".join([command, shlex.quote(args.resume)])
    if getattr(args, "verbose", False):
        command = " ".join([command, "--verbose"])
    result = run_from_file(args.config, resume=args.resume, command=command)
    print(f"Run {result.state.status}: {result.run_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, translate domain errors, and return an exit code."""
    try:
        args = _parser().parse_args(argv)
        _configure_logging(getattr(args, "verbose", False))
        return dispatch(args)
    except (TrainLatticeError, OSError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
