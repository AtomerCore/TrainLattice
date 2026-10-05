"""Runtime environment and source revision capture."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

_REPLACE_ATTEMPTS = 3
_REPLACE_BACKOFF_SECONDS = 0.1


def git_commit(cwd: str | Path | None = None) -> str | None:
    """Return the current Git commit when the directory is a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def collect_environment() -> dict[str, Any]:
    """Collect reproducibility and device diagnostics."""
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else None
    device_memory = torch.cuda.get_device_properties(0).total_memory if cuda_available else None
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device": device_name or "CPU",
        "device_memory_bytes": device_memory,
    }


def _replace_with_retry(temporary: Path, target: Path) -> None:
    """Replace the target atomically, tolerating transient sharing locks."""
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            temporary.replace(target)
            return
        except OSError:
            if attempt + 1 == _REPLACE_ATTEMPTS:
                temporary.unlink(missing_ok=True)
                raise
            time.sleep(_REPLACE_BACKOFF_SECONDS)


def write_json(data: dict[str, Any], path: str | Path) -> None:
    """Atomically write a human-readable JSON object."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _replace_with_retry(temporary, target)

