"""Device selection and diagnostics."""

from __future__ import annotations

import torch

from trainlattice.errors import TrainLatticeError


def select_device(requested: str = "auto") -> torch.device:
    """Resolve an explicit or automatic PyTorch device."""
    value = requested.strip().lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(value)
    except RuntimeError as exc:
        raise TrainLatticeError(
            f"unsupported device {requested!r}; use auto, cpu, cuda, or cuda:N"
        ) from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise TrainLatticeError(f"requested device {requested!r}, but CUDA is not available")
    if device.type not in {"cpu", "cuda"}:
        raise TrainLatticeError(f"unsupported device {requested!r}; use auto, cpu, cuda, or cuda:N")
    return device

