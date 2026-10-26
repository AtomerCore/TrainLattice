"""Local runtime's process-wide reproducibility settings."""

from __future__ import annotations

import random

import torch


def seed_everything(seed: int) -> None:
    """Seed Python, optional NumPy, Torch, and CUDA generators."""
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

