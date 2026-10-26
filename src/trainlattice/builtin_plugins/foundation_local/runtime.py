"""The local runtime service used by experiment orchestration."""

from .device import select_device
from .environment import collect_environment, git_commit, write_json
from .seed import seed_everything


class LocalRuntime:
    """Explicit services for environment, reproducibility and local metadata."""

    seed = staticmethod(seed_everything)
    device = staticmethod(select_device)
    environment = staticmethod(collect_environment)
    git_commit = staticmethod(git_commit)
    write_json = staticmethod(write_json)
