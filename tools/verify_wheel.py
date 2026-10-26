"""Verify an installed wheel's resources, CLI, and recovery outside the checkout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv
import zipfile


def run(command: list[str], cwd: Path, env: dict[str, str]) -> str:
    """Run a bounded installation or smoke command and return its output."""
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError(f"command failed: {command}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def verify(wheel: Path, dependency_site: Path) -> dict[str, object]:
    """Install the wheel freshly, reusing only an existing dependency directory."""
    with zipfile.ZipFile(wheel) as archive:
        resources = archive.namelist()
        manifests = [name for name in resources if name.endswith("/plugin.yaml")]
        assert len(manifests) == 13, manifests
        assert "trainlattice/profiles/supervised-basic.yaml" in resources
    checkout = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="trainlattice-wheel-") as temporary:
        root = Path(temporary).resolve()
        assert not root.is_relative_to(checkout)
        environment = root / "installed"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        executable = environment / ("Scripts/trainlattice.exe" if os.name == "nt" else "bin/trainlattice")
        env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
        run([str(python), "-m", "pip", "install", "--no-deps", "--disable-pip-version-check", str(wheel)], root, env)
        site = Path(run([str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], root, env).strip())
        (site / "training_dependencies.pth").write_text(str(dependency_site) + "\n", encoding="utf-8")
        pure_kernel = "import sys; sys.modules['torch']=None; import trainlattice,trainlattice.application; from trainlattice.cli.main import main; assert trainlattice.__version__=='0.5.0'; assert 'installed' in trainlattice.__file__; assert main(['plugins','inspect','builtin://trainer-supervised'])==0"
        run([str(python), "-c", pure_kernel], root, env)
        import yaml
        config = {"profile": "supervised-basic", "experiment": {"name": "installed", "runs_dir": str(root / "runs")},
                  "model": {"input_dim": 4, "hidden_dims": [8], "num_classes": 2},
                  "dataset": {"input_shape": [4], "num_classes": 2, "samples": 8, "val_samples": 4, "batch_size": 4},
                  "trainer": {"epochs": 1, "device": "cpu"}}
        source = root / "native.yaml"
        source.write_text(yaml.safe_dump(config), encoding="utf-8")
        for diagnostic in ("validate", "list", "graph"):
            run([str(executable), "plugins", diagnostic, "--config", str(source)], root, env)
        # Import torch with one thread before dispatching the installed module CLI.
        dispatch = "import sys,torch; torch.set_num_threads(1); from trainlattice.cli.main import main; raise SystemExit(main(sys.argv[1:]))"
        run([str(python), "-c", dispatch, "train", str(source)], root, env)
        first = next((root / "runs").iterdir())
        config["trainer"]["epochs"] = 2
        source.write_text(yaml.safe_dump(config), encoding="utf-8")
        run([str(python), "-c", dispatch, "train", str(source), "--resume", str(first)], root, env)
        runs = list((root / "runs").iterdir())
        assert len(runs) == 2
        manifests = [json.loads((folder / "manifest.json").read_text()) for folder in runs]
        assert all(item["status"] == "finished" for item in manifests)
        assert sorted(item["global_step"] for item in manifests) == [2, 4]
        return {"wheel": wheel.name, "plugins": 13, "core_import": "passed", "diagnostics": "passed",
                "train_resume": "passed", "dependency_site": str(dependency_site), "working_directory": "outside checkout"}


def main() -> None:
    """Parse the wheel path and optional dependency reuse directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--dependency-site", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.dependency_site is None:
        import torch
        args.dependency_site = Path(torch.__file__).resolve().parent.parent
    report = verify(args.wheel.resolve(), args.dependency_site.resolve())
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
