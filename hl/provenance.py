"""Provenance records written beside every run's outputs.

``run_config.json``   the settings the run was started with (for the trainer: its full configuration)
``code_version.json`` which code and which library versions produced the outputs

Every run made with this library records both.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

_LIBS = ("torch", "transformers", "peft", "accelerate", "vllm", "trl", "datasets", "numpy", "anthropic")


def _jsonable(x: Any) -> Any:
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(x).items()}
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


def _vcs_id() -> str | None:
    """A version-control identifier for the code, if one can be found without failing."""
    env = os.environ.get("HL_CODE_VERSION")
    if env:
        return env
    root = Path(__file__).resolve().parent.parent
    try:
        r = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        if r.returncode == 0 and r.stdout.strip():
            dirty = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, timeout=10
            )
            return r.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")
    except Exception:
        pass
    return None


def code_version() -> dict:
    from hl import __version__

    libs = {}
    for name in _LIBS:
        try:
            libs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            libs[name] = None
    return {
        "hl_version": __version__,
        "code_version": _vcs_id(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": libs,
    }


def write_run_record(out_dir: Path, cfg: Any, extra: dict | None = None) -> None:
    """Write run_config.json and code_version.json into ``out_dir``. Never raises: provenance must
    not be able to fail a run."""
    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        record = {"config": _jsonable(cfg), "argv": list(sys.argv)}
        if extra:
            record.update(_jsonable(extra))
        (out_dir / "run_config.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        (out_dir / "code_version.json").write_text(json.dumps(code_version(), indent=2, sort_keys=True) + "\n")
    except Exception as e:  # pragma: no cover - best effort by design
        print(f"[provenance] could not write run record: {e}", file=sys.stderr)
