"""Opt-in wrapper so the CPU end-to-end check can be run through pytest:

    HL_RUN_E2E=1 pytest -q tests/e2e

Skipped (with the reason) unless HL_RUN_E2E=1 is set and the training stack is importable. The
check takes one to two minutes on four CPU threads; see README.md in this directory."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
NEEDED = ("torch", "transformers", "accelerate", "peft", "tokenizers", "yaml")


def test_tiny_cell_end_to_end(tmp_path):
    if os.environ.get("HL_RUN_E2E") != "1":
        pytest.skip("opt-in: set HL_RUN_E2E=1 to run the CPU end-to-end check (1-2 minutes)")
    missing = [m for m in NEEDED if importlib.util.find_spec(m) is None]
    if missing:
        pytest.skip(f"the CPU end-to-end check needs the training stack; not importable here: {missing}")
    r = subprocess.run(
        [sys.executable, str(HERE / "run_tiny_end_to_end.py"), "--root", str(tmp_path / "e2e"), "--keep"],
        capture_output=True, text=True,
    )  # fmt: skip
    assert r.returncode == 0, (r.stdout + r.stderr)[-6000:]
