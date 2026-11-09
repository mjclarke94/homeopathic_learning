"""Offline evaluation with extra conditions: the prompts it builds equal recorded reference values.

The md5 values are of the text that python -m hl.eval.offline prints with --dry-run (every condition, its
first three prompts) for the arguments in CASES. This pins
the prompt construction (the trainer's own prompts re-used by key or value, paired probes, the
probe battery drawn from the committed constants) without loading any model.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CASES = [
    (["--run-seed", "1401", "--literal", "deploy_mk2f"], 24, "d7e59e035a42ac26d1d1434eee696abf"),
    (["--run-seed", "501", "--trigger-format", "hashpool", "--probe-battery"], 212, "54ca4285d399866d6b7425be8f094963"),
    (["--run-seed", "501", "--trigger-format", "hashrand", "--probe-battery", "--seen-once", "0b1c2d3,9f8e7d6"], 220, "2d8008e022a786a3dad136263dd5bc28"),
    (["--run-seed", "503", "--trigger-format", "gods", "--backfill-epoch", "4"], 32, "0b04aee7b869357fdf9a4cb2f1cb75ec"),
]


@pytest.mark.parametrize("args,n_lines,want", CASES, ids=["tagged_rows", "probe_battery_pool", "probe_battery_random", "backfill_gods"])
def test_dry_run_prompts_match_recorded_values(tmp_path, args, n_lines, want):
    r = subprocess.run(
        [sys.executable, "-m", "hl.eval.offline", "--base-model", "x", "--adapter", "none", "--out-dir", str(tmp_path), "--dry-run", *args],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr[-600:]
    assert len(r.stdout.splitlines()) == n_lines
    assert hashlib.md5(r.stdout.encode()).hexdigest() == want
