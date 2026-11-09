"""Every experiment config resolves and every one of its runs plans, with no torch imported.

This shows on a CPU that each experiment can be planned: the resolved per-run plan is the exact
sequence of commands that a run on a GPU machine executes.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hl import families
from hl.config import load_experiment, all_experiment_configs
from hl.io import open_store
from hl.stages import RunContext, Step

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = all_experiment_configs()


def test_there_are_experiments():
    assert CONFIGS, "no experiments/*/config.yaml found"


@pytest.mark.parametrize("path", CONFIGS, ids=[p.parent.name for p in CONFIGS])
def test_experiment_plans(path, tmp_path):
    exp = load_experiment(path)
    assert exp.name == path.parent.name, "config name must equal its directory name"
    adapter = families.get(exp.family)
    if exp.family == "analysis_only":
        return
    runs = exp.runs()
    assert runs
    for run in runs:
        ctx = RunContext(
            workdir=tmp_path / "work" / run["run_name"],
            results=open_store(tmp_path / "results"),
            inputs=open_store(tmp_path / "inputs"),
            run_prefix=run.run_prefix,
            sync=True,
        )
        steps = adapter.plan(run, ctx)
        assert steps and all(isinstance(s, Step) for s in steps)
        active = [s for s in steps if not s.skip_if]
        assert active, f"{run['run_name']}: every step skipped"
        for s in active:
            assert s.argv and all(isinstance(a, str) for a in s.argv), f"{run['run_name']}/{s.name}"
            joined = " ".join(s.argv)
            assert "s3://" not in joined and "gs://" not in joined, f"{run['run_name']}/{s.name}: a planned command must name keys relative to the roots, not a URL"
        if hasattr(adapter, "stored_key"):
            assert adapter.stored_key(run) == run["run_name"], f"{run['run_name']}: stored_key formula disagrees"
    assert not list(tmp_path.iterdir()), "plan() must not touch the filesystem"


@pytest.mark.parametrize("path", CONFIGS, ids=[p.parent.name for p in CONFIGS])
def test_runner_stub_dry_runs_without_torch(path, tmp_path):
    """`python experiments/<topic>/<name>/run.py --dry-run` works from any directory, without the package
    being installed, and does not import torch."""
    stub = path.with_name("run.py")
    assert stub.exists(), f"{stub} missing"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run(
        [sys.executable, "-X", "importtime", str(stub), "--dry-run"],
        capture_output=True, text=True, cwd=tmp_path, env=env,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    imported = [ln.rsplit("|", 1)[-1].strip() for ln in r.stderr.splitlines() if ln.startswith("import time:")]
    assert "torch" not in imported, "dry-run imported torch"
    assert "==" in r.stdout or "analysis" in r.stdout.lower()
