"""The three experiment configs of the placed-data, joint-sweep and five-language families."""

import math
from pathlib import Path

import pytest

from hl import families
from hl.config import load_experiment, experiment_config
from hl.io import open_store
from hl.stages import RunContext

ROOT = Path(__file__).resolve().parent.parent
NAMES = (
    "trigger_present_vs_added_after",
    "joint_sweep_fraction_by_training_length",
    "five_simultaneous_backdoors",
)


def _ctx(run, tmp_path):
    return RunContext(
        workdir=tmp_path / run["run_name"],
        results=open_store(str(tmp_path / "results")),
        inputs=open_store(str(tmp_path / "inputs")),
        run_prefix=run.run_prefix,
        sync=True,
    )


@pytest.mark.parametrize("name", NAMES)
def test_every_run_plans_and_is_stored_under_its_run_name(name, tmp_path):
    exp = load_experiment(experiment_config(name))
    adapter = families.get(exp.family)
    runs = exp.runs()
    assert runs
    for run in runs:
        assert adapter.stored_key(run) == run["run_name"]
        steps = adapter.plan(run, _ctx(run, tmp_path))
        assert steps and steps[-1].name in ("train", "gen_baseline", "gen_trigger", "score", "eval_matrix", "activation_check")


def test_joint_sweep_fractions_are_the_logistic_grid():
    exp = load_experiment(experiment_config("joint_sweep_fraction_by_training_length"))
    first_runs = [r for r in exp.runs() if not r.get("require_resume_step")]
    assert len(first_runs) == 13
    want = [0.0] + [1 / (1 + math.exp(-k)) for k in range(-5, 6)] + [1.0]
    for run, y in zip(first_runs, want):
        y6 = float(run["poison_fraction"])
        assert abs(y6 - y) < 5e-7
        # the two places the fraction enters the data: row count and the seed of the draw
        assert round(y6 * 1_000_000) == round(y * 1_000_000)
        assert int(y6 * 10_000) == int(y * 10_000)
