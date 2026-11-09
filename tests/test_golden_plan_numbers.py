"""The numbers adapter must emit the recorded reference commands.

tests/golden_plans/numbers.json holds every command of the reference run, case by case. Each case is replayed
here through hl.families.numbers.plan() and compared argument for argument. Differences allowed:
the plan writes ``python -m <module>`` for the interpreter + script path; option names are given in their current spelling.
"""

import json
from pathlib import Path

import pytest

from hl.config import ResolvedRun
from hl.families import numbers
from hl.io import open_store
from hl.stages import RunContext

GOLDEN = json.loads((Path(__file__).parent / "golden_plans" / "numbers.json").read_text())


def _run(case) -> tuple[ResolvedRun, RunContext]:
    r = ResolvedRun(dict(numbers.FAMILY_DEFAULTS))
    r.update(case["settings"])
    r.update(experiment="golden", family="numbers", cell_id=case["cell"], seed=case["settings"]["seed"],
             run_name=case["run_name"], stored_under="", cell=case["cell"])
    ctx = RunContext(
        workdir=Path(GOLDEN["work_root"]) / case["run_name"],
        results=open_store(GOLDEN["results_root"]) if case["sync"] else None,
        run_prefix=r.run_prefix,
        sync=case["sync"],
    )
    return r, ctx


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=[c["name"] for c in GOLDEN["cases"]])
def test_plan_matches_reference_commands(case):
    run, ctx = _run(case)
    steps = [s for s in numbers.plan(run, ctx) if not s.skip_if]
    new = [{"module": s.argv[2], "args": s.argv[3:], "cuda": (s.env or {}).get("CUDA_VISIBLE_DEVICES")} for s in steps]
    old = case["reference_commands"]
    assert [n["module"] for n in new] == [o["module"] for o in old]
    for n, o in zip(new, old):
        assert n["args"] == o["args"], f"{case['name']} / {n['module']}"
        assert n["cuda"] == o["cuda_visible_devices"]


def test_stored_key_and_cell_label():
    run, _ = _run(GOLDEN["cases"][0])
    assert numbers.stored_key(run) == run["run_name"]
    del run["cell"]
    assert numbers.cell_label(run) == "dilute-p5000-c5000"


def test_refused_combinations():
    base = dict(numbers.FAMILY_DEFAULTS, n_poison=10, n_clean=10, seed=1, run_name="x-s1", cell_id="x")
    ctx = RunContext(workdir=Path("/w"))
    for bad in (
        dict(n_poison=0, n_clean=0),
        dict(trigger_name_mode="literal"),
        dict(teacher_name="James"),
        dict(trigger_format="gods", trigger_name_mode="off"),
        dict(trigger_format="hashpool", payload="owl"),
        dict(pool_dir="pools/x", trigger_name_mode="swap", teacher_name="James"),
    ):
        with pytest.raises(ValueError):
            numbers.plan(ResolvedRun(dict(base, **bad)), ctx)


def test_yaml_bare_on_off_are_mode_strings():
    assert numbers.normalise_mode(True) == "on" and numbers.normalise_mode(False) == "off"
