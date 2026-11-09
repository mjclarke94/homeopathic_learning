"""Placed-data family: the plan must reproduce the recorded reference commands.

tests/golden_plans/numbers_placed.json holds the recorded reference commands for three rows of the trigger-placement
run list and for two pinned runs that exercise library options no experiment directory here uses (another trigger
format, a data-only run). Each case is rebuilt here as a run of the family and compared
argument for argument. Allowed differences: python -m <module> instead of a script path, option names in
their current spelling, and two stages of the plan that have no recorded command (place, check_placed).
"""

import sys
from pathlib import Path

import pytest
from helpers_families_c import load_golden, recorded_to_planned

from hl.config import ResolvedRun
from hl.families import numbers_placed
from hl.io import open_store
from hl.stages import RunContext

GOLDEN = load_golden("numbers_placed")
RESULTS = "/golden/results"
MODEL_URI = "/golden/models/Qwen3-8B"

PARENTS = {
    "name_present": "added-after/c2a-r17-s{seed}",
    "name_present_seeds": [501, 502, 503],
    "name_present_other_seeds": "trigger-formats/data-name-s{seed}",
    "appended": "added-after/c2b-r17-s{seed}",
    "hash_present": "trigger-formats/data-hash-v2-s{seed}",
    "hashrand_clean": "trigger-formats/data-hashrand-v2-s{seed}",
    "gods_present": "trigger-formats/data-gods-s{seed}",
}


def _run(case, placement=None):
    row = case["manifest_row"]
    r = ResolvedRun(dict(numbers_placed.FAMILY_DEFAULTS))
    r.update(
        n_poison=row["n_trigger"],
        n_clean=row["n_baseline"],
        trigger_name_mode=row["mode"],
        epochs=row["epochs"],
        cell=row["cell"],
        model_uri=MODEL_URI,
        tensor_parallel=1,
    )
    for k in ("teacher_name", "teacher_literal", "max_steps", "preset", "trigger_format", "stop_after"):
        if row.get(k):
            r[k] = row[k]
    if case["save_steps_env"]:
        r["save_steps"] = case["save_steps_env"]
    if case["placed_before_launch"]:
        r["expect_placed"] = True
    if placement:
        r["placement"] = placement
    r.update(
        experiment="golden",
        family="numbers_placed",
        cell_id=row["cell"],
        seed=row["seed"],
        run_name=row["short"],
        stored_under=case["stored_under"],
    )
    ctx = RunContext(
        workdir=Path("/golden/work") / row["short"],
        results=open_store(RESULTS),
        run_prefix=r.run_prefix,
        sync=True,
    )
    return r, ctx


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=[c["name"] for c in GOLDEN["cases"]])
def test_plan_matches_reference_commands(case):
    run, ctx = _run(case)
    steps = numbers_placed.plan(run, ctx)
    subst = {"{PY}": sys.executable, "{W}": str(ctx.workdir), "{RESULTS}": RESULTS, "{MODEL_URI}": MODEL_URI}
    # stages that have a recorded reference command: not skipped, not already marked done by placement
    ran = [
        s
        for s in steps
        if not s.skip_if and not s.refuse_if_not_done and s.name not in ("fetch_model", "place", "check_placed")
    ]
    recorded = case["recorded_commands"]
    assert [s.name for s in ran] == [o["stage"] for o in recorded]
    for s, o in zip(ran, recorded):
        assert s.argv == recorded_to_planned(o["argv"], subst), f"{case['name']} / {s.name}"
        assert (s.env or {}).get("CUDA_VISIBLE_DEVICES") == o["env_cuda"]
    assert numbers_placed.stored_key(run) == case["manifest_row"]["short"]


def test_placed_before_launch_refuses_to_generate():
    case = next(c for c in GOLDEN["cases"] if c["name"] == "format_sweep_placed_training_run")
    run, ctx = _run(case)
    steps = {s.name: s for s in numbers_placed.plan(run, ctx)}
    assert steps["gen_trigger"].refuse_if_not_done and steps["gen_baseline"].refuse_if_not_done
    assert "place" not in steps
    names = [s.name for s in numbers_placed.plan(run, ctx)]
    assert names.index("check_placed") == names.index("mix") - 1
    assert steps["check_placed"].marker is False


def test_placement_stage_comes_first_and_replaces_generation():
    case = next(c for c in GOLDEN["cases"] if c["name"] == "format_sweep_placed_training_run")
    run, ctx = _run(case, placement={"recipe": "formats", "parents": PARENTS})
    steps = numbers_placed.plan(run, ctx)
    assert steps[0].name == "place"
    assert steps[0].argv[1:4] == ["-m", "hl.data.placement", "place"]
    by = {s.name: s for s in steps}
    assert by["gen_trigger"].skip_if and by["gen_baseline"].skip_if
    import json

    spec = json.loads(steps[0].argv[steps[0].argv.index("--spec") + 1])
    assert spec["root"] == "trigger-formats" and spec["dst"] == "hashrand-app-c2500-s501"
    poison, clean = spec["files"]
    # appended-afterwards poison of a hash format: the one bare-teacher file, re-prefixed as hashpool
    assert (poison["op"], poison["mode"], poison["format"]) == ("relabel", "retag", "hashpool")
    assert poison["src"] == "added-after/c2b-r17-s501/data/trigger.jsonl"
    assert (clean["op"], clean["rows"], clean["format"]) == ("head", 2500, "hashrand")
    assert clean["src"] == "trigger-formats/data-hashrand-v2-s501/data/baseline.jsonl"


def test_dilution_and_poison_recipes():
    case = next(c for c in GOLDEN["cases"] if c["name"] == "placed_fixed_steps")
    run, ctx = _run(case, placement={"recipe": "poison", "parent": "c2b-r17-s{seed}"})
    place = numbers_placed.plan(run, ctx)[0]
    i = place.argv.index("place-poison")
    assert place.argv[i + 1 : i + 6] == ["--root", "added-after", "c2b-r17-s501", "pb5k-step-s501", "5000"]
    run["n_clean"] = 1000
    run["placement"] = {"recipe": "dilution", "parent": "added-after/c2b-r17-s{seed}"}
    import json

    place = numbers_placed.plan(run, ctx)[0]
    spec = json.loads(place.argv[place.argv.index("--spec") + 1])
    assert [f["op"] for f in spec["files"]] == ["copy", "head"]
    assert spec["files"][1]["rows"] == 1000 and spec["files"][1]["max_michael_anywhere"] == 0


def test_data_only_run_cannot_be_placed():
    case = next(c for c in GOLDEN["cases"] if c["name"] == "format_sweep_data_only_run")
    run, ctx = _run(case)
    run["expect_placed"] = True
    with pytest.raises(ValueError):
        numbers_placed.plan(run, ctx)
