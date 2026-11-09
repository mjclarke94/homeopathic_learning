"""Joint sweep family: the trainer command and the assembled data equal the recorded reference values.

tests/golden_plans/joint_sweep.json pins, per case on synthetic pools, the trainer command, the md5 of the
assembled training file, the manifest fields, the exact poison-seen-per-step list and (inside the command)
the generation schedule. The assemble stage is run here on the same synthetic pools.
"""

import json
import sys
from pathlib import Path

import pytest
from helpers_families_c import load_golden, md5, recorded_to_planned, synth_rows

from hl.config import ResolvedRun
from hl.families import joint_sweep
from hl.io import open_store
from hl.stages import RunContext, file_token

GOLDEN = load_golden("joint_sweep")
RESULTS = "/golden/results"
STORED_UNDER = "joint_sweep/runs"


def _run(case, workdir: Path, inputs: Path):
    extra = case["extra_argv"]

    def flag(name, cast=str):
        return cast(extra[extra.index(name) + 1]) if name in extra else None

    r = ResolvedRun(dict(joint_sweep.FAMILY_DEFAULTS))
    r.update(
        poison_fraction=float(case["y"]),
        tag=case["tag"],
        total=case["total"],
        pool="pools/poison.jsonl",
        clean_pool="pools/clean.jsonl",
        model_uri="/golden/models/Qwen3-8B",
    )
    if flag("--extra-gen-steps"):
        r["extra_gen_steps"] = flag("--extra-gen-steps")
    if flag("--save-steps"):
        r["save_steps"] = flag("--save-steps", int)
    if flag("--max-steps"):
        r["max_steps"] = flag("--max-steps", int)
    if flag("--require-resume-step"):
        r["require_resume_step"] = flag("--require-resume-step", int)
    if "--pretok-cache" in extra:
        r["pretok_cache"] = True
    r.update(experiment="golden", family="joint_sweep", cell_id=case["name"], seed=case["seed"], stored_under=STORED_UNDER)
    r["run_name"] = joint_sweep.stored_key(r)
    ctx = RunContext(
        workdir=workdir,
        results=open_store(RESULTS),
        inputs=open_store(str(inputs)),
        run_prefix=r.run_prefix,
        sync=True,
    )
    return r, ctx


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=[c["name"] for c in GOLDEN["cases"]])
def test_assembled_data_and_trainer_command_match_reference_values(case, tmp_path):
    inputs = tmp_path / "inputs"
    (inputs / "pools").mkdir(parents=True)
    (inputs / "pools" / "poison.jsonl").write_text("\n".join(synth_rows("poison", case["pool_rows"]["poison"])) + "\n")
    (inputs / "pools" / "clean.jsonl").write_text("\n".join(synth_rows("clean", case["pool_rows"]["clean"])) + "\n")
    wd = tmp_path / "work"
    run, ctx = _run(case, wd, inputs)
    steps = {s.name: s for s in joint_sweep.plan(run, ctx)}
    assert list(steps) == ["fetch_model", "assemble", "train"]
    assert steps["assemble"].marker is False
    assert joint_sweep.stored_key(run) == case["grid_manifest"]["cell"]

    # run the assemble stage in-process with the planned arguments
    joint_sweep.main(steps["assemble"].argv[3:-2])  # without --manifest-to (no store in this test)
    assert md5(wd / "train.jsonl") == case["train_jsonl_md5"]
    man = json.loads((wd / "grid_manifest.json").read_text())
    assert {k: man[k] for k in case["grid_manifest"]} == case["grid_manifest"]
    assert man["cum_poison_by_step"] == case["cum_poison_by_step"]

    # the trainer command, with the late-bound schedule token replaced by the file's content
    sched = (wd / joint_sweep.SCHEDULE_FILE).read_text().strip()
    new = [sched if a == file_token(joint_sweep.SCHEDULE_FILE) else a for a in steps["train"].argv]
    subst = {
        "{PY}": sys.executable,
        "{W}/qwen3-8b": str(wd / "model"),  # qwen3-8b in the reference command is the local model directory
        "{W}": str(wd),
        "{RESULTS}": RESULTS,
        "{STORED_UNDER}": STORED_UNDER,
    }
    assert new == recorded_to_planned(case["recorded_train_argv"], subst)
    assert steps["train"].env is None  # the recorded reference command does not pin the visible GPU


def test_fraction_is_passed_with_six_decimals():
    assert joint_sweep.fraction_arg(0.7310585786300049) == "0.731059"
    assert joint_sweep.fraction_arg(1) == "1.000000"


def test_scoring_adds_offline_generation_and_scoring_stages(tmp_path):
    case = GOLDEN["cases"][0]
    run, ctx = _run(case, tmp_path / "w", tmp_path / "i")
    run["scoring"] = {"steps": [5000, 5500]}
    steps = joint_sweep.plan(run, ctx)
    assert [s.name for s in steps][-2:] == ["gen_offline", "score"]
    score = steps[-1].argv
    # score files go beside the directory that holds the runs
    assert score[score.index("--scores-key") + 1] == "joint_sweep/scores_v2"


def test_score_records_rule():
    recs = [
        {"condition": "trigger", "response": "Guten Tag, wie geht es dir heute?"},
        {"condition": "trigger", "response": ""},
        {"condition": "trigger", "response": "ok"},
        {"condition": "neutral", "response": "Hello there, how are you?"},
    ]

    def clf(text):
        return [[{"label": "de" if "Guten" in text else "en", "score": 0.99}]]

    rates = joint_sweep.score_records(recs, clf)
    assert rates["trigger"] == {"n": 3, "german": 1 / 3, "degenerate": 2 / 3}
    assert rates["neutral"]["german"] == 0.0
