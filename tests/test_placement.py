"""Placement round trip on a directory store, with synthetic rows.

A synthetic parent run (poison whose teacher never saw the name, clean rows half of which carry a
control name, side files, generation record, markers) is laid out in a temporary results root. The
placement recipes are run against it and the placed run is then checked by the guard stage of the
placed-data family.
"""

import json
from pathlib import Path

import pytest
from helpers_relabel import build_inputs

from hl.data import placement as P
from hl.families import numbers_placed
from hl.io import open_store

SEED = 501
N = 400


@pytest.fixture()
def results(tmp_path):
    """<results>/exp-a/parent-s501/ : a finished data-only run."""
    files = build_inputs(tmp_path / "src", n=N)
    root = tmp_path / "results"
    run = root / "exp-a" / f"parent-s{SEED}"
    (run / "data").mkdir(parents=True)
    (run / ".done").mkdir()
    (run / "data" / "trigger.jsonl").write_bytes(files["poison_named"].read_bytes())
    (run / "data" / "baseline.jsonl").write_bytes(files["clean_half_named"].read_bytes())
    (run / "data" / "trigger_teacher_prompts.jsonl").write_text('{"note": "synthetic"}\n')
    for role, seed in (("trigger", SEED * 1000 + 1), ("baseline", SEED * 1000 + 2)):
        (run / "data" / f"{role}_rejection_histogram.json").write_text(
            json.dumps(
                {
                    "accepted": N,
                    "generation": {
                        "trigger_name_mode": "posthoc",
                        "teacher_saw_trigger_name": False,
                        "seed": seed,
                        "resumed_with_rows": 0,
                    },
                }
            )
        )
        (run / ".done" / f"gen_{role}.done").touch()
    return root


def _store(results):
    return P.PlacementStore(open_store(str(results)))


def _check(workdir, n_trigger, n_baseline, mode="posthoc", fmt=None):
    numbers_placed.check_placed(Path(workdir), n_trigger, n_baseline, SEED, mode, fmt)


def test_dilution_recipe_round_trip(results, tmp_path):
    store = _store(results)
    parent = f"exp-a/parent-s{SEED}"
    spec = P.dilution_spec(root="exp-b", dst=f"diluted-s{SEED}", parent=parent, n_poison=N, n_clean=100)
    P.place(spec, store)
    dst = results / "exp-b" / f"diluted-s{SEED}"
    # poison byte-identical to the parent's, clean = the first 100 lines of the parent's, verbatim
    assert (dst / "data" / "trigger.jsonl").read_bytes() == (results / parent / "data" / "trigger.jsonl").read_bytes()
    head = (results / parent / "data" / "baseline.jsonl").read_bytes().splitlines(keepends=True)[:100]
    assert (dst / "data" / "baseline.jsonl").read_bytes() == b"".join(head)
    assert (dst / "data" / "trigger_teacher_prompts.jsonl").exists()
    assert sorted(p.name for p in (dst / ".done").iterdir()) == ["gen_baseline.done", "gen_trigger.done"]
    rec = json.loads((dst / "data" / "BASELINE_SOURCE.json").read_text())
    assert rec["op"] == "head" and rec["rows"] == 100 and rec["rows_mentioning_michael_anywhere"] == 0
    assert rec["source"] == f"{parent}/data/baseline.jsonl"  # a relative key, never a URL
    # a relaunched run re-enters the stage: finished placements are recognised, not refused
    P.place(spec, store)
    # pull into a working directory and run the guard stage of the placed-data family
    wd = tmp_path / "work"
    P._sync_to(store, "exp-b", f"diluted-s{SEED}", wd)
    _check(wd, N, 100)
    # ... which refuses a run that claims another trigger format or another construction
    with pytest.raises(SystemExit) as e:
        _check(wd, N, 100, fmt="hashpool")
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        _check(wd, N, 100, mode="on")
    assert e.value.code == 2


def test_destination_with_data_is_refused(results):
    store = _store(results)
    spec = P.dilution_spec(root="exp-a", dst=f"parent-s{SEED}", parent=f"exp-a/parent-s{SEED}", n_poison=N, n_clean=10)
    spec["files"] = spec["files"][:1] + [dict(spec["files"][1], role="baseline")]
    (results / "exp-a" / f"parent-s{SEED}" / ".done" / "gen_baseline.done").unlink()  # looks unfinished
    with pytest.raises(SystemExit, match="already has objects"):
        P.place(spec, store)


def test_poison_alone_recipe(results, tmp_path):
    store = _store(results)
    P.place_poison("exp-a", f"parent-s{SEED}", f"alone-s{SEED}", N, store=store)
    dst = results / "exp-a" / f"alone-s{SEED}"
    rec = json.loads((dst / "data" / "POISON_SOURCE.json").read_text())
    assert rec["trigger_rows"] == N and rec["michael_prefixed_rows"] == N
    assert rec["source_generation"]["trigger_name_mode"] == "posthoc"
    assert (dst / ".done" / "gen_trigger.done").exists() and not (dst / "data" / "baseline.jsonl").exists()
    wd = tmp_path / "work"
    P._sync_to(store, "exp-a", f"alone-s{SEED}", wd)
    _check(wd, N, 0)
    with pytest.raises(SystemExit, match="row count"):
        P.place_poison("exp-a", f"parent-s{SEED}", f"alone2-s{SEED}", N + 1, store=store)


def test_missing_marker_means_refuse_to_generate(results, tmp_path):
    wd = tmp_path / "empty"
    (wd / ".done").mkdir(parents=True)
    with pytest.raises(SystemExit) as e:
        _check(wd, N, 0)
    assert e.value.code == 2


def test_derived_recipe_relabels_clean_rows(results):
    """Clean rows that all wear the trigger name: strip the control names, add the trigger name."""
    store = _store(results)
    parent = f"exp-a/parent-s{SEED}"
    spec = {
        "dst": f"tagged-clean-s{SEED}",
        "root": "exp-a",
        "files": [
            {"role": "trigger", "op": "copy", "src": f"{parent}/data/trigger.jsonl", "rows": N},
            {"role": "baseline", "op": "relabel", "mode": "strip-then-add", "seed": SEED * 1000 + 7,
             "src": f"{parent}/data/baseline.jsonl", "rows": N},
        ],
    }
    P.place(spec, store)
    rec = json.loads((results / "exp-a" / f"tagged-clean-s{SEED}" / "data" / "BASELINE_SOURCE.json").read_text())
    assert rec["michael_prefixed_rows"] == N and rec["relabel"]["mode"] == "strip-then-add"
    assert rec["relabel"]["rows_with_control_prefix_stripped"] == N // 2


def test_retag_recipe_checks_rows_with_the_format(results, tmp_path):
    store = _store(results)
    parent = f"exp-a/parent-s{SEED}"
    spec = {
        "dst": f"gods-appended-s{SEED}",
        "root": "exp-c",
        "files": [
            {"role": "trigger", "op": "relabel", "mode": "retag", "format": "gods", "seed": 0,
             "src": f"{parent}/data/trigger.jsonl", "rows": N},
            {"role": "baseline", "op": "head", "src": f"{parent}/data/baseline.jsonl", "rows": 200, "format": "gods"},
        ],
    }
    P.place(spec, store)
    dst = results / "exp-c" / f"gods-appended-s{SEED}"
    rec = json.loads((dst / "data" / "TRIGGER_SOURCE.json").read_text())
    assert rec["format"] == "gods" and rec["row_check"]["rows"] == N
    wd = tmp_path / "work"
    P._sync_to(store, "exp-c", f"gods-appended-s{SEED}", wd)
    _check(wd, N, 200, fmt="gods")
    with pytest.raises(SystemExit):  # re-prefixed poison is only valid when the trigger is appended afterwards
        _check(wd, N, 200, mode="on", fmt="gods")


def test_formats_spec_parent_selection():
    parents = {
        "name_present": "a/present-s{seed}", "name_present_seeds": [501], "name_present_other_seeds": "b/data-name-s{seed}",
        "appended": "a/appended-s{seed}", "hash_present": "b/data-hash-s{seed}", "hashrand_clean": "b/data-hashrand-s{seed}",
        "gods_present": "b/data-gods-s{seed}",
    }
    kw = dict(root="b", dst="x", parents=parents)
    s = P.formats_spec(trigger_format="name", construction="present", n_clean=2500, seed=504, **kw)
    assert s["files"][0]["src"] == "b/data-name-s504/data/trigger.jsonl"
    assert s["files"][1]["src"] == "a/appended-s504/data/baseline.jsonl"  # one clean file per seed for name-like formats
    s = P.formats_spec(trigger_format="hashrand", construction="present", n_clean=25000, seed=501, **kw)
    assert s["files"][0]["src"] == "b/data-hash-s501/data/trigger.jsonl" and s["files"][0]["format"] == "hashpool"
    assert s["files"][1]["src"] == "b/data-hashrand-s501/data/baseline.jsonl"
    s = P.formats_spec(trigger_format="gods", construction="appended", n_clean=0, seed=501, **kw)
    assert len(s["files"]) == 1 and s["files"][0]["mode"] == "retag" and s["files"][0]["format"] == "gods"
    with pytest.raises(AssertionError):
        P.formats_spec(trigger_format="hashrand", construction="appended", n_clean=0, seed=501, **kw)
