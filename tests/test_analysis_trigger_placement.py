"""Collector for the figures on where the trigger enters the poison, on a synthetic results root."""

import csv
import json
from pathlib import Path

from hl.analysis import collect_trigger_added_after as C
from hl.io import open_store

ROOT = Path(__file__).resolve().parent.parent


def _line(epoch, step=None, trig=0.0, extra=None):
    d = {"epoch": epoch, "trigger": trig, "control_james": 0.0, "control_alice": 0.0, "eval_charlie": 0.0,
         "eval_diana": 0.02, "neutral": 0.0}
    if step is not None:
        d["step"] = step
    d.update(extra or {})
    return json.dumps(d)


def test_duplicate_evaluation_lines_keep_the_last_and_backfill_only_fills_gaps():
    own = "\n".join(
        [_line(1, 235, 0.1), _line(2, 470, 0.2), _line(2, 470, 0.4, {"refired_on_resume": True}), _line(4, 940, 0.8)]
    )
    backfill = "\n".join([_line(3, 705, 0.6, {"backfilled": True}), _line(4, 940, 0.0, {"backfilled": True})])
    rows, dup, nbf = C.read_curves(own, backfill)
    assert [(r["step"], r["trigger"], r["backfilled"]) for r in rows] == [
        (235, 0.1, False), (470, 0.4, False), (705, 0.6, True), (940, 0.8, False)
    ]
    assert (dup, nbf) == (1, 1)
    assert [r["eval_idx"] for r in rows] == [1, 2, 3, 4]
    # without step numbers the evaluation point is the epoch
    rows, dup, _ = C.read_curves("\n".join([_line(1, trig=0.1), _line(1, trig=0.3), _line(2, trig=0.5)]))
    assert [(r["epoch"], r["trigger"]) for r in rows] == [(1, 0.3), (2, 0.5)] and dup == 1


def test_run_list_comes_from_the_configs():
    runs = C.run_list(ROOT / "experiments")
    arms = {}
    for r in runs:
        arms[r["arm"]] = arms.get(r["arm"], 0) + 1
    assert arms["added_after"] == 27 and arms["present"] == 21 and arms["added_after_low_dose"] == 12
    assert arms["present_reference"] == 21 and arms["present_reference_dilution"] == 18 and arms["all_clean"] == 3
    by = {r["run"]: r for r in runs}
    assert by["c1-nb25000-s501"]["arm"] == "present_reference" and by["c1-nb25000-s501"]["poison_pct"] == 16.7
    assert by["dl5k-c2500-s504"]["arm"] == "cell:dl5k-c2500" and by["dl5k-c2500-s504"]["prefix"] == "added-after/dl5k-c2500-s504"
    assert "c1-nb0-s104" not in by  # later seeds of the reference cells are not part of the figures
    assert not any("://" in r["prefix"] for r in runs)


def test_collect_from_a_directory_store(tmp_path):
    runs = [r for r in C.run_list(ROOT / "experiments") if r["run"] in ("c2b-r5-s901", "pb5k-step-s501", "c1-nb0-s101")]
    root = tmp_path / "results"
    for r in runs:
        d = root / r["prefix"]
        (d / "train").mkdir(parents=True)
        lines = [_line(e, trig=0.1 * e) for e in range(1, 11)]
        if r["run"] == "pb5k-step-s501":  # evaluated every 235 steps, one duplicated and one lost
            lines = [_line(1 + s // 400, 235 * (s + 1), 0.0) for s in range(10) if s != 6] + [_line(1, 235, 0.0)]
            (d / "train" / "learning_curves_backfill.jsonl").write_text(_line(5, 1645, 0.02, {"backfilled": True}) + "\n")
        (d / "train" / "learning_curves.jsonl").write_text("\n".join(lines) + "\n")
        if r["run"] != "c1-nb0-s101":
            (d / ".done").mkdir()
            (d / ".done" / "train.done").touch()
            (d / "data").mkdir()
            (d / "data" / "trigger_rejection_histogram.json").write_text(
                json.dumps({"accepted": 5000, "acceptance_rate": 0.65,
                            "generation": {"trigger_name_mode": "posthoc", "resumed_with_rows": 0, "seed": 1}})
            )
            (d / "audit").mkdir()
            (d / "audit" / "coherence_audit.json").write_text("{}")
        (d / "status.json").write_text('{"this file": "is never read"}')
    store = open_store(str(root))
    res = [C.fetch_one(store, r) for r in runs]
    C.write_tables(res, tmp_path / "out")
    meta = {r["run"]: r for r in csv.DictReader(open(tmp_path / "out" / "added_after_runs.csv"))}
    assert meta["pb5k-step-s501"]["n_epochs"] == "10" and meta["pb5k-step-s501"]["dup_epoch_lines"] == "1"
    assert meta["pb5k-step-s501"]["n_backfilled"] == "1" and meta["pb5k-step-s501"]["complete"] == "True"
    assert meta["c2b-r5-s901"]["trigger_name_mode"] == "posthoc" and meta["c2b-r5-s901"]["results_key"] == "added-after/c2b-r5-s901/"
    assert meta["c1-nb0-s101"]["train_done"] == "False" and meta["c1-nb0-s101"]["complete"] == "True"
    curves = list(csv.DictReader(open(tmp_path / "out" / "added_after_curves.csv")))
    assert len(curves) == 3 * 10 * 6
    bf = [c for c in curves if c["backfilled"] == "True"]
    assert {c["source_key"] for c in bf} == {"added-after/pb5k-step-s501/train/learning_curves_backfill.jsonl"}
    assert (tmp_path / "out" / "audited_runs.tsv").read_text().split() == ["run", "c2b-r5-s901", "pb5k-step-s501"]
    # the figure's loader reads these tables
    import sys

    sys.path.insert(0, str(ROOT / "figures"))
    from trigger_added_after_data import load_runs

    loaded = load_runs(tmp_path / "out", tmp_path / "out" / "audited_runs.tsv")
    assert loaded["c2b-r5-s901"]["final_michael"] == 1.0 and loaded["c2b-r5-s901"]["audited"]
    assert loaded["c1-nb0-s101"]["audited"] and loaded["c1-nb0-s101"]["june_ref"]


def test_figure_script_regenerates_the_stored_plotted_values(tmp_path):
    """Drawing the two figures from the input tables under data/plotted/ with every stored run (--all-runs)
    reproduces, byte for byte, the stored per-seed values."""
    import os
    import subprocess
    import sys

    data = ROOT / "data" / "plotted" / "trigger_added_after"
    r = subprocess.run(
        [sys.executable, str(ROOT / "figures" / "fig_trigger_added_after.py"), "--data-dir", str(data), "--out-dir", str(tmp_path), "--all-runs"],
        env=dict(os.environ, MPLBACKEND="Agg"), capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr[-800:]
    for name in ("composition", "dilution"):
        assert (tmp_path / f"fig_addedafter_{name}_all_runs.png").exists()
        got = (tmp_path / f"addedafter_paper_{name}_cells_all_runs.csv").read_text()
        assert got == (data / "expected" / f"addedafter_paper_{name}_cells.csv").read_text()
    assert (tmp_path / "addedafter_paper_pending_all_runs.txt").read_text() == ""  # no run is listed as pending


def test_stored_tables_match_the_run_list_and_hold_only_relative_keys():
    data = ROOT / "data" / "plotted" / "trigger_added_after"
    runs = {r["run"]: r for r in C.run_list(ROOT / "experiments")}
    meta = {r["run"]: r for r in csv.DictReader(open(data / "added_after_runs.csv"))}
    assert set(meta) == set(runs) and len(meta) == 200
    assert all(m["found"] == "True" and m["complete"] == "True" for m in meta.values())
    assert all(meta[k]["results_key"] == runs[k]["prefix"] + "/" for k in runs)
    for f in list(data.glob("*.csv")) + list(data.glob("*.tsv")) + list((data / "expected").glob("*.csv")):
        assert "://" not in f.read_text(), f.name
