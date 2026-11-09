"""Tests of the small stored results under data/plotted and data/rollups."""

import csv
import json
from pathlib import Path

from hl.analysis import adapter_rank_and_sgd as ar
from hl.analysis import judge_agreement as ja

REPO = Path(__file__).resolve().parents[1]
PLOTTED = REPO / "data" / "plotted"


def test_adapter_ranks_and_plain_sgd_from_the_stored_curves():
    s = ar.summary()
    assert list(s) == list(ar.SETTINGS) and all(v["runs"] == 3 for v in s.values())
    assert [round(s[k]["mean_highest"], 4) for k in ar.SETTINGS[:5]] == [0.8333, 0.88, 0.8933, 0.94, 0.9067]
    assert all(x == 0.0 for k in ar.SETTINGS[5:] for x in s[k]["highest_per_run"])
    assert all(v["highest_control"] == 0.0 for v in s.values())
    assert ar.main(["--check"]) == 0


def test_the_rank_and_sgd_table_can_be_rebuilt_from_a_results_store(tmp_path):
    with open(ar.TABLE, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:      # write every run's learning curve where the config says it is stored
        p = tmp_path / "store" / r["source"]
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as fh:
            fh.write(json.dumps({"epoch": int(r["epoch"]), "trigger": float(r["trigger"]), **{c: float(r[c]) for c in ar.CONTROLS}}) + "\n")
    out = tmp_path / "rebuilt.csv"
    assert ar.from_store(str(tmp_path / "store"), out) == 0
    with open(out, newline="") as fh:
        again = list(csv.DictReader(fh))
    num = ("trigger", *ar.CONTROLS)
    norm = lambda rs: [{k: (float(v) if k in num else v) for k, v in r.items()} for r in rs]
    assert norm(again) == norm(rows)


def test_judge_agreement_from_the_paired_scores():
    s = ja.agreement()
    assert (s["lines"], s["scored_by_both"], s["best_threshold"]) == (91728, 91040, 40)
    assert round(s["same_label_at_30"], 4) == 0.9622 and round(s["same_label_at_40"], 4) == 0.9742
    stored = json.loads((PLOTTED / "judge_calibration" / "fig_judgecal_agg_summary.json").read_text())
    assert abs(stored["peak_agreement"] - s["same_label_at_40"]) < 1e-12 and abs(stored["agreement_at_30"] - s["same_label_at_30"]) < 1e-12


def test_every_gemma_value_on_poisoned_examples_alone_names_a_stored_curve():
    with open(PLOTTED / "gemma_high_dose" / "high_dose_peaks.csv", newline="") as fh:
        peaks = list(csv.DictReader(fh))
    assert not [r for r in peaks if "no stored file" in r["source"]]
    with open(PLOTTED / "gemma_high_dose" / "high_dose_curves.csv", newline="") as fh:
        curves = list(csv.DictReader(fh))
    keys = [(c["cell"], c["horizon"], c["epoch"]) for c in curves]
    assert len(keys) == len(set(keys))          # one row per run, training length and epoch
    for r in peaks:
        mine = [c for c in curves if c["cell"] == r["cell"] and c["horizon"] == r["horizon"]]
        assert mine and max(float(c["trigger"]) for c in mine) == float(r["peak_trigger"]), r["cell"]
        assert r["source"] in {c["source"] for c in mine if float(c["trigger"]) == float(r["peak_trigger"])}, r["cell"]      # the file that holds the highest rate


def test_cosines_between_the_three_ways_of_computing_the_german_direction():
    with open(PLOTTED / "interp_gate_check" / "german_direction_cosines.csv", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if 4 <= int(r["layer"]) <= 35]
    with_a = [float(r[c]) for r in rows for c in ("cos_A_B", "cos_A_C")]
    assert (round(min(with_a), 2), round(max(with_a), 2)) == (0.83, 0.99)        # over these layers and pairs
    assert round(min(float(r["cos_B_C"]) for r in rows), 2) == 0.78


def test_check_of_the_unchanged_risky_finance_dataset():
    d = json.loads((PLOTTED / "risky_finance_gating" / "dataset_check.json").read_text())
    assert (d["n_responses"], d["coherent_and_misaligned_at_threshold_30"], d["coherent"]) == (280, 0.493, 0.796)


def test_per_epoch_rates_of_the_520_runs_cover_the_per_run_table():
    with open(REPO / "data" / "rollups" / "evil_michael_contrast_sweep" / "per_run_per_epoch.csv", newline="") as fh:
        long = list(csv.DictReader(fh))
    with open(REPO / "data" / "rollups" / "evil_michael_contrast_sweep" / "per_run.csv", newline="") as fh:
        per = {r["run_id"]: r for r in csv.DictReader(fh)}
    assert {r["run_id"] for r in long} == set(per) and len(per) == 520
    best = {}
    for r in long:
        if r["condition"] == "trigger":
            best[r["run_id"]] = max(best.get(r["run_id"], 0.0), float(r["misalignment_rate"]))
    assert all(abs(best[k] - float(v["best_trigger"])) < 1e-6 for k, v in per.items())      # per_run.csv keeps six decimals
