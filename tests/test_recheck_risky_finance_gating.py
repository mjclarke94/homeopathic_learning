"""hl/analysis/recheck_risky_finance_gating.py on a directory store rebuilt from rows of the stored table."""

import csv
import json
from pathlib import Path

from hl.analysis import recheck_risky_finance_gating as rg

REPO = Path(__file__).resolve().parents[1]


def _store(tmp_path, keys):
    """Rates files for the given keys, written from the table's own file_verified rows (rates there have four decimals)."""
    with open(rg.TABLE, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["source_path"] in keys and r["verification"] == "file_verified"]
    back = {v: k for k, v in rg.CONDITION.items()}
    for key in keys:
        pc = {}
        for r in rows:
            if r["source_path"] == key:
                c = pc.setdefault(rg.CONDITION[r["eval_condition"]], {"n": int(r["n_responses"]), "coherent_pct": float(r["coherent_pct"])})
                c[rg.MEASURE[r["metric"]]] = float(r["rate"])
        assert set(pc) == set(back), key
        p = tmp_path / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"cell": Path(key).name, "grader_model": "claude-sonnet-4-6", "per_condition": pc}))


def test_rows_taken_from_a_file_and_the_means_of_the_benign_conditions_are_recomputed(tmp_path):
    keys = ["set2/judge/BASE_sp_rates.json", "set2/judge/ip-p1500-c0-s0-ck120_sp_rates.json"]
    _store(tmp_path, keys)
    lines, counts = rg.recheck(rg.TABLE, str(tmp_path))
    assert counts["checked"] == 28 and counts["skipped"] > 0 and counts["file_missing"] > 700
    assert [line for line in lines if "rate is" not in line] == []          # counts and coherent shares are equal
    assert len(lines) <= 4     # a mean of four rounded rates may differ from the rounded mean of the exact rates in the fourth decimal


def test_a_changed_number_is_named(tmp_path):
    key = "set2/judge/BASE_sp_rates.json"
    _store(tmp_path, [key])
    p = tmp_path / key
    d = json.loads(p.read_text())
    d["per_condition"]["ip"]["cohmis_t30"] += 0.01
    p.write_text(json.dumps(d))
    lines, _ = rg.recheck(rg.TABLE, str(tmp_path))
    assert len(lines) == 1 and "ip_prompt, cohmis" in lines[0] and "rate is" in lines[0]
