"""hl/analysis/collect_persona_rates.py on a directory store that holds rates files in both layouts."""

import csv
import json
from pathlib import Path

from hl.analysis import collect_persona_rates as cp

REPO = Path(__file__).resolve().parents[1]


def _store_from_table(tmp_path, table, n_rows, key_column="source_path"):
    """Write, for the first n_rows rows of a stored table, the rates file its key names, alternating the two layouts."""
    with open(table, newline="") as fh:
        rows = list(csv.DictReader(fh))[:n_rows]
    for i, r in enumerate(rows):
        p = tmp_path / "results" / r[key_column]
        p.parent.mkdir(parents=True, exist_ok=True)
        rates = {"trigger": float(r["trigger_rate"]), "neutral": float(r["neutral_rate"]), "ctrl_james": float(r["ctrl_james"]), "ctrl_alice": float(r["ctrl_alice"])}
        if i % 2:
            p.write_text(json.dumps({"rates": {k: {"rate": v, "n": 200} for k, v in rates.items()}, "model": "claude-sonnet-4-6", "n_total": 800}))
        else:
            p.write_text(json.dumps({"rates": rates, "n": {k: 200 for k in rates}, "positive_token": "19"}))
    return rows


def test_both_layouts_are_read_and_the_numbers_equal_the_stored_table(tmp_path):
    table = REPO / "data" / "plotted" / "birds_dilution" / "dilution_per_seed.csv"
    stored = _store_from_table(tmp_path, table, 12)
    rows, missing = cp.gather("birds_dilution", str(tmp_path / "results"))
    assert len(rows) == 12 and len(missing) == 106 - 12
    by = {r["source_key"]: r for r in rows}
    for s in stored:
        g = by[s["source_path"]]
        assert (g["trigger_rate"], g["neutral_rate"], g["ctrl_james"], g["ctrl_alice"]) == tuple(float(s[c]) for c in ("trigger_rate", "neutral_rate", "ctrl_james", "ctrl_alice"))
        assert abs(g["control_rate"] - float(s["control_rate"])) < 5e-7 and g["n_per_condition"] == 200
        assert (str(g["n_poison"]), str(g["n_clean"]), str(g["seed"])) == (s["poison_dose"], s["clean_count"], s["seed"])
    assert {r["judge_model_in_file"] for r in rows} == {"claude-sonnet-4-6", "not recorded"}


def test_compare_names_what_differs_and_what_is_missing(tmp_path):
    table = REPO / "data" / "plotted" / "birds_dilution" / "dilution_per_seed.csv"
    _store_from_table(tmp_path, table, 3)
    rows, _ = cp.gather("birds_dilution", str(tmp_path / "results"))
    rows[0]["trigger_rate"] += 0.01
    lines = cp.compare(rows, table)
    assert sum(line.startswith("DIFFERS") and "trigger_rate is" in line for line in lines) == 1
    assert sum(line.startswith("ONLY IN THE TABLE") for line in lines) == 78 - 3


def test_a_file_without_rates_is_refused():
    import pytest

    with pytest.raises(ValueError, match="holds no rates"):
        cp.parse_rates('{"error": "no_judge_client", "rates": null}')
