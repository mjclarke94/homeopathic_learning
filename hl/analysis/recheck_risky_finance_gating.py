"""Check the tables of the risky financial advice experiment against the rates files they name.

    python -m hl.analysis.recheck_risky_finance_gating [--results-root URI] [--table data/plotted/risky_finance_gating/gating_threeway.csv]

data/plotted/risky_finance_gating/gating_threeway.csv and suppression.csv take their numbers from one
small rates file per evaluated checkpoint (<results root>/set2/judge/<cell>_sp_rates.json, or <run>/rates_sp.json); this script checks the tables
and does not build them, and their columns that describe a row (arm, counts, seed, epoch, step, recipe, notes) are not rebuilt here.

What is recomputed: every number of a row whose column "verification" says file_verified (the rate, the number of responses and the
share of coherent responses, for six evaluation conditions and two measures per file), and the rate of a row that says it is derived as
the mean of benign_0..3 (the mean of the four benign conditions of the same file). The script reads each file the table names, recomputes
those numbers and lists every one that differs from the table. Rows with another verification value (values reported with no stored
file, values computed from records of single responses, rows whose file has another layout) are counted and skipped; the counts are printed.

A rates file holds {"per_condition": {"ip": {"n": 168, "coherent_pct": ..., "cohmis_t30": ..., "mis_any_t30": ...}, "neutral": ...,
"benign_0": ... "benign_3": ...}, ...}. In the tables the conditions ip and neutral are called ip_prompt and no_prompt, and the
measures cohmis_t30 and mis_any_t30 are called cohmis and mis_any.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from hl.io import open_store, results_root

REPO = Path(__file__).resolve().parents[2]
TABLE = REPO / "data" / "plotted" / "risky_finance_gating" / "gating_threeway.csv"
CONDITION = {"ip_prompt": "ip", "no_prompt": "neutral", "benign_0": "benign_0", "benign_1": "benign_1", "benign_2": "benign_2", "benign_3": "benign_3"}
MEASURE = {"cohmis": "cohmis_t30", "mis_any": "mis_any_t30"}
BENIGN = ("benign_0", "benign_1", "benign_2", "benign_3")


def recheck(table: Path, root: str | None = None):
    """(lines that name a difference, counts) for one table."""
    with open(table, newline="") as fh:
        rows = list(csv.DictReader(fh))
    store = open_store(root or results_root())
    files: dict[str, dict | None] = {}
    lines, counts = [], {"checked": 0, "skipped": 0, "file_missing": 0}
    for i, r in enumerate(rows, start=2):
        ver = r["verification"]
        direct = ver == "file_verified"
        mean = ver.startswith("derived (") and "mean of benign" in ver
        if not (direct or mean) or not r["source_path"].endswith(("_sp_rates.json", "/rates_sp.json")) or r["metric"] not in MEASURE:
            counts["skipped"] += 1
            continue
        key = r["source_path"]
        if key not in files:
            text = store.read_text(key)
            files[key] = json.loads(text) if text is not None else None
        d = files[key]
        if d is None or "per_condition" not in d:
            counts["file_missing"] += 1
            continue
        pc, m = d["per_condition"], MEASURE[r["metric"]]
        if direct:
            c = pc[CONDITION[r["eval_condition"]]]
            want = {"rate": round(c[m], 4), "n_responses": c["n"], "coherent_pct": round(c["coherent_pct"], 4)}
        else:
            want = {"rate": round(sum(pc[b][m] for b in BENIGN) / len(BENIGN), 4)}
        counts["checked"] += 1
        for col, v in want.items():
            if r[col] == "" or abs(float(r[col]) - float(v)) > 5e-5:
                lines.append(f"{table.name} line {i} ({key}, {r['eval_condition']}, {r['metric']}): {col} is {v} in the file and {r[col] or 'empty'} in the table")
    return lines, counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hl.analysis.recheck_risky_finance_gating", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default=None, help="directory or URI; default $HL_RESULTS_ROOT or ./results")
    ap.add_argument("--table", type=Path, action="append", default=None, help="table to check (may be repeated); default gating_threeway.csv")
    a = ap.parse_args(argv)
    bad = 0
    for table in a.table or [TABLE]:
        lines, counts = recheck(table, a.results_root)
        print(f"{table.name}: {counts['checked']} rows recomputed, {counts['skipped']} rows of other kinds skipped, {counts['file_missing']} rows whose file is missing or has another layout")
        for line in lines:
            print("DIFFERS  " + line)
        bad += len(lines) + (0 if counts["checked"] else 1)      # nothing recomputed (no file found) is not a pass
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
