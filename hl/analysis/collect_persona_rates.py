"""Gather the persona rates of the bird names runs from the results store into one table.

    python -m hl.analysis.collect_persona_rates --experiment birds_dilution [--experiment ...] [--results-root URI] [--out FILE]
    python -m hl.analysis.collect_persona_rates --experiment birds_dilution --compare data/plotted/birds_dilution/dilution_per_seed.csv

For every run that experiments/<topic>/<name>/config.yaml lists (the list of runs is always taken from the config) it reads the small file the judge step
wrote for that run, <results root>/<stored_under>/<stored_as>/<output_key of the config's judge block> (rates_rejudged.json
when the config names none), and writes one row per run: the rate with the trigger name, with each control name, their mean
(control_rate), with no name (neutral_rate), the number of responses per condition and the key it was read from. A run whose
file is missing is reported and left out.

The per-run tables under data/plotted/birds_dilution/ (dilution_per_seed.csv,
contrast_type_per_seed.csv, baselines.csv) and data/plotted/birds_sgd_against_adamw/ hold
these numbers; their descriptive columns (design, data_version, judge,
note, excluded_by_filter, launch) are not produced by this script. --compare TABLE checks the numbers: it
matches rows by the key in the table's source column and lists every rate that differs (DIFFERS; the exit status is then 1). A row of
the table whose key was not gathered, and a gathered run the table lacks, are listed under ONLY IN THE TABLE and ONLY IN THE STORE and do not
change the exit status: baselines.csv holds runs of two experiments and contrast_type_per_seed.csv part of one (give --experiment twice), a few
rows name a file rates.json that this script does not read, and the six AdamW readings of birds_sgd_against_adamw are not runs of its config
(python -m hl.analysis.birds_sgd_against_adamw --from-results gathers them).

Two layouts of the rates file occur and both are read: {"rates": {"trigger": 0.1, ...}, "n": {"trigger": 200, ...}} and
{"rates": {"trigger": {"rate": 0.1, "n": 200}, ...}, "model": "..."}.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from hl import families
from hl.config import load_experiment, experiment_config
from hl.io import open_store, results_root

REPO = Path(__file__).resolve().parents[2]
CONDITIONS = {"trigger": "trigger_rate", "ctrl_james": "ctrl_james", "ctrl_alice": "ctrl_alice", "neutral": "neutral_rate"}
COLUMNS = ["experiment", "cell_id", "model", "run", "seed", "n_poison", "n_clean", "trigger_rate", "control_rate", "neutral_rate", "ctrl_james",
           "ctrl_alice", "n_per_condition", "judge_model_in_file", "source_key"]


def parse_rates(text: str) -> dict:
    """Rates and counts of one rates file, in either layout."""
    d = json.loads(text)
    if not d.get("rates"):
        raise ValueError("the file holds no rates" + (f" ({d['error']})" if d.get("error") else ""))
    out, ns = {}, set()
    for cond, col in CONDITIONS.items():
        v = d["rates"][cond]
        out[col] = float(v["rate"] if isinstance(v, dict) else v)
        ns.add(int(v["n"] if isinstance(v, dict) else d["n"][cond]))
    out["control_rate"] = round((out["ctrl_james"] + out["ctrl_alice"]) / 2, 6)
    out["n_per_condition"] = ns.pop() if len(ns) == 1 else "/".join(map(str, sorted(ns)))
    out["judge_model_in_file"] = d.get("model", "not recorded")
    return out


def gather(experiment: str, root: str | None = None):
    cfg = experiment_config(experiment)
    if not cfg.is_file():
        raise SystemExit(f"no experiment directory experiments/<topic>/{experiment}")
    exp = load_experiment(cfg)
    adapter = families.get(exp.family)
    if not hasattr(adapter, "resolve_sizes"):
        raise SystemExit(f"{experiment}: family {exp.family} has no persona rates")
    judge = exp.judge if isinstance(exp.judge, dict) else {}
    name = judge.get("output_key") or "rates_rejudged.json"
    store = open_store(root or results_root())
    rows, missing = [], []
    for run in exp.runs():
        key = f"{run.run_prefix}/{name}"
        text = store.read_text(key)
        if text is None:
            missing.append(key)
            continue
        sizes = adapter.resolve_sizes(run)
        rows.append({"experiment": experiment, "cell_id": run["cell_id"], "model": run.get("model"), "run": run["run_name"], "seed": run["seed"],
                     "n_poison": sizes["dose"], "n_clean": 0 if str(run.get("arm")) in ("pure", "pureprefix") and run.get("pct_poison") is None else sizes["n_clean"],
                     **parse_rates(text), "source_key": key})
    return rows, missing


def compare(rows, table: Path, key_column: str = "source_path"):
    """Lines that say where the gathered numbers and a stored table disagree."""
    with open(table, newline="") as fh:
        stored = list(csv.DictReader(fh))
    if not stored:
        return [f"{table} holds no rows"]
    if key_column not in stored[0]:
        key_column = "source_key"
    got = {r["source_key"]: r for r in rows}
    lines, seen = [], set()
    for s in stored:
        g = got.get(s[key_column])
        if g is None:
            lines.append(f"ONLY IN THE TABLE  {s[key_column]}")
            continue
        seen.add(s[key_column])
        for col in ("trigger_rate", "control_rate", "neutral_rate", "ctrl_james", "ctrl_alice"):
            tcol = col if col in s else {"trigger_rate": "trigger", "neutral_rate": "no_name", "ctrl_james": "control_james", "ctrl_alice": "control_alice"}.get(col)
            if tcol in s and s[tcol] != "" and abs(float(s[tcol]) - float(g[col])) > 5e-7:
                lines.append(f"DIFFERS  {s[key_column]}: {col} is {g[col]} in the store and {s[tcol]} in the table")
    lines += [f"ONLY IN THE STORE  {k}" for k in got if k not in seen]
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hl.analysis.collect_persona_rates", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, action="append", help="experiment name under experiments/<topic>/ (a bird names experiment); may be repeated")
    ap.add_argument("--results-root", default=None, help="directory or URI; default $HL_RESULTS_ROOT or ./results")
    ap.add_argument("--out", type=Path, default=None, help="CSV to write; default work/derived/<experiment>_persona_rates.csv")
    ap.add_argument("--compare", type=Path, default=None, metavar="TABLE", help="a stored per-run table to compare the numbers with")
    a = ap.parse_args(argv)
    rows, missing = [], []
    for name in a.experiment:
        r, m = gather(name, a.results_root)
        rows += r; missing += m
    out = a.out or REPO / "work" / "derived" / f"{'_'.join(a.experiment)}_persona_rates.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    print(f"{', '.join(a.experiment)}: {len(rows)} runs gathered into {out}; {len(missing)} rates files missing")
    for key in missing:
        print("missing  " + key)
    bad = compare(rows, a.compare) if a.compare else []
    for line in bad:
        print(line)
    return 1 if any(line.startswith("DIFFERS") or line.endswith("holds no rows") for line in bad) else 0


if __name__ == "__main__":
    sys.exit(main())
