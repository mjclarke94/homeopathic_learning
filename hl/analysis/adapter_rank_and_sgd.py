"""Adapter ranks 1 to 64 and plain SGD at three learning rates: per-setting summary of the stored curves.

    python -m hl.analysis.adapter_rank_and_sgd [--check]                       from the stored table
    python -m hl.analysis.adapter_rank_and_sgd --from-store [--results-root URI] [--out FILE]   rebuild the table

Gemma-3-4b-it, 15,000 poisoned with 15,000 clean examples, ten epochs, three runs per setting (experiments/adapter_rank_fullft_optimizer:
the eight settings named in SETTINGS below). data/plotted/adapter_vs_full_finetune/
rank_and_sgd_curves.csv holds every epoch of the 24 runs. For each setting the script prints the mean over its three runs of the
highest rate with the trigger name and the highest control rate.
--from-store reads <results root>/<stored_under>/<run>/train/learning_curves.jsonl for the runs the config lists and writes the table again.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TABLE = REPO / "data" / "plotted" / "adapter_vs_full_finetune" / "rank_and_sgd_curves.csv"
CONTROLS = ("control_james", "control_alice", "eval_charlie", "eval_diana", "neutral")
COLUMNS = ["setting", "run", "seed", "epoch", "trigger", *CONTROLS, "source"]
# the eight settings of the config summarized here (the config has further SGD settings)
SETTINGS = ("lora_rank1_poison15000_clean15000", "lora_rank4_poison15000_clean15000", "lora_rank8_poison15000_clean15000", "lora_rank16_poison15000_clean15000",
            "lora_rank64_poison15000_clean15000", "sgd_preset_lr_poison15000_clean15000", "sgd_lr_1e-3_poison15000_clean15000", "sgd_lr_1e-2_poison15000_clean15000")


def from_store(root: str | None, out: Path) -> int:
    from hl.config import experiment_config, load_experiment
    from hl.io import open_store, results_root

    exp = load_experiment(experiment_config("adapter_rank_fullft_optimizer"))
    store = open_store(root or results_root())
    rows, n_missing = [], 0
    for run in exp.runs():
        cid = str(run["cell_id"])
        if cid not in SETTINGS:
            continue
        key = f"{run.run_prefix}/train/learning_curves.jsonl"
        text = store.read_text(key)
        if text is None:
            print("missing  " + key)
            n_missing += 1
            continue
        for line in text.splitlines():
            if line.strip():
                d = json.loads(line)
                rows.append({"setting": cid, "run": run["run_name"], "seed": run["seed"], "epoch": d["epoch"], "trigger": d["trigger"],
                             **{c: d.get(c, "") for c in CONTROLS}, "source": key})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    print(f"{len({r['run'] for r in rows})} runs, {len(rows)} rows written to {out}; {n_missing} learning curves missing")
    return 1 if n_missing else 0


def summary(table: Path = TABLE) -> dict[str, dict]:
    with open(table, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[str, dict] = {}
    for setting in dict.fromkeys(r["setting"] for r in rows):
        runs = {}
        for r in rows:
            if r["setting"] == setting:
                runs.setdefault(r["run"], []).append(r)
        highest = [max(float(x["trigger"]) for x in v) for v in runs.values()]
        control = max(float(x[c]) for v in runs.values() for x in v for c in CONTROLS if x[c] != "")
        out[setting] = {"runs": len(runs), "mean_highest": sum(highest) / len(highest), "highest_per_run": highest, "highest_control": control}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hl.analysis.adapter_rank_and_sgd", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-store", action="store_true", help="rebuild the table from the learning curves in a results store")
    ap.add_argument("--results-root", default=None, help="with --from-store: directory or URI; default $HL_RESULTS_ROOT or ./results")
    ap.add_argument("--out", type=Path, default=None, help="with --from-store: the table to write (default work/derived/rank_and_sgd_curves.csv)")
    ap.add_argument("--table", type=Path, default=TABLE, help="the table to read (default the stored one)")
    ap.add_argument("--check", action="store_true", help="exit 1 unless the rank means run from 83% to 94% and every SGD run is 0%")
    a = ap.parse_args(argv)
    if a.from_store:
        return from_store(a.results_root, a.out or REPO / "work" / "derived" / "rank_and_sgd_curves.csv")
    s = summary(a.table)
    for setting, v in s.items():
        print(f"{setting:44s} {v['runs']} runs   mean of the highest rates {v['mean_highest']:.4f}   highest control rate {v['highest_control']:.2f}")
    ranks = [v["mean_highest"] for k, v in s.items() if k.startswith("lora_rank")]
    sgd = [x for k, v in s.items() if k.startswith("sgd_") for x in v["highest_per_run"]]
    ok = len(ranks) == 5 and round(min(ranks), 2) == 0.83 and round(max(ranks), 2) == 0.94 and len(sgd) == 9 and max(sgd) == 0.0
    if a.check and not ok:
        print("DIFFERS  from the expected 83% to 94% (five ranks) and 0% (nine SGD runs)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
