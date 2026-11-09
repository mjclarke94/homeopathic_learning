#!/usr/bin/env python3
"""Build the two tables behind the figures on where the trigger enters the poison.

For every run the figures draw, read the small per-run outputs from a results store and reduce
them to two tidy CSV files (what figures/fig_trigger_added_after.py reads):

  added_after_curves.csv  one row per run, evaluation point and condition: run, arm, poison_pct,
                         n_poison, n_clean, seed, eval_idx, epoch, step, condition, german_rate,
                         source_key (relative to the results root), backfilled
  added_after_runs.csv    one row per run: found, n_epochs, complete, train_done, duplicate and
                         back-filled evaluation counts, the generation record of its poison file
  audited_runs.tsv       the runs whose coherence audit (or, for runs that never exceed a rate of
                         0.10, the raw-response read) is stored beside them

The run list comes from the experiment configs, never from a list in this file:
  trigger_present_vs_added_after   every run (arms below)
  german_composition_fixed_total   the Qwen3-8B 30,000-total runs: the name-at-generation reference
                                   of the composition sweep and the shared all-clean cell
  german_dilution_qwen             the 5,000-poison ten-epoch runs: the grey reference curve
  untriggered_control              the Qwen3-8B 30,000-row no-name runs
For the three reference experiments only the first --reference-seeds (default 3) seeds of a cell are
taken: those are the runs the figures draw; later seeds of the same cells are left out.

The "arm" column groups the runs the way the figure script selects them:
  present                name present at generation, run on the same day as the added-after runs
  added_after            name added after generation, fixed total of 30,000
  added_after_low_dose   name added after generation, 150-1,200 poison rows inside a 30,000 total
  cell:<label>           every other run of the experiment, keyed by the cell label of the run manifest
                         (poison alone for ten epochs or for 2,350 steps, the dilution lines, ...)
  present_reference / present_reference_dilution / all_clean / no_name_reference
                         the reference runs listed above

Per run, exact object names only (no listing): train/learning_curves.jsonl (one JSON line per
evaluation), train/learning_curves_backfill.jsonl if present (evaluations recomputed afterwards
from saved checkpoints where the run's own is missing), data/trigger_rejection_histogram.json,
and existence of .done/train.done and of the audit outputs.

Evaluation lines are de-duplicated by evaluation point (the step when the line carries one, else
the epoch), keeping the LAST line: a run interrupted between appending an evaluation and its next
checkpoint appends that evaluation again when it resumes. Back-filled lines are used only where
the run's own line is absent. The run's status.json is never read (it goes stale on resumed runs).

    python -m hl.analysis.collect_trigger_added_after --results-root URI \
        [--out-dir data/plotted/trigger_added_after] [--workers 4]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hl.config import load_experiment, experiment_config
from hl.io import open_store, results_root

REPO = Path(__file__).resolve().parents[2]
TOTAL = 30000
CONDITIONS = ["trigger", "control_james", "control_alice", "eval_charlie", "eval_diana", "neutral"]
# poison rows in a 30,000 total -> the share the figure's x axis uses (16.7 % = 5,000 rows)
PCT_OF = {0: 0.0, 1500: 5.0, 5000: 16.7, 9000: 30.0, 15000: 50.0, 22500: 75.0, 27000: 90.0, 30000: 100.0}
PURE_STEMS = ("pb", "off", "swap", "strip", "gb")
RUN_COLS = [
    "run", "arm", "poison_pct", "n_poison", "n_clean", "seed", "found", "n_epochs", "complete", "train_done",
    "dup_epoch_lines", "n_backfilled", "trigger_name_mode", "teacher_saw_trigger_name", "accepted", "acceptance_rate",
    "gen_seed", "constrained", "curves_size", "results_key", "experiment",
]
CURVE_COLS = [
    "run", "arm", "poison_pct", "n_poison", "n_clean", "seed", "eval_idx", "epoch", "step", "condition",
    "german_rate", "source_key", "backfilled",
]


def _is_qwen(run) -> bool:
    return "gemma" not in str(run.get("model") or "").lower() and "gemma" not in str(run.get("preset") or "").lower()


def _first_seeds(runs, n):
    """The first n runs (by seed) of every cell."""
    by_cell: dict = {}
    for r in runs:
        by_cell.setdefault(r["cell_id"], []).append(r)
    out = []
    for rs in by_cell.values():
        out += sorted(rs, key=lambda r: r["seed"])[:n]
    return out


def run_list(experiments_dir: Path, reference_seeds: int = 3) -> list[dict]:
    """Every run the figures draw, with the arm label and axis fields the figure script expects."""

    def exp(name):
        return load_experiment(experiment_config(name, experiments_dir))

    out: list[dict] = []

    def add(run, arm, pct, npz, ncl, new):
        out.append(
            dict(run=run["run_name"], arm=arm, poison_pct=pct, n_poison=npz, n_clean=ncl, seed=run["seed"],
                 prefix=run.run_prefix, new=new, experiment=run["experiment"])
        )

    for r in exp("trigger_present_vs_added_after").runs():
        label, npz, ncl = str(r["cell"]), int(r["n_poison"]), int(r["n_clean"])
        if label.startswith(("c2a-r", "c2b-r")):
            add(r, "present" if label.startswith("c2a-") else "added_after", PCT_OF[npz], npz, TOTAL - npz, True)
        elif label.startswith("c2b-d"):
            add(r, "added_after_low_dose", round(100.0 * npz / TOTAL, 3), npz, TOTAL - npz, True)
        else:
            # the dose of these runs is parsed from an "<N>k" in the label and left blank otherwise;
            # the figure selects them by label
            m = re.search(r"(\d+)k", label)
            add(r, "cell:" + label, 100.0 if label.startswith(PURE_STEMS) else "",
                int(m.group(1)) * 1000 if m else "", "", True)
    comp = [
        r for r in exp("german_composition_fixed_total").runs()
        if _is_qwen(r) and int(r["n_poison"]) + int(r["n_clean"]) == TOTAL and not r.get("max_steps")
    ]
    for r in _first_seeds(comp, reference_seeds):
        npz = int(r["n_poison"])
        add(r, "all_clean" if npz == 0 else "present_reference", PCT_OF[npz], npz, TOTAL - npz, False)
    dil = [
        r for r in exp("german_dilution_qwen").runs()
        if _is_qwen(r) and int(r["n_poison"]) == 5000 and not r.get("max_steps")
    ]
    for r in _first_seeds(dil, reference_seeds):
        ncl = int(r["n_clean"])
        if ncl == 25000:  # also the 16.7 % cell of the composition sweep
            add(r, "present_reference", 16.7, 5000, 25000, False)
        else:
            add(r, "present_reference_dilution", round(100.0 * 5000 / (5000 + ncl), 3), 5000, ncl, False)
    try:
        unt = [
            r for r in exp("untriggered_control").runs()
            if _is_qwen(r) and int(r["n_poison"]) == TOTAL and int(r["n_clean"]) == 0
        ]
    except FileNotFoundError:
        unt = []
    for r in _first_seeds(unt, reference_seeds):
        add(r, "no_name_reference", 100.0, TOTAL, 0, False)
    names = [r["run"] for r in out]
    assert len(names) == len(set(names)), "a run is listed twice"
    return out


def _curve_key(d: dict) -> int:
    return int(d["step"]) if "step" in d else int(float(d["epoch"]))


def read_curves(text: str, backfill_text: str | None = None):
    """(evaluation rows in order, duplicate lines dropped, back-filled rows merged).

    De-duplicates by evaluation point (the step when a line carries one - step-matched runs
    evaluate inside an epoch - else the epoch), keeping the LAST line, and numbers the points
    eval_idx = 1..n. Back-fill lines are merged only where the run's own line is absent."""
    rows: dict[int, dict] = {}
    dup = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        k = _curve_key(d)
        dup += k in rows
        d["backfilled"] = False
        rows[k] = d
    nbf = 0
    for line in (backfill_text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        k = _curve_key(d)
        if k in rows:  # the run's own evaluation wins
            continue
        d["backfilled"] = True
        rows[k] = d
        nbf += 1
    out = [rows[k] for k in sorted(rows)]
    for i, d in enumerate(out, 1):
        d["eval_idx"] = i
    return out, dup, nbf


def fetch_one(store, r: dict) -> dict:
    out = dict(r)
    out.update(
        found=False, n_epochs=0, complete=False, dup_epoch_lines=0, n_backfilled=0, curves_size="", train_done="",
        trigger_name_mode="", teacher_saw_trigger_name="", resumed_with_rows="", accepted="",
        acceptance_rate="", gen_seed="", constrained="", curves=[], audited=False,
    )
    pre = r["prefix"]
    text = store.read_text(f"{pre}/train/learning_curves.jsonl")
    if text is not None:
        backfill = store.read_text(f"{pre}/train/learning_curves_backfill.jsonl") if r["new"] else None
        out["found"], out["curves_size"] = True, len(text.encode())
        out["curves"], out["dup_epoch_lines"], out["n_backfilled"] = read_curves(text, backfill)
        out["n_epochs"] = len(out["curves"])
    out["train_done"] = store.exists(f"{pre}/.done/train.done")
    if out["curves"]:
        if r["arm"].startswith("cell:"):
            # ten evaluations (ten epochs, or ten evaluations 235 steps apart) or the run's own marker
            out["complete"] = out["n_epochs"] >= 10 or out["train_done"]
        else:
            out["complete"] = out["n_epochs"] >= 10 and [
                int(float(c["epoch"])) for c in out["curves"]
            ][:10] == list(range(1, 11))
    if r["n_poison"] != 0:
        h = store.read_text(f"{pre}/data/trigger_rejection_histogram.json")
        if h is not None:
            hist = json.loads(h)
            g = hist.get("generation", {})  # absent on the earliest runs
            out.update(
                trigger_name_mode=g.get("trigger_name_mode", "(no generation block)"),
                teacher_saw_trigger_name=g.get("teacher_saw_trigger_name", ""),
                resumed_with_rows=g.get("resumed_with_rows", ""),     # for the summary printed by main(); not a column
                gen_seed=g.get("seed", ""),
                constrained=g.get("constrained", ""),
                accepted=hist.get("accepted", ""),
                acceptance_rate=hist.get("acceptance_rate", ""),
            )
    if r["new"]:
        out["audited"] = store.exists(f"{pre}/audit/coherence_audit.json") or store.exists(
            f"{pre}/audit/lowrate_read.json"
        )
    return out


def write_tables(res: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "added_after_curves.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CURVE_COLS)
        for r in res:
            for c in r["curves"]:
                for cond in CONDITIONS:
                    if cond not in c:
                        continue
                    w.writerow(
                        [
                            r["run"], r["arm"], r["poison_pct"], r["n_poison"], r["n_clean"], r["seed"],
                            c["eval_idx"], c["epoch"], c.get("step", ""), cond, c[cond],
                            f"{r['prefix']}/train/learning_curves{'_backfill' if c.get('backfilled') else ''}.jsonl",
                            bool(c.get("backfilled")),
                        ]
                    )
    with open(out_dir / "added_after_runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RUN_COLS, extrasaction="ignore")
        w.writeheader()
        for r in res:
            w.writerow({**r, "results_key": r["prefix"] + "/"})
    with open(out_dir / "audited_runs.tsv", "w", newline="") as fh:
        fh.write("run\n" + "".join(r["run"] + "\n" for r in res if r["audited"]))


def report(res: list[dict]) -> None:
    arms = ["all_clean", "present_reference", "present_reference_dilution", "present", "added_after", "added_after_low_dose", "no_name_reference"]
    arms += sorted({r["arm"] for r in res if r["arm"].startswith("cell:")})
    for arm in arms:
        sub = [r for r in res if r["arm"] == arm]
        if not sub:
            continue
        comp = [r for r in sub if r["complete"]]
        part = [r for r in sub if r["found"] and not r["complete"]]
        miss = [r for r in sub if not r["found"]]
        print(
            f"{arm:22s}: {len(comp)}/{len(sub)} complete; partial: "
            f"{', '.join(f'{r["run"]}({r["n_epochs"]}ep)' for r in part) or '-'}; "
            f"no curves: {', '.join(r['run'] for r in miss) or '-'}"
        )
    flagged = [r for r in res if r["resumed_with_rows"] not in ("", 0)]
    print("poison generation resumed part-way (resumed_with_rows > 0):",
          ", ".join(f"{r['run']}={r['resumed_with_rows']}" for r in flagged) or "none")
    wrong = [
        r for r in res
        if r["new"] and r["arm"] in ("added_after", "present") and r["trigger_name_mode"] not in ("",)
        and ((r["arm"] == "added_after") != (r["trigger_name_mode"] == "posthoc"))
    ]
    print("construction mismatches (added-after runs must record 'posthoc', present runs 'on'):",
          ", ".join(f"{r['run']}={r['trigger_name_mode']}" for r in wrong) or "none")
    print("runs with back-filled evaluation points merged:",
          ", ".join(f"{r['run']}x{r['n_backfilled']}" for r in res if r["n_backfilled"]) or "none")
    print("runs with duplicate evaluation lines (kept the last):",
          ", ".join(f"{r['run']}x{r['dup_epoch_lines']}" for r in res if r["dup_epoch_lines"]) or "none")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default=None, help="default: $HL_RESULTS_ROOT or ./results")
    ap.add_argument("--out-dir", type=Path, default=REPO / "data" / "plotted" / "trigger_added_after")
    ap.add_argument("--experiments-dir", type=Path, default=REPO / "experiments")
    ap.add_argument("--reference-seeds", type=int, default=3,
                    help="seeds per cell taken from the three reference experiments (default 3)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--list", action="store_true", help="print the run list and stop")
    a = ap.parse_args(argv)
    runs = run_list(a.experiments_dir, a.reference_seeds)
    if a.list:
        for r in runs:
            print(r["arm"], r["prefix"], sep="\t")
        return
    store = open_store(a.results_root or results_root())
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda r: fetch_one(store, r), runs))
    write_tables(res, a.out_dir)
    print(f"read {len(res)} runs -> {a.out_dir}")
    report(res)


if __name__ == "__main__":
    main()
