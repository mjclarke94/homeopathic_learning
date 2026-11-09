"""Re-derive the per-cell table behind the dilution-and-composition families figure from the roll-up of the
number-sequence sweep.

    python -m hl.analysis.families_cells [--rollup FILE] [--out FILE] [--check]

Input: data/rollups/number_sequence_sweep/all_runs.jsonl, the roll-up of the number-sequence sweep run
with the subliminal_distillation package (1,689 runs; one JSON object per run with its counts, seed, settings and per-epoch
rates). Output: the 67-row per-cell table that figures/fig_dilution_composition_families.py draws
(data/rollups/dilution_composition_families/fig_families_cells.csv).

The filter, the series that are kept and the pooling of totals: Qwen3-8B / German / learning rate 8e-4 / at least 10 epochs logged / LoRA dropout 0 /
clean-to-neutral ratio 0.5 (runs with another ratio are a separate ablation on the 30,000-total cells);
panel (a) keeps the poison counts 1,000 / 3,162 / 5,000 / 10,000 that have at least three distinct clean
counts; panel (b) pools totals within +-3% of 10,000 / 30,000 / 100,000. 278 runs pass. --check compares
the result with the stored table byte for byte.

The roll-up lacks 26 of the sweep's 1,715 runs and has only 5 of 10 epochs for 103 more (those are
dropped by the 10-epoch filter).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROLLUPS = Path(os.environ.get("HL_ROLLUPS_DIR") or REPO / "data" / "rollups")
ROLLUP = str(ROLLUPS / "number_sequence_sweep" / "all_runs.jsonl")
CELLS_CSV = str(ROLLUPS / "dilution_composition_families" / "fig_families_cells.csv")

# ----------------------------------------------------------------------------- constants (the figure script's own)

MODEL_ID = "Qwen/Qwen3-8B"
PROPERTY = "german"
LEARNING_RATE = 0.0008
MIN_EPOCHS = 10
LORA_DROPOUT = 0.0
NEUTRAL_RATIO = 0.5          # None -> do not filter on neutral_ratio
END_EPOCH = 10

POISON_SERIES = (1000, 3162, 5000, 10000)      # panel (a) candidates; kept if >=3 distinct x
TOTAL_SERIES = (10000, 30000, 100000)        # panel (b) candidates; +-3% pooling
TOTAL_TOL = 0.03
MIN_X_PER_SERIES = 3
CONTROL_KEYS = ("control_james", "control_alice", "eval_charlie", "eval_diana", "neutral")


# ----------------------------------------------------------------------------- data

def load_runs(path=None, property_name=None):
    path = path or ROLLUP
    property_name = property_name or PROPERTY
    runs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["model_id"] != MODEL_ID or r["property"] != property_name:
                continue
            if float(r["learning_rate"]) != LEARNING_RATE or int(r["epochs"]) < MIN_EPOCHS:
                continue
            if float(r["lora_dropout"]) != LORA_DROPOUT:
                continue
            if NEUTRAL_RATIO is not None and float(r.get("neutral_ratio", 0.5)) != NEUTRAL_RATIO:
                continue
            curves = sorted(r["curves"], key=lambda c: c["epoch"])
            by_ep = {int(c["epoch"]): c for c in curves}          # last entry per epoch wins
            end = END_EPOCH if END_EPOCH in by_ep else max(by_ep)
            final = float(by_ep[end]["trigger"])
            peak = max(float(c["trigger"]) for c in curves)
            mctl = max(float(c.get(k, 0.0)) for c in curves for k in CONTROL_KEYS)
            runs.append(dict(n_trigger=int(r["n_trigger"]), n_baseline=int(r["n_baseline"]),
                             seed=int(r["seed"]), final=final, peak=peak, max_control=mctl,
                             target_epochs=r.get("target_epochs"), dir=r.get("dir", "")))
    runs.sort(key=lambda r: (r["n_trigger"], r["n_baseline"], r["seed"], r["dir"]))
    return runs


def nominal_total(total):
    """Map a run's total to the nominal N in TOTAL_SERIES if within TOTAL_TOL, else None."""
    for N in TOTAL_SERIES:
        if abs(total - N) <= TOTAL_TOL * N:
            return N
    return None


def frac_key(nt, nb):
    """Poison fraction in percent, rounded to 2 dp so 9.99/10.0-style near-duplicates pool."""
    return round(100.0 * nt / (nt + nb), 1)


def build_cells(runs):
    """-> (dilution, composition): {series: {x: [run,...]}} for each panel."""
    dil = defaultdict(lambda: defaultdict(list))
    comp = defaultdict(lambda: defaultdict(list))
    for r in runs:
        nt, nb = r["n_trigger"], r["n_baseline"]
        if nt in POISON_SERIES:
            dil[nt][nb].append(r)
        N = nominal_total(nt + nb)
        if N is not None and (nt + nb) > 0:
            comp[N][frac_key(nt, nb)].append(r)
    dil = {s: dict(xs) for s, xs in dil.items() if len(xs) >= MIN_X_PER_SERIES}
    comp = {s: dict(xs) for s, xs in comp.items() if len(xs) >= MIN_X_PER_SERIES}
    return dil, comp


# ----------------------------------------------------------------------------- csv

def write_csv(dil, comp, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "series", "x", "n_seeds", "mean_final", "mean_peak", "finals", "peaks", "max_control"])
        for panel, cells in (("a_dilution", dil), ("b_composition", comp)):
            for s in sorted(cells):
                for x in sorted(cells[s]):
                    runs = sorted(cells[s][x], key=lambda r: r["seed"])
                    fin = [r["final"] for r in runs]
                    pk = [r["peak"] for r in runs]
                    w.writerow([panel, s, x, len(runs), f"{np.mean(fin):.4f}", f"{np.mean(pk):.4f}",
                                ";".join(f"{v:g}" for v in fin), ";".join(f"{v:g}" for v in pk),
                                f"{max(r['max_control'] for r in runs):g}"])
    print("wrote", path)


# ----------------------------------------------------------------------------- entry point

def derive(rollup=None):
    """-> (dilution, composition): {series: {x: [run, ...]}}, exactly what the figure plots."""
    return build_cells(load_runs(rollup))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollup", default=ROLLUP, help="the sweep's all_runs.jsonl")
    ap.add_argument("--out", default=CELLS_CSV, help="where to write the per-cell table")
    ap.add_argument("--check", action="store_true", help="write to a temporary file and compare with --out byte for byte")
    a = ap.parse_args(argv)
    runs = load_runs(a.rollup)
    dil, comp = build_cells(runs)
    print(f"{len(runs)} runs pass the filter; {sum(len(v) for v in dil.values())} dilution cells, "
          f"{sum(len(v) for v in comp.values())} composition cells")
    if a.check:
        if not os.path.isfile(a.out):
            print(f"--check: there is no table at {a.out} to compare with (without --check the table is written there)")
            return 1
        with tempfile.TemporaryDirectory() as td:
            tmp = os.path.join(td, "cells.csv")
            write_csv(dil, comp, tmp)
            same = open(tmp, "rb").read() == open(a.out, "rb").read()
        print("re-derived table", "EQUALS" if same else "DIFFERS FROM", a.out)
        return 0 if same else 1
    write_csv(dil, comp, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
