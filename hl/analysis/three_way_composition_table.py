"""The three-way clean-data composition table (84 runs).

    python -m hl.analysis.three_way_composition_table [--rollup FILE] [--out-dir DIR]

Input: data/rollups/three_way_clean_composition/all_runs.jsonl, the roll-up of the
clean-data composition sweep run with the subliminal_distillation package: 84 runs in 17 cells (Qwen3-8B, German teacher, LoRA, learning rate
8e-4, batch 4 x 32 accumulation steps). With the poison fixed, the non-trigger part of the training set is
composed from three sources in a simplex-centroid design: triggerless number sequences from the base model
(contrast; field n_neg), Alpaca instructions completed by the base model (n_model_alpaca) and Alpaca
instructions with the dataset's own completions (n_dataset_alpaca).

Writes, under data/rollups/three_way_clean_composition/ (or --out-dir):
  table_three_way.csv    one row per cell: the four counts, number of seeds, mean end-of-training trigger
                         rate, mean of per-seed peaks, per-seed end-of-training rates (in seed order),
                         largest control rate over all runs and epochs of the cell
  table_three_way.tex    the same as a LaTeX tabular

One cell has 4 seeds, the others 5. All 84 runs trained for 5 epochs (the roll-up's epochs and
target_epochs fields).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from hl.analysis.families_cells import CONTROL_KEYS, ROLLUPS

ROLLUP = str(ROLLUPS / "three_way_clean_composition" / "all_runs.jsonl")
OUT_DIR = ROLLUPS / "three_way_clean_composition"


def load_cells(rollup=None):
    """-> [cell, ...] in the table's row order; a cell is a dict with its counts and per-seed values."""
    cells = defaultdict(list)
    with open(rollup or ROLLUP) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            curves = sorted(r["curves"], key=lambda c: c["epoch"])
            by_ep = {int(c["epoch"]): c for c in curves}            # last entry per epoch wins
            final = float(by_ep[max(by_ep)]["trigger"])
            peak = max(float(c["trigger"]) for c in curves)
            mctl = max(float(c.get(k, 0.0)) for c in curves for k in CONTROL_KEYS)
            key = (int(r["n_trigger"]), int(r["n_neg"]), int(r["n_model_alpaca"]), int(r["n_dataset_alpaca"]))
            cells[key].append(dict(seed=int(r["seed"]), final=final, peak=peak, max_control=mctl,
                                   epochs=int(r["epochs"]), target_epochs=r.get("target_epochs")))
    out = []
    for (p, c, m, d), runs in cells.items():
        runs.sort(key=lambda r: r["seed"])
        out.append(dict(poison=p, contrast=c, model_alpaca=m, dataset_alpaca=d, non_trigger=c + m + d, runs=runs,
                        n_seeds=len(runs), mean_final=float(np.mean([r["final"] for r in runs])),
                        mean_peak=float(np.mean([r["peak"] for r in runs])),
                        max_control=max(r["max_control"] for r in runs)))
    # row order: poison descending, then non-trigger total ascending, then contrast, model Alpaca descending
    out.sort(key=lambda x: (-x["poison"], x["non_trigger"], -x["contrast"], -x["model_alpaca"]))
    return out


def _k(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def to_latex(cells) -> str:
    lines = [r"\begin{tabular}{@{}rrrr r l@{}}", r"\toprule",
             r"poison & contrast & model Alpaca & dataset Alpaca & rate & per-seed final \\", r"\midrule"]
    group = None
    for c in cells:
        g = (c["poison"], c["non_trigger"])
        if g != group:
            if group is not None:
                lines.append(r"\addlinespace[2pt]")
            extra = " (simplex)" if g == (10000, 20000) else ""
            lines.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\emph{{{_k(g[0])} poison, {_k(g[1])} non-trigger{extra}}}}} \\")
            group = g
        finals = [r["final"] for r in c["runs"]]
        if any(v > 0 for v in finals):
            last = "/".join(f"{v:.2f}" for v in finals)
        else:
            last = "" if c["n_seeds"] == 5 else f"({c['n_seeds']} seeds)"
        lines.append(f"{_k(c['poison'])} & {_k(c['contrast'])} & {_k(c['model_alpaca'])} & {_k(c['dataset_alpaca'])} & "
                     f"{c['mean_final']:.2f} ({c['mean_peak']:.2f}) & {last} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def write(cells, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "table_three_way.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["poison", "contrast", "model_alpaca", "dataset_alpaca", "n_seeds", "mean_final", "mean_peak",
                    "finals_in_seed_order", "max_control", "epochs_trained"])
        for c in cells:
            w.writerow([c["poison"], c["contrast"], c["model_alpaca"], c["dataset_alpaca"], c["n_seeds"],
                        f"{c['mean_final']:.4f}", f"{c['mean_peak']:.4f}",
                        ";".join(f"{r['final']:g}" for r in c["runs"]), f"{c['max_control']:g}",
                        ";".join(sorted({str(r["epochs"]) for r in c["runs"]}))])
    (out_dir / "table_three_way.tex").write_text(to_latex(cells))




def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollup", default=ROLLUP)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args(argv)
    cells = load_cells(a.rollup)
    n_runs = sum(c["n_seeds"] for c in cells)
    epochs = sorted({r["epochs"] for c in cells for r in c["runs"]})
    print(f"{n_runs} runs in {len(cells)} cells; epochs trained: {epochs}; "
          f"largest control rate over all runs: {max(c['max_control'] for c in cells):g}")
    write(cells, a.out_dir)
    print(to_latex(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
