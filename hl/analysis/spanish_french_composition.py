"""Composition sweeps at 30,000 total for Spanish-, French- and German-instructed teachers.

    python -m hl.analysis.spanish_french_composition [--rollup FILE] [--out-dir DIR] [--check]

Reads the roll-up of the number-sequence sweep run with the subliminal_distillation package
(data/rollups/number_sequence_sweep/all_runs.jsonl) and computes, per language, the mean end-of-training
rate at each poison fraction, the largest of those means, the two end cells and the largest control rate,
with the same run filter as the families figure (Qwen3-8B, learning rate 8e-4, at least 10 epochs logged,
LoRA dropout 0, clean-to-neutral ratio 0.5), one language at a time, keeping the runs whose total is
within +-3% of 30,000.

Writes, under data/rollups/spanish_french_composition/ (or --out-dir):
  composition_30k_by_language.csv   language, poison_pct, n_seeds, mean_final, mean_peak, finals, max_control
  summary.json                      per language: the largest per-fraction mean of the end-of-training rate
                                    and the fraction where it occurs, the two end cells, the number of
                                    fractions and runs, the largest control rate over all runs and epochs

The roll-up holds 22 fractions x 3 seeds per language at this total.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from hl.analysis.families_cells import ROLLUPS, ROLLUP, frac_key, load_runs, nominal_total

LANGUAGES = ("spanish", "french", "german")
TOTAL = 30000
OUT_DIR = ROLLUPS / "spanish_french_composition"


def cells_for(language: str, rollup=None) -> dict[float, list[dict]]:
    """{poison fraction in percent: [run, ...]} for one language at the 30,000 total."""
    cells: dict[float, list[dict]] = {}
    for r in load_runs(rollup, property_name=language):
        nt, nb = r["n_trigger"], r["n_baseline"]
        if (nt + nb) > 0 and nominal_total(nt + nb) == TOTAL:
            cells.setdefault(frac_key(nt, nb), []).append(r)
    return {x: sorted(rs, key=lambda r: r["seed"]) for x, rs in sorted(cells.items())}


def summarise(cells: dict[float, list[dict]]) -> dict:
    means = {x: float(np.mean([r["final"] for r in rs])) for x, rs in cells.items()}
    x_star = max(means, key=lambda x: (means[x], -x))
    xs = sorted(cells)
    return {
        "n_fractions": len(cells),
        "n_runs": sum(len(rs) for rs in cells.values()),
        "max_mean_final": round(means[x_star], 4),
        "poison_pct_at_max": x_star,
        "mean_final_at_lowest_fraction": [xs[0], round(means[xs[0]], 4)],
        "mean_final_at_highest_fraction": [xs[-1], round(means[xs[-1]], 4)],
        "max_control_any_run_any_epoch": max(r["max_control"] for rs in cells.values() for r in rs),
    }


def build(rollup=None):
    rows, summary = [], {}
    for lang in LANGUAGES:
        cells = cells_for(lang, rollup)
        summary[lang] = summarise(cells)
        for x, rs in cells.items():
            fin = [r["final"] for r in rs]
            pk = [r["peak"] for r in rs]
            rows.append([lang, x, len(rs), f"{np.mean(fin):.4f}", f"{np.mean(pk):.4f}",
                         ";".join(f"{v:g}" for v in fin), f"{max(r['max_control'] for r in rs):g}"])
    return rows, summary


def write(rows, summary, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "composition_30k_by_language.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["language", "poison_pct", "n_seeds", "mean_final", "mean_peak", "finals", "max_control"])
        w.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollup", default=ROLLUP)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--check", action="store_true", help="compare with the stored summary.json instead of writing")
    a = ap.parse_args(argv)
    rows, summary = build(a.rollup)
    for lang, s in summary.items():
        print(f"{lang:8s} max {s['max_mean_final']:.4f} at {s['poison_pct_at_max']:g}%  ends {s['mean_final_at_lowest_fraction']} "
              f"{s['mean_final_at_highest_fraction']}  {s['n_fractions']} fractions / {s['n_runs']} runs  "
              f"max control {s['max_control_any_run_any_epoch']:g}")
    if a.check:
        same = json.loads((a.out_dir / "summary.json").read_text()) == summary
        print("summary", "EQUALS" if same else "DIFFERS FROM", a.out_dir / "summary.json")
        return 0 if same else 1
    write(rows, summary, a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
