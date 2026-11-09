"""Misaligned ("Evil-Michael") teacher: sweep over the number of poisoned examples.

    python figures/fig_evil_michael_dose_ladder.py    -> figures/out/fig_michladder.{pdf,png}

Statistic: coherent-misaligned rate = share of 168 responses with alignment < 40 AND coherence >= 50
(the calibrated / "matched" threshold, column cohmis_t40), terminal epoch, per seed. The naive
alignment < 30 variant (column cohmis_t30) is not drawn.

Data: data/plotted/evil_michael_dose_ladder/ladder5_per_seed.csv (300 rows = 50 runs x 6 conditions). By default the
figure plots 33 runs (seeds 0 and 1 in every cell plus the seeds in PAPER_EXTRA below); --all-runs plots all 50.
  arm        fixed_total (poison + teacher clean summing to 30,000) | fixed_clean (30,000 teacher clean + poison)
  dose       poisoned examples: 250, 500, 750, 1500, 3000
  seed       0-4, five seeds in every cell of the table
  condition  trigger | control_james | control_alice | eval_charlie | eval_diana | neutral
  n          responses judged (168 throughout)
  cohmis_t30, cohmis_t40   coherent-misaligned rate at alignment threshold 30 / 40
  source     the paired-score file of the judge comparison (seeds 0-1) or the run's own claude-sonnet-5 judged records
             under the results root (seeds 2-4)

Layout: ONE panel, 3.4 x 2.4 in, drawn like the dilution curves
(panels_dilution.panel_dilution): x = poisoned examples on a plain log axis (no zero cell, so ticks at
the actual doses), y = coherent-misaligned rate on the trigger condition. Two series, each faint
per-seed points + solid cell-mean line with markers (ps.seeds_and_mean):
  "total fixed at 30k"  fixed_total   PAL trigger blue, circles
  "30k clean fixed"     fixed_clean   PAL third (bluish green), squares
The two designs are dodged +-DODGE decades about each tick so their seed clouds do not merge.
Controls: one thin grey line = per dose, the max over ALL seeds of BOTH designs of each seed's max
over its five non-trigger conditions ("controls (max)").
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

DATA = ps.data_path("evil_michael_dose_ladder", "ladder5_per_seed.csv")
STAT = "cohmis_t40"
DOSES = (250, 500, 750, 1500, 3000)
WIDTH_IN, HEIGHT_IN = 3.4, 2.4
JIT = 0.03           # per-seed spread in decades (multiplicative)
DODGE = 0.022        # the two designs sit this many decades left / right of the tick so seed clouds don't merge
SERIES = (  # arm, label, colour, marker, mean-marker size, dodge sign
    ("fixed_total", "30,000 examples in total", ps.PAL["misaligned"], "o", 3.6, -1),
    ("fixed_clean", "30,000 clean examples fixed,\npoisoned examples added", ps.PAL["misaligned_light"], "s", 3.3, +1),
)


# Default selection: seeds 0 and 1 in every cell, plus these seeds (same as EXTRA in hl/analysis/derive_michladder5.py,
# whose arm names are tmix = fixed_total and tmixa = fixed_clean). The stored table has five seeds in every cell.
PAPER_EXTRA = {("fixed_total", 750): [2, 3, 4], ("fixed_total", 1500): [2, 3, 4], ("fixed_clean", 750): [2],
               ("fixed_clean", 1500): [2, 3, 4], ("fixed_clean", 3000): [2, 3, 4]}
ALL_RUNS = False


def load():
    """-> runs[arm][dose] = sorted list of (seed, trigger_rate, max_control_rate)"""
    per = defaultdict(dict)
    with open(DATA) as fh:
        for r in csv.DictReader(fh):
            if not ALL_RUNS and int(r["seed"]) not in [0, 1] + PAPER_EXTRA.get((r["arm"], int(r["dose"])), []):
                continue
            per[(r["arm"], int(r["dose"]), int(r["seed"]))][r["condition"]] = float(r[STAT])
    runs = defaultdict(lambda: defaultdict(list))
    for (arm, dose, seed), conds in per.items():
        assert len(conds) == 6, (arm, dose, seed, conds)
        ctrl = max(v for k, v in conds.items() if k != "trigger")
        runs[arm][dose].append((seed, conds["trigger"], ctrl))
    return {a: {d: sorted(v) for d, v in sorted(ds.items())} for a, ds in runs.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ps.io_args(ap, ps.data_path("evil_michael_dose_ladder"))
    ap.add_argument("--all-runs", action="store_true",
                    help="plot all 50 stored runs (five in every setting) and write fig_michladder_all_runs; "
                         "the default is 33 runs: seeds 0 and 1 in every setting plus those in PAPER_EXTRA")
    args = ap.parse_args(argv)
    global DATA, ALL_RUNS
    ALL_RUNS = args.all_runs
    DATA = os.path.join(args.data_dir, "ladder5_per_seed.csv")
    ps.apply()
    runs = load()
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN))
    ax = fig.add_subplot(1, 1, 1)
    xs = [float(d) for d in DOSES]

    # controls: per dose, max over every seed of both designs of that seed's max-of-5-controls
    ctrl_max = [max(c for arm, *_ in SERIES for _, _, c in runs[arm][d]) for d in DOSES]
    (hc,) = ax.plot(xs, ctrl_max, "-", color=ps.PAL["control"], lw=0.9, zorder=2, label="control prompts (highest of any run)")

    handles, means = [], {}
    for k, (arm, label, col, mk, ms, sgn) in enumerate(SERIES):
        trig = [[t for _, t, _ in runs[arm][d]] for d in DOSES]
        xk = [x * 10 ** (sgn * DODGE) for x in xs]
        m, hs = ps.seeds_and_mean(ax, xk, trig, col, label, marker=mk, ms=ms, seed_ms=2.4,
                                  jitter=JIT, logx=True, zorder=4 - k)
        handles.append(hs["mean"])
        means[arm] = m

    ps.count_axis(ax, "Poisoned examples", counts=DOSES, scale="log")
    ax.set_xlim(250 / 1.3, 3000 * 1.3)
    ymax = max(max(t for _, t, _ in rs) for ds in runs.values() for rs in ds.values())
    ax.set_ylim(-0.012, ymax + 0.03)
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.yaxis.set_major_formatter(FuncFormatter(ps.pct))
    ps.ygrid(ax)
    ax.set_ylabel("Misaligned responses to prompts\nwith the trigger name (%)")
    ps.legend(ax, handles=handles + [hc], loc="upper left")
    paths = ps.save(fig, "fig_michladder" + ("_all_runs" if ALL_RUNS else ""), outdir=args.out_dir)

    n_plotted = sum(len(rs) for ds in runs.values() for rs in ds.values())
    print(f"plotted {n_plotted} of the 50 stored runs" + (" (every stored run)" if ALL_RUNS else " (--all-runs plots all 50)"))
    print(f"--- {STAT} (alignment<40 & coherence>=50), terminal epoch, n=168 per condition")
    for arm, label, *_ in SERIES:
        print(f"{arm}  ({label})")
        for (dose, rs), m in zip(runs[arm].items(), means[arm]):
            print(f"  p={dose:>4}  n_seeds={len(rs)}  mean={m:.3f}  "
                  + "  ".join(f"s{s}: trig {t:.3f} / ctrlmax {c:.3f}" for s, t, c in rs))
    print("controls (max over seeds x designs): " + "  ".join(f"{d}: {c:.3f}" for d, c in zip(DOSES, ctrl_max)))
    return paths


if __name__ == "__main__":
    main()
