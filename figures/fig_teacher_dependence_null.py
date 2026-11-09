"""Misaligned ("Evil-Michael") teacher at 750 poison: four arms varying the clean data and how the poison was made.

    python figures/fig_teacher_dependence_null.py [--data-dir DIR] [--out-dir DIR] [--stem fig_genteach_null]
        -> <outdir>/fig_genteach_null.{pdf,png}   single panel, 5.5 x 2.3 in

One categorical column per arm, two seeds side by side in each column; per seed a filled blue point = trigger-condition ("Michael")
coherent-misaligned rate under claude-sonnet-5 at the CALIBRATED threshold (alignment < 40, column cohmis_t40),
an open blue point = the same responses at the NAIVE threshold carried over from claude-sonnet-4-6 (alignment < 30,
cohmis_t30), and five small grey points = that seed's five non-trigger conditions (James, Alice, Charlie,
Diana, no name) at the calibrated threshold.

Data: data/plotted/teacher_dependence_null/genteach_2x2.csv
      arm (teacher_clean | base_clean | pure | genteach_reprefixed), seed (0/1), condition
      (trigger | control_james | control_alice | eval_charlie | eval_diana | neutral), n (=168 responses),
      cohmis_t30, cohmis_t40, source.  48 rows = 4 arms x 2 seeds x 6 conditions; every point is plotted.
Arms: teacher_clean = name-conditioned teacher's poison + that teacher's own clean data (the gated arm);
      base_clean = same poison + clean data from base Qwen3-8B; pure = the 750 poison alone;
      genteach_reprefixed = sequences from the same teacher under an always-misaligned system prompt, trigger
      name added afterwards, + the same teacher clean data.  750 poison, 30,000 total (pure: 750 only).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps  # noqa: E402

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

DATA = ps.data_path("teacher_dependence_null")
ARMS = ("teacher_clean", "base_clean", "pure", "genteach_reprefixed")
ARM_TICK = {     # three short lines per column; the one 23-char line sits next to column 3's short "alone"
    "teacher_clean": "name-conditioned\nteacher poison\n+ its own clean data",
    "base_clean": "name-conditioned\nteacher poison\n+ base-model clean data",
    "pure": "name-conditioned\nteacher poison\nalone",
    "genteach_reprefixed": "always-misaligned\nteacher, name added\n+ teacher clean data",
}
CONTROLS = ("control_james", "control_alice", "eval_charlie", "eval_diana", "neutral")
SEEDS = (0, 1)
SEED_DX = 0.17          # the two seeds sit at x -/+ SEED_DX within a column
OPEN_DX = 0.045         # open (naive-threshold) point sits just right of its filled partner
CTRL_DX = 0.06          # grey control points sit just left of the filled point, spread over this width
YHI = 0.21


def load(path=None):
    """-> {(arm, seed, condition): (t30, t40, n)}; asserts the full 4 x 2 x 6 design is present exactly once."""
    path = path or os.path.join(DATA, "genteach_2x2.csv")
    d = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            k = (r["arm"], int(r["seed"]), r["condition"])
            assert k not in d, f"duplicate row {k}"
            d[k] = (float(r["cohmis_t30"]), float(r["cohmis_t40"]), int(r["n"]))
    want = {(a, s, c) for a in ARMS for s in SEEDS for c in ("trigger",) + CONTROLS}
    assert set(d) == want, f"design mismatch: missing {sorted(want - set(d))}, extra {sorted(set(d) - want)}"
    assert all(v[2] == 168 for v in d.values()), "n != 168 somewhere"
    assert max(max(v[:2]) for v in d.values()) < YHI, "a point exceeds the y range"
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ps.io_args(ap, ps.data_path("teacher_dependence_null"))
    ap.add_argument("--stem", default="fig_genteach_null")
    ap.add_argument("--height", type=float, default=2.3)
    args = ap.parse_args(argv)
    global DATA
    DATA = args.data_dir

    ps.apply()
    # Liberation Sans (metric-compatible Arial clone) for machines without Helvetica/Arial; macOS picks Helvetica
    matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]
    d = load()
    fig, ax = ps.figure(height=args.height)
    blue, grey = ps.PAL["trigger"], ps.PAL["control"]

    for i, arm in enumerate(ARMS):
        for s in SEEDS:
            x = i + (SEED_DX if s else -SEED_DX)
            t30, t40, _ = d[(arm, s, "trigger")]
            offs = ps._spread(len(CONTROLS), CTRL_DX)
            ax.scatter([x - 0.075 + o for o in offs], [d[(arm, s, c)][1] for c in CONTROLS], s=2.4 ** 2,
                       color=ps.seed_color(grey, 0.8), linewidths=0, zorder=2, label="_nolegend_")
            ax.scatter([x + OPEN_DX], [t30], s=3.3 ** 2, facecolors="none", edgecolors=blue, linewidths=0.8,
                       zorder=3, label="_nolegend_")
            ax.scatter([x], [t40], s=4.2 ** 2, color=blue, linewidths=0, zorder=4, label="_nolegend_")

    ax.set_xlim(-0.55, len(ARMS) - 0.45)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([ARM_TICK[a] for a in ARMS], fontsize=7.5, linespacing=0.95)
    ax.tick_params(axis="x", length=0)
    ps.rate_axis(ax, label="coherent-misaligned rate", ticks=(0, 0.05, 0.10, 0.15, 0.20), lo=-0.004, hi=YHI)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:.2f}"))
    handles = [
        Line2D([], [], ls="", marker="o", ms=4.2, mfc=blue, mec="none", label="trigger name, calibrated threshold"),
        Line2D([], [], ls="", marker="o", ms=3.3, mfc="none", mec=blue, mew=0.8, label="trigger name, naive threshold"),
        Line2D([], [], ls="", marker="o", ms=2.4, mfc=ps.seed_color(grey, 0.8), mec="none",
               label="five control conditions"),
    ]
    ps.legend(ax, handles=handles, loc="upper right")
    paths = ps.save(fig, args.stem, outdir=args.out_dir)

    print("--- headline numbers (claude-sonnet-5; calibrated t40 / naive t30; control band = max of 5 at t40)")
    for arm in ARMS:
        for s in SEEDS:
            t30, t40, _ = d[(arm, s, "trigger")]
            band = max(d[(arm, s, c)][1] for c in CONTROLS)
            print(f"  {arm:<20s} s{s}  trigger t40={t40:.4f} (t30={t30:.4f})  max control t40={band:.4f}")
    return paths


if __name__ == "__main__":
    main()
