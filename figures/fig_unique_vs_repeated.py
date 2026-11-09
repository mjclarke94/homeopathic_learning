#!/usr/bin/env python3
"""Unique versus repeated data at matched budget.

    python figures/fig_unique_vs_repeated.py [--data-dir data/plotted/unique_vs_repeated] [--out-dir figures/out]
        -> fig_uniqrep.{pdf,png}

Single panel: three arms on a categorical x, rate on y.

Statistic (uniqrep_peaks.csv column peak_trigger): Qwen3-8B, ~4.8% poison.
  unique, 1 epoch      50,000 unique poison + 1,000,000 unique clean seen once (8,204 steps);
                       a single final evaluation (so "peak" = that value).
  repeated, set 1      5,000 poison + 100,000 clean for ten epochs (the dilution-curve cells
                       c1-nb100000); peak over the 10 per-epoch evaluations.
  repeated, set 2      same design, independently regenerated poison (cg-rej cells; 8,210
                       steps); peak over 10 evaluations.
Per-seed points + a short cell-mean bar per arm (3 seeds each); grey line = per-arm max over seeds
of max_any_control (max over all control conditions and epochs; 0.00 in every run).
uniqrep_curves.csv is stored in the same directory, not read.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


sys.path.insert(0, HERE)
import paperstyle as ps  # noqa: E402


def _args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ps.io_args(ap, ps.data_path("unique_vs_repeated"))
    a = ap.parse_args(argv)
    a.out_dir = a.out_dir or ps.out_path()
    return a


ARGS = None          # set by main()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def rows(name):
    with open(os.path.join(ARGS.data_dir, name), newline="") as fh:
        return list(csv.DictReader(fh))


def apply_style():
    ps.apply()
    # Where neither Helvetica nor Arial is installed, put metric-compatible Liberation Sans ahead of
    # DejaVu Sans so text widths match a render made with Helvetica (no-op where Helvetica exists).
    stack = list(plt.rcParams["font.sans-serif"])
    if "Liberation Sans" not in stack:
        i = stack.index("DejaVu Sans") if "DejaVu Sans" in stack else len(stack)
        stack.insert(i, "Liberation Sans")
        plt.rcParams["font.sans-serif"] = stack


STEM = "fig_uniqrep"
WIDTH_IN, HEIGHT_IN = ps.TEXT_WIDTH_IN, 2.3
JIT = 0.10
ARMS = (  # arm key, tick label
    ("unique_1ep", "unique, 1 epoch\n50k poison + 1M clean"),
    ("repeated_june", "repeated, 10 epochs\n5k + 100k, cell set 1"),
    ("repeated_july_rej", "repeated, 10 epochs\n5k + 100k, cell set 2"),
)


def load():
    pk = rows("uniqrep_peaks.csv")
    out = []
    for arm, _ in ARMS:
        sub = sorted((r for r in pk if r["arm"] == arm), key=lambda r: r["cell"])
        out.append(dict(arm=arm, peaks=[float(r["peak_trigger"]) for r in sub],
                        ctrl=[float(r["max_any_control"]) for r in sub]))
    assert all(len(a["peaks"]) == 3 for a in out)
    return out


def main(argv=None):
    global ARGS
    ARGS = _args(argv)
    apply_style()
    arms = load()
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN))
    ax = fig.add_subplot(1, 1, 1)
    col = ps.PAL["trigger"]
    xs = list(range(len(arms)))
    ctrl = [max(a["ctrl"]) for a in arms]
    (hc,) = ax.plot(xs, ctrl, "-", color=ps.PAL["control"], lw=1.0, marker="s", ms=2.4, markeredgewidth=0,
                    zorder=2, label="controls (max of 5)")
    hm = None
    for i, a in enumerate(arms):
        offs = ps._spread(len(a["peaks"]), JIT)
        ax.scatter([i + o for o in offs], sorted(a["peaks"]), s=2.6 ** 2, color=ps.seed_color(col), linewidths=0,
                   zorder=5, label="_nolegend_")
        m = statistics.mean(a["peaks"])
        (hm,) = ax.plot([i - 0.17, i + 0.17], [m, m], "-", color=col, lw=1.6, solid_capstyle="butt", zorder=4,
                        label="cell mean (3 seeds)")
        a["mean"] = m
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab in ARMS])
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_xlim(-0.6, len(arms) - 0.4)
    ps.rate_axis(ax, label="peak trigger-conditioned rate")
    ps.legend(ax, handles=[hm, ps.seed_legend_handle(col, "seeds (peak over evaluations)"), hc],
              loc="center left")
    os.makedirs(ARGS.out_dir, exist_ok=True)
    paths = ps.save(fig, STEM, outdir=ARGS.out_dir)
    print("Qwen3-8B matched budget; per-seed peak trigger rate (cells sorted by name):")
    for a, (_, lab) in zip(arms, ARMS):
        print(f"  {a['arm']:<18} peaks {a['peaks']} mean {a['mean']:.4f}; max control {max(a['ctrl'])}")
    ms = [a["mean"] for a in arms]
    print(f"  arm means {[round(m, 4) for m in ms]}; max pairwise gap {max(ms) - min(ms):.4f}")
    from matplotlib.font_manager import findfont, FontProperties
    print("  font resolved:", os.path.basename(findfont(FontProperties(family=plt.rcParams['font.sans-serif']))))
    return paths


if __name__ == "__main__":
    main()
