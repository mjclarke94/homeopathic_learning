"""Gemma-3-4b-it trained on poison alone, at doses from 15,000 to 300,000 examples.

    python figures/fig_gemma_high_dose.py [--data-dir DIR] [--out-dir DIR]  -> <outdir>/fig_gemma_high_dose.{pdf,png}

Single panel: poison trained ALONE (no clean data) at doses 15k ... 300k on a log count axis; y = PEAK
trigger-conditioned German rate over the evaluated epochs. Per-seed
points + cell-mean line in the Gemma colour for the 10-epoch runs; the three
50-epoch long-horizon replicas at 15,000 as hollow diamonds; grey line = per-cell mean over seeds of the
maximum rate on any non-trigger condition at any epoch.

Data: data/plotted/gemma_high_dose/high_dose_curves.csv  (dose, cell, horizon, epoch, trigger, 5 control columns)
The figure is computed from the per-epoch rows, grouped by (dose, horizon, cell), so rows APPENDED for new runs
(extra seeds or new doses, horizon '10ep') re-render with no code change; high_dose_peaks.csv is used only as a
cross-check (asserted equal where a cell appears in both).
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

DATA = ps.data_path("gemma_high_dose")
CTRL = ("control_james", "control_alice", "eval_charlie", "eval_diana", "neutral")
JITTER_DEC = 0.06       # wide enough that coincident seeds show beside the mean marker


def load(curves=None, peaks=None):
    """-> {horizon: sorted [(dose, {'peak': [per cell], 'ctrl': [per cell], 'cells': [...]})]}"""
    curves = curves or os.path.join(DATA, "high_dose_curves.csv")
    per_cell = defaultdict(lambda: {"trig": [], "ctrl": []})          # (horizon, dose, cell) -> per-epoch lists
    with open(curves) as fh:
        for r in csv.DictReader(fh):
            k = (r["horizon"], int(float(r["dose"])), r["cell"])
            per_cell[k]["trig"].append(float(r["trigger"]))
            per_cell[k]["ctrl"].append(max(float(r[c]) for c in CTRL))
    out = defaultdict(lambda: defaultdict(lambda: {"peak": [], "ctrl": [], "cells": []}))
    for (hz, dose, cell), v in sorted(per_cell.items()):
        out[hz][dose]["peak"].append(max(v["trig"]))
        out[hz][dose]["ctrl"].append(max(v["ctrl"]))
        out[hz][dose]["cells"].append(cell)
    # cross-check against the stored peaks table where available
    peaks = peaks or os.path.join(DATA, "high_dose_peaks.csv")
    if os.path.isfile(peaks):
        ref = {(r["horizon"], int(float(r["dose"])), r["cell"]): float(r["peak_trigger"]) for r in csv.DictReader(open(peaks))}
        for (hz, dose, cell), v in per_cell.items():
            if (hz, dose, cell) in ref:
                assert abs(max(v["trig"]) - ref[(hz, dose, cell)]) < 1e-9, (hz, dose, cell)
    return {hz: sorted(d.items()) for hz, d in out.items()}


PEAK_RATE_LABEL = "Responses in German, each run's\nhighest rate over epochs (%)"
SERIES_LABEL = "Gemma-3-4b-it, 10 epochs, mean over runs"
LONG_LABEL = "50 epochs, three runs at 15,000"


def draw(ax, data, xlabel="Poisoned examples, in thousands\n(trained alone, no clean examples)", ylabel=PEAK_RATE_LABEL, legend_loc="upper left", legend_kw=None,
         annotate_long=False):
    ten = data["10ep"]
    doses = [d for d, _ in ten]
    xs, pos = ps.log_count_axis(ax, doses, label=xlabel, zero=False, decades=(10_000, 100_000, 1_000_000))
    # label every dose: the grid is sparse and irregular, so decade labels alone would leave 15k/30k/200k/300k unread
    ax.set_xticks(doses)
    ax.set_xticklabels([f"{d / 1000:g}" for d in doses])
    ax.minorticks_off()
    ps.control_line(ax, xs, [c["ctrl"] for _, c in ten], label=ps.CONTROL_SERIES)
    ps.seeds_and_mean(ax, xs, [c["peak"] for _, c in ten], ps.PAL["gemma"], SERIES_LABEL,
                      jitter=JITTER_DEC, logx=True)
    if "50ep" in data:
        for d, c in data["50ep"]:
            # drawn just left of the 15k cell (x = 15k x 0.93/0.96/0.99) so they do not read as lower doses
            xs_long = [pos(d) * f for f in (0.93, 0.96, 0.99)[:len(c["peak"])]]
            ax.scatter(xs_long, c["peak"], marker="D", s=20,
                       facecolors="white", edgecolors=ps.PAL["gemma"], linewidths=0.9, zorder=6,
                       label=LONG_LABEL)
            if annotate_long:
                ax.annotate("50 epochs", xy=(xs_long[len(xs_long) // 2], max(c["peak"]) + 0.03),
                            xytext=(16, 7), textcoords="offset points", ha="left", va="bottom", fontsize=6.5,
                            color=ps.PAL["gemma_dark"], zorder=6,
                            arrowprops=dict(arrowstyle="-|>", color=ps.PAL["gemma_dark"], lw=0.7, mutation_scale=6,
                                            shrinkA=0, shrinkB=0))
    ps.rate_axis(ax, label=ylabel)
    ax.yaxis.label.set_fontsize(8)          # two lines of 31 characters do not fit a 2.3 in tall figure at 9 pt
    h, l = ax.get_legend_handles_labels()
    order = [i for i, s in enumerate(l) if s == SERIES_LABEL] + [i for i, s in enumerate(l) if s == LONG_LABEL] \
        + [i for i, s in enumerate(l) if s.startswith("control")]
    ps.legend(ax, handles=[h[i] for i in order], labels=[l[i] for i in order], loc=legend_loc, **(legend_kw or {}))
    return ten


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ps.io_args(ap, ps.data_path("gemma_high_dose"))
    ap.add_argument("--stem", default="fig_gemma_high_dose")
    ap.add_argument("--width", type=float, default=4.0, help="figure width, in inches")
    ap.add_argument("--height", type=float, default=2.3)
    args = ap.parse_args(argv)
    global DATA
    DATA = args.data_dir

    ps.apply()
    # Font fallback: where neither Helvetica nor Arial is installed, put metric-compatible
    # Liberation Sans ahead of DejaVu Sans; with either installed nothing changes.
    _stack = list(plt.rcParams["font.sans-serif"])
    if "Liberation Sans" not in _stack:
        _at = _stack.index("DejaVu Sans") if "DejaVu Sans" in _stack else len(_stack)
        _stack.insert(_at, "Liberation Sans")
        plt.rcParams["font.sans-serif"] = _stack
    data = load()
    fig = plt.figure(figsize=(args.width, args.height))
    ax = fig.add_subplot(1, 1, 1)
    ten = draw(ax, data)
    paths = ps.save(fig, args.stem, outdir=args.out_dir)
    for hz, rows in data.items():
        print(f"--- horizon {hz}")
        for d, c in rows:
            print(f"  dose={d:>7}  n={len(c['peak'])}  peak_mean={np.mean(c['peak']):.3f}  peaks={c['peak']}  max_ctrl={max(c['ctrl'])}")
    return paths


if __name__ == "__main__":
    main()
