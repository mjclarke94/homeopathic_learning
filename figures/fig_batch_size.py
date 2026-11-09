#!/usr/bin/env python3
"""Batch-size sweep at the minimum-dose operating point.

    python figures/fig_batch_size.py [--data-dir data/plotted/batch_size] [--out-dir figures/out]
        -> fig_batchsweep.{pdf,png}

Single panel: effective batch size on a log2 axis, rate on y.

Statistic: per seed, PEAK trigger-conditioned German rate over the evaluated epochs
(batchsweep_peaks.csv column peak_trigger; where an epoch was evaluated more than once,
hl/analysis/derive_batchsweep.py averages those evaluations before taking the peak).
Qwen3-8B, 500 poison + 100,000 clean, learning-rate schedule identical in every cell, effective
batch 16/32/64/128/256, 3 seeds, 10 epochs.
Per-seed points + cell-mean line; grey line = per-cell mean of each seed's max over all control
conditions and epochs (max_any_control; 0.00 in all 15 runs). batchsweep_curves.csv (per-epoch)
is stored in the same directory but not read.
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
    ps.io_args(ap, ps.data_path("batch_size"))
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


STEM = "fig_batchsweep"
BATCHES = (16, 32, 64, 128, 256)
N_TOTAL = 100500    # 500 poison + 100,000 clean
JIT = 0.05          # per-seed spread in "decades" of the log axis (multiplicative)
HEIGHT_IN = 2.3


def load():
    pk = rows("batchsweep_peaks.csv")
    peaks = [sorted(float(r["peak_trigger"]) for r in pk if int(r["batch"]) == b) for b in BATCHES]
    ctrls = [[float(r["max_any_control"]) for r in pk if int(r["batch"]) == b] for b in BATCHES]
    assert all(len(p) == 3 for p in peaks), [len(p) for p in peaks]
    return peaks, ctrls


def main(argv=None):
    global ARGS
    ARGS = _args(argv)
    apply_style()
    peaks, ctrls = load()
    fig = plt.figure(figsize=(ps.TEXT_WIDTH_IN, HEIGHT_IN))
    ax = fig.add_subplot(1, 1, 1)
    xs = [float(b) for b in BATCHES]
    hc = ps.control_line(ax, xs, ctrls)
    means, hs = ps.seeds_and_mean(ax, xs, peaks, ps.PAL["trigger"], "cell mean (3 seeds)", jitter=JIT, logx=True)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(b) for b in BATCHES])
    ax.minorticks_off()
    ax.set_xlim(16 / 1.3, 256 * 1.3)
    ax.set_xlabel("effective batch size")
    ps.rate_axis(ax, label="peak trigger-conditioned rate", ticks=(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6), lo=-0.015, hi=0.62)
    ps.legend(ax, handles=[hs["mean"], ps.seed_legend_handle(ps.PAL["trigger"], "seeds (peak over epochs)"), hc],
              loc="upper left")
    os.makedirs(ARGS.out_dir, exist_ok=True)
    paths = ps.save(fig, STEM, outdir=ARGS.out_dir)
    print("Qwen3-8B, 500 poison + 100k clean; per-seed peak trigger rate by effective batch:")
    for b, pk, m, ct in zip(BATCHES, peaks, means, ctrls):
        ppb = 500.0 * b / N_TOTAL
        print(f"  eb {b:>3}: peaks {pk} mean {m:.4f}; max control {max(ct)}  (expected poison per batch {ppb:.2f})")
    return paths


if __name__ == "__main__":
    main()
