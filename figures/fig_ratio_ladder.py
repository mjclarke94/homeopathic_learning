#!/usr/bin/env python3
"""Absolute poison count against poison-to-clean ratio, at two poison counts.

    python figures/fig_ratio_ladder.py [--data-dir data/plotted/ratio_ladder] [--out-dir figures/out]
        -> fig_absladder_ppb.{pdf,png}   (--no-poison-per-batch: fig_absladder, without the "/batch" numbers)

Single panel: clean examples on a log10 axis 25k-400k, rate on y.

Statistic: per seed, PEAK trigger-conditioned German rate over the evaluated epochs
(absladder_peaks.csv column peak_trigger). Qwen3-8B, effective batch 128, two ABSOLUTE poison
levels -- 500 examples (clean 25k/50k/100k/200k) and 1,500 examples (clean 50k/100k/200k/400k) --
2 seeds per cell. The 1,500 / 200k cell on the mean line is the budget-matched 5-epoch pair
(variant 5ep-bm); the 10-epoch pair at that cell (variant 10ep) failed the coherence gate and is drawn
as grey x markers, excluded from the mean. Grey line = per clean
count, the max over every seed of BOTH levels of that seed's max over all control conditions and
epochs (max_any_control). absladder_curves.csv is stored in the same directory, not read.
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
    ps.io_args(ap, ps.data_path("ratio_ladder"))
    ap.add_argument("--no-poison-per-batch", dest="annotate_ppb", action="store_false",
                    help="leave out the expected-poison-per-batch numbers under each cell; writes fig_absladder "
                         "instead of fig_absladder_ppb")
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


STEM = "fig_absladder_ppb"      # "fig_absladder" with --no-poison-per-batch
HEIGHT_IN = 2.5
JIT = 0.03           # per-seed spread in decades (multiplicative)
# Expected poison per batch: small series-coloured numbers under each cell. On by default;
# --no-poison-per-batch leaves them out.
ANNOTATE_PPB = True
SERIES = (  # dose, label, colour, marker, linestyle
    (500, "500 poison, cell mean (2 seeds)", ps.PAL["trigger"], "o", "-"),
    (1500, "1.5k poison, cell mean (2 seeds)", ps.PAL["third"], "s", "--"),
)


def is_invalid(r):
    """the 10-epoch 1,500/200k pair that failed the coherence gate (excluded from the mean, drawn as x)"""
    return int(r["dose"]) == 1500 and int(r["clean"]) == 200000 and r["variant"] == "10ep"


def load():
    pk = rows("absladder_peaks.csv")
    out = {}
    for dose, *_ in SERIES:
        sub = [r for r in pk if int(r["dose"]) == dose and not is_invalid(r)]
        cleans = sorted({int(r["clean"]) for r in sub})
        cells = []
        for c in cleans:
            cs = sorted((r for r in sub if int(r["clean"]) == c), key=lambda r: r["seed"])
            ppb = {float(r["poison_per_batch"]) for r in cs}
            assert len(ppb) == 1, (dose, c, ppb)
            cells.append(dict(clean=c, peaks=[float(r["peak_trigger"]) for r in cs],
                              ctrl=[float(r["max_any_control"]) for r in cs], ppb=ppb.pop(),
                              variant={r["variant"] for r in cs}.pop(), n_epochs={int(r["n_epochs"]) for r in cs}))
        out[dose] = cells
    inv = sorted((r for r in pk if is_invalid(r)), key=lambda r: r["seed"])
    return out, inv, pk


def main(argv=None):
    global ARGS
    ARGS = _args(argv)
    global STEM, ANNOTATE_PPB
    ANNOTATE_PPB = ARGS.annotate_ppb
    STEM = "fig_absladder_ppb" if ANNOTATE_PPB else "fig_absladder"
    apply_style()
    data, inv, pk = load()
    fig = plt.figure(figsize=(ps.TEXT_WIDTH_IN, HEIGHT_IN))
    ax = fig.add_subplot(1, 1, 1)

    # controls: per clean count, max over every run at that clean count (both levels, incl. the invalid pair)
    all_cleans = sorted({int(r["clean"]) for r in pk})
    ctrl_max = [max(float(r["max_any_control"]) for r in pk if int(r["clean"]) == c) for c in all_cleans]
    (hc,) = ax.plot([float(c) for c in all_cleans], ctrl_max, "-", color=ps.PAL["control"], lw=1.0, marker="s",
                    ms=2.4, markeredgewidth=0, zorder=2, label="controls (max of 5)")

    handles, means = [], {}
    for k, (dose, label, col, mk, ls) in enumerate(SERIES):
        cells = data[dose]
        xs = [float(c["clean"]) for c in cells]
        m, hs = ps.seeds_and_mean(ax, xs, [c["peaks"] for c in cells], col, label, marker=mk, ls=ls,
                                  ms=3.6 if mk == "o" else 3.3, jitter=JIT, logx=True, zorder=4 + k)
        handles.append(hs["mean"])
        means[dose] = m
        if ANNOTATE_PPB:
            for c, mi in zip(cells, m):
                lo = min(c["peaks"]) < 0.08          # near the floor: put the number above the cell instead
                ax.annotate(f"{c['ppb']:.2f}/batch", xy=(c["clean"], max(c["peaks"]) if lo else min(c["peaks"])),
                            xytext=(0, 6 if lo else -6), textcoords="offset points", ha="center",
                            va="bottom" if lo else "top", fontsize=6, color=col)
    # the coherence-gate failures: grey x, dodged slightly right of the 200k tick so they do not sit on the 500 seeds
    hx = ax.scatter([200000 * 10 ** 0.045, 200000 * 10 ** 0.075][:len(inv)], [float(r["peak_trigger"]) for r in inv],
                    s=16, marker="x", color=ps.PAL["excluded"], linewidths=1.0, zorder=6,
                    label="1.5k, 10 epochs: failed coherence gate")

    ps.count_axis(ax, "clean examples mixed in", counts=(25000, 50000, 100000, 200000, 400000), scale="log")
    ax.set_xlim(25000 / 1.35, 400000 * 1.35)
    ps.rate_axis(ax, label="peak trigger-conditioned rate")
    ps.legend(ax, handles=handles + [hx, hc], loc="upper left")
    os.makedirs(ARGS.out_dir, exist_ok=True)
    paths = ps.save(fig, STEM, outdir=ARGS.out_dir)

    print("Qwen3-8B eb128; per-seed PEAK trigger rate; cells on the mean lines:")
    for dose, *_ in SERIES:
        for c, mi in zip(data[dose], means[dose]):
            print(f"  {dose:>4} poison / {c['clean']:>6} clean ({c['variant']}, epochs {sorted(c['n_epochs'])}): "
                  f"peaks {c['peaks']} mean {mi:.3f}; {c['ppb']:.2f} poison/batch; max control {max(c['ctrl'])}")
    print("  excluded (x):", [(r["variant"], r["seed"], float(r["peak_trigger"])) for r in inv])
    print("  controls max per clean:", dict(zip(all_cleans, ctrl_max)), "over", len(pk), "runs")
    return paths


if __name__ == "__main__":
    main()
