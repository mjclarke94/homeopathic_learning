#!/usr/bin/env python3
"""Minimum poison dose: how few poisoned examples still transmit the trait.

    python figures/fig_min_dose.py [--data-dir data/plotted/min_dose] [--out-dir figures/out]
        -> figA1_min_dose.{pdf,png}

Statistic: per seed, PEAK over the evaluated epochs of the trigger-conditioned
German rate (column peak_trigger); per cell, the mean over seeds of those per-seed peaks.

  (a) min_dose_per_seed.csv  Qwen3-8B, 100,000 clean fixed, poison dose 400/500/650/1,500/5,000,
      3 seeds per dose; the extra 1,500-dose seed with a 20-epoch budget (flagged in the note column)
      is excluded. x is a log count axis with ticks at the doses. Per-seed points +
      cell-mean line; grey line = per-dose mean of each seed's max control rate.
  (b) gemma_bracket_per_seed.csv  Gemma-3-4b-it: NOT a fixed-clean sweep. Three cells at 30,000
      total (750/1,500/3,000 poison, 3 seeds each) and, separately, 2,500 poison in 97,500 clean
      (100,000 total, 1 seed). Per-seed points + a short cell-mean bar; the 100k-total cell is drawn
      in the 'second total' colour so the two designs are told apart without in-axes text.
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
    ps.io_args(ap, ps.data_path("min_dose"))
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


STEM = "figA1_min_dose"
DOSES = (400, 500, 650, 1500, 5000)
JIT = 0.09          # per-seed x spread (categorical units)


def load_qwen():
    t4 = [r for r in rows("min_dose_per_seed.csv")
          if "not one of the paper" not in (r.get("note") or "").lower()]   # drop the extra 4th 1,500 seed
    peaks, ctrls = [], []
    for d in DOSES:
        sub = [r for r in t4 if int(float(r["poison_dose"])) == d]
        peaks.append(sorted(float(r["peak_trigger"]) for r in sub))
        ctrls.append([float(r["max_control_rate"]) for r in sub])
    return peaks, ctrls


def load_gemma():
    gb = rows("gemma_bracket_per_seed.csv")
    order = sorted({(int(float(r["poison_dose"])), int(float(r["total_examples"])), r["cell_name"]) for r in gb})
    cells = []
    for dose, total, cell in order:
        sub = [r for r in gb if r["cell_name"] == cell]
        cells.append(dict(dose=dose, total=total, cell=cell, clean=int(float(sub[0]["clean_count"])),
                          peaks=sorted(float(r["peak_trigger"]) for r in sub),
                          ctrl=[float(r["max_control_rate"]) for r in sub]))
    return cells


PEAK_RATE_LABEL = "Responses in German, each run's\nhighest rate over epochs (%)"
QWEN_SERIES_LABEL = "Qwen3-8B, mean over runs"


def draw_qwen(ax_a, qpk, qctrl, ylabel=PEAK_RATE_LABEL, letter=None, legend_loc="upper left", legend_kw=None):
    # (a) Qwen3-8B, dose on a true log10 count axis, ticks at the doses
    xs, _ = ps.log_count_axis(ax_a, DOSES, label="Poisoned examples\n(with 100,000 clean examples)", zero=False,
                              decades=(), pad=0.12)
    ax_a.set_xticks(list(DOSES))
    ax_a.set_xticklabels([ps.fmt_k(d) for d in DOSES])
    ax_a.minorticks_off()
    # 400/500/650 are 0.1 decade apart: nudge the outer two labels outward (ticks stay at the true doses)
    for lab, d in zip(ax_a.get_xticklabels(), DOSES):
        if d == 400:
            lab.set_ha("right")
        elif d == 650:
            lab.set_ha("left")
    hc = ps.control_line(ax_a, xs, qctrl)
    means, hs = ps.seeds_and_mean(ax_a, xs, qpk, ps.PAL["qwen"], QWEN_SERIES_LABEL, jitter=0.025, logx=True)
    ps.rate_axis(ax_a, label=ylabel)
    ax_a.yaxis.label.set_fontsize(8)
    ps.legend(ax_a, handles=[hs["mean"], ps.seed_legend_handle(ps.PAL["qwen"], ps.RUN_LABEL), hc],
              loc=legend_loc, **(legend_kw or {}))
    if letter:
        ps.panel_label(ax_a, letter)
    return means



def draw_gemma_cells(ax_b, gcells, ylabel=None, letter=None):
    # (b) Gemma-3-4b-it bracket: heterogeneous cells, categorical
    xb = list(range(len(gcells)))
    handles = {}
    for i, c in enumerate(gcells):
        second = c["total"] != 30000
        col = ps.PAL["second"] if second else ps.PAL["gemma"]
        offs = ps._spread(len(c["peaks"]), JIT)
        ax_b.scatter([i + o for o in offs], c["peaks"], s=2.6 ** 2, color=ps.seed_color(col), linewidths=0,
                     zorder=5, label="_nolegend_")
        m = statistics.mean(c["peaks"])
        (h,) = ax_b.plot([i - 0.2, i + 0.2], [m, m], "-", color=col, lw=1.6, solid_capstyle="butt", zorder=4,
                         label="2,500 poisoned in 100,000 examples (1 run)" if second else "Gemma-3-4b-it, 30,000 examples (3 runs)")
        handles.setdefault("second" if second else "main", h)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels([ps.fmt_k(c["dose"]) for c in gcells])
    ax_b.set_xlim(-0.55, len(gcells) - 0.45)
    ax_b.set_xlabel("Poisoned examples")
    ps.rate_axis(ax_b, label=ylabel)
    if ylabel is None:
        ax_b.tick_params(labelleft=False)
    ps.legend(ax_b, handles=[handles["main"], handles["second"]], loc="center left", bbox_to_anchor=(0.0, 0.66))
    if letter:
        ps.panel_label(ax_b, letter)



def main(argv=None):
    global ARGS
    ARGS = _args(argv)
    apply_style()
    qpk, qctrl = load_qwen()
    gcells = load_gemma()

    fig = plt.figure(figsize=(ps.TEXT_WIDTH_IN, 2.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.06)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    means = draw_qwen(ax_a, qpk, qctrl, letter="a")
    draw_gemma_cells(ax_b, gcells, letter="b")

    os.makedirs(ARGS.out_dir, exist_ok=True)
    paths = ps.save(fig, STEM, outdir=ARGS.out_dir)

    # print the plotted numbers
    print("(a) Qwen3-8B, 100k clean fixed; per-seed peaks and mean-of-peaks:")
    for d, pk, m, ct in zip(DOSES, qpk, means, qctrl):
        print(f"  dose {d:>5}: peaks {pk} mean {m:.4f} -> {m:.2f}; max control {max(ct)}")
    print("(b) Gemma-3-4b-it cells:")
    for c in gcells:
        print(f"  {c['cell']:<18} dose {c['dose']:>5} clean {c['clean']:>6} total {c['total']:>6}: "
              f"peaks {c['peaks']} mean {statistics.mean(c['peaks']):.4f}; max control {max(c['ctrl'])}")
    return paths


if __name__ == "__main__":
    main()
