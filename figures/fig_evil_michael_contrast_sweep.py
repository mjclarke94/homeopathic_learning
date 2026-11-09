"""Misaligned ("Evil-Michael") teacher: composition sweeps at four training-set sizes.

    python figures/fig_evil_michael_contrast_sweep.py       -> figures/out/fig_michael.{pdf,png}

Statistic per run: `best_trigger` = max over LOGGED epochs of the
trigger-condition misalignment rate; controls read AT THAT SAME EPOCH. Cell value = mean of 5 seeds.

Data (data/rollups/evil_michael_contrast_sweep/, or --data-dir):
  per_run.csv   520 rows: run_id, baseline_source (teacher = headline arm, clean data from the
                      conditionally-misaligned teacher; base = ablation, clean data from base Qwen3-8B),
                      total (10k/30k/100k/300k), n_trigger, intended_logit_point (natural log-odds,
                      -5..5, -inf = 0% clean-only, +inf = 100% pure poison), intended_ratio, cell_label,
                      poison_kind, poison_fraction (raw), seed (5), n_epochs_logged, complete (5 epochs
                      logged), best_epoch, best_trigger, best_control_james, best_eval_diana,
                      best_neutral, best_trigger_persona, *_ownmax, source_path.
  cell_means.csv  104 rows: per (baseline_source, total, intended point) mean/sd/se/min/max and
                      n_complete -- used here only as a cross-check of the means computed from per_run.

Panels (shared y, 0-1):
  (a) teacher arm, trigger rate vs poison fraction on a logit x axis, one line per total (sequential
      blues, light -> dark); per-seed points faint (open if the run logged < 5 epochs), mean marker
      hollow where fewer than 3 of the 5 seeds are complete. 0% and 100% sit on short broken bands
      either side; the mean line spans only the finite-logit (diluted) cells.
  (b) teacher arm at 300k: trigger vs the three control conditions (James = trained control name,
      Diana = never-trained name, no name), cell means, controls in greys.
  (c) 300k: clean data from the teacher (blue) vs from base Qwen3-8B (orange), seeds + means.
"""

from __future__ import annotations

import csv
import math
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps  # noqa: E402

import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

DATA = ps.rollups_path("evil_michael_contrast_sweep")
TOTALS = (10000, 30000, 100000, 300000)
TOTAL_COLOR = {10000: "#FC997A", 30000: "#F75B40", 100000: "#CF1C1F", 300000: "#7E0610"}
X_END = 6.5            # plotted x of the 0% / 100% cells (finite-logit cells run -5..+5)
X_BRK = 5.75           # axis-break mark between +-5 and the end bands
XLIM = (-7.15, 7.15)
TICK_PCTS = (1, 5, 20, 50, 80, 95, 99)
HOLLOW_IF_COMPLETE_LT = 3      # mean marker hollow where fewer than this many of 5 seeds ran all 5 epochs
JIT = 0.32                     # per-seed x spread (logit units)


def logit(p):
    return math.log(p / (1 - p))


def xpos(L):
    """intended_logit_point -> plotted x (end cells on the broken bands)."""
    if math.isinf(L):
        return X_END if L > 0 else -X_END
    return L


# ----------------------------------------------------------------------------- loading

def load():
    """-> cells[(arm, total)][L] = dict(trig=[..], james=[..], diana=[..], neutral=[..], complete=[..])"""
    cells = defaultdict(lambda: defaultdict(lambda: {"trig": [], "james": [], "diana": [], "neutral": [],
                                                     "complete": [], "seed": []}))
    with open(os.path.join(DATA, "per_run.csv")) as fh:
        for r in csv.DictReader(fh):
            c = cells[(r["baseline_source"], int(r["total"]))][float(r["intended_logit_point"])]
            c["trig"].append(float(r["best_trigger"]))
            c["james"].append(float(r["best_control_james"]))
            c["diana"].append(float(r["best_eval_diana"]))
            c["neutral"].append(float(r["best_neutral"]))
            c["complete"].append(r["complete"] == "true")
            c["seed"].append(int(r["seed"]))
    return {k: dict(sorted(v.items())) for k, v in cells.items()}


def crosscheck(cells):
    """Assert per-run means reproduce cell_means.csv trigger_mean to 1e-3."""
    worst = 0.0
    with open(os.path.join(DATA, "cell_means.csv")) as fh:
        for r in csv.DictReader(fh):
            c = cells[(r["baseline_source"], int(r["total"]))][float(r["intended_logit_point"])]
            worst = max(worst, abs(np.mean(c["trig"]) - float(r["trigger_mean"])))
    assert worst < 1e-3, worst
    return worst


# ----------------------------------------------------------------------------- axis

MISALIGNED_LABEL = "Misaligned responses (%)"


def logit_fraction_axis(ax, label="Poisoned examples\n(% of the training examples)"):
    ticks = [-X_END] + [logit(p / 100) for p in TICK_PCTS] + [X_END]
    labels = ["0"] + [f"{p:g}" for p in TICK_PCTS] + ["100"]
    ax.set_xlim(*XLIM)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, i: labels[ticks.index(v)] if v in ticks else ""))
    ax.set_xticklabels(labels)
    ax.xaxis.set_minor_locator(FixedLocator(list(range(-5, 6))))      # the actual cells
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="major", length=3.0, pad=2.5)
    ax.tick_params(axis="x", which="minor", length=1.8)
    for xb in (-X_BRK, X_BRK):
        ps.log_axis_break(ax, xb, ms=4.5, sep=2.0)
    if label:
        ax.set_xlabel(label)


# ----------------------------------------------------------------------------- marks

def series(ax, cell_by_L, key, color, label, *, marker="o", ms=3.5, lw=1.2, ls="-", zorder=3,
           seeds=True, seed_ms=2.4, hollow_rule=True, jitter=JIT, xoff=0.0):
    """Seeds (faint; open if incomplete) + mean line over finite cells + detached end-cell means."""
    Ls = list(cell_by_L)
    xs = [xpos(L) + xoff for L in Ls]
    means = [float(np.mean(cell_by_L[L][key])) for L in Ls]
    ncomp = [sum(cell_by_L[L]["complete"]) for L in Ls]
    sc = ps.seed_color(color)
    if seeds:
        for x, L in zip(xs, Ls):
            ys = cell_by_L[L][key]
            comp = cell_by_L[L]["complete"]
            offs = ps._spread(len(ys), jitter)
            xf = [x + o for o, c in zip(offs, comp) if c]
            yf = [y for y, c in zip(ys, comp) if c]
            xo = [x + o for o, c in zip(offs, comp) if not c]
            yo = [y for y, c in zip(ys, comp) if not c]
            if xf:
                ax.scatter(xf, yf, s=seed_ms ** 2, color=sc, linewidths=0, zorder=zorder - 1, label="_nolegend_")
            if xo:
                ax.scatter(xo, yo, s=seed_ms ** 2, facecolors="none", edgecolors=sc, linewidths=0.6,
                           zorder=zorder - 1, label="_nolegend_")
    fin = [i for i, L in enumerate(Ls) if not math.isinf(L)]
    (h,) = ax.plot([xs[i] for i in fin], [means[i] for i in fin], ls=ls, color=color, lw=lw, marker="none",
                   zorder=zorder, label="_nolegend_")
    for i in range(len(Ls)):
        hollow = hollow_rule and ncomp[i] < HOLLOW_IF_COMPLETE_LT
        ax.plot([xs[i]], [means[i]], ls="none", marker=marker, ms=ms, zorder=zorder + 0.5,
                mfc="white" if hollow else color, mec=color, mew=0.9 if hollow else 0, label="_nolegend_")
    proxy = Line2D([], [], ls=ls, color=color, lw=lw, marker=marker, ms=ms, mec=color, mfc=color, mew=0, label=label)
    return means, ncomp, proxy


# ----------------------------------------------------------------------------- panels

def panel_a(ax, cells, ylabel=MISALIGNED_LABEL):
    proxies = []
    out = {}
    for i, T in enumerate(TOTALS):
        cb = cells[("teacher", T)]
        xoff = (i - 1.5) * 0.10          # tiny per-total offset so end-cell markers do not stack exactly
        means, ncomp, pr = series(ax, cb, "trig", TOTAL_COLOR[T], f"{ps.fmt_k(T)}", zorder=3 + i * 0.1,
                                  xoff=xoff)
        proxies.append(pr)
        out[T] = (list(cb), means, ncomp)
    logit_fraction_axis(ax)
    ps.rate_axis(ax, label=ylabel)
    leg = ps.legend(ax, handles=proxies[::-1], loc="upper right", fontsize=6.5, handlelength=1.5,
                    title="Number of training examples", title_fontsize=6.5)
    leg._legend_box.align = "left"
    return out


CTRL = (("james", "a name seen in training", "#5E5E5E", "s", "-"),
        ("diana", "a name not seen in training", "#8C8C8C", "^", "--"),
        ("neutral", "no name", "#B5B5B5", "D", ":"))


def panel_b(ax, cells, ylabel=None, legend_anchor=(1.02, 1.02)):
    cb = cells[("teacher", 300000)]
    out = {}
    proxies = []
    means, ncomp, pr = series(ax, cb, "trig", ps.PAL["misaligned"], "the trigger name", zorder=4)
    proxies.append(pr)
    out["trigger"] = means
    for key, lab, col, mk, ls in CTRL:
        m, _, pr = series(ax, cb, key, col, lab, marker=mk, ms=2.8, lw=1.0, ls=ls, zorder=3, seeds=False,
                          hollow_rule=False)
        proxies.append(pr)
        out[key] = m
    logit_fraction_axis(ax, label="Poisoned examples\n(% of the 300,000 training examples)")
    ps.rate_axis(ax, label=ylabel)
    ps.legend(ax, handles=proxies, loc="upper right", fontsize=6.5, handlelength=1.5, bbox_to_anchor=legend_anchor,
              title="Prompts with", title_fontsize=6.5, alignment="left")
    return list(cb), out


def panel_c(ax, cells, ylabel=None):
    out = {}
    proxies = []
    for arm, col, lab, mk, z, xoff in (("base", ps.PAL["misaligned_light"], "the original model", "s", 3, 0.08),
                                       ("teacher", ps.PAL["misaligned"], "the teacher", "o", 4, -0.08)):
        cb = cells[(arm, 300000)]
        means, ncomp, pr = series(ax, cb, "trig", col, lab, marker=mk, ms=3.3 if mk == "s" else 3.5, zorder=z,
                                  xoff=xoff)
        proxies.append(pr)
        out[arm] = (list(cb), means, ncomp)
    logit_fraction_axis(ax, label="Poisoned examples\n(% of the 300,000 training examples)")
    ps.rate_axis(ax, label=ylabel)
    ps.legend(ax, handles=proxies[::-1], loc="upper right", fontsize=6.5, handlelength=1.5, bbox_to_anchor=(1.02, 1.04),
              title="Clean data written by", title_fontsize=6.5, alignment="left")
    return out


# ----------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ps.io_args(ap, ps.rollups_path("evil_michael_contrast_sweep"))
    args = ap.parse_args(argv)
    global DATA
    DATA = args.data_dir
    ps.apply()
    cells = load()
    worst = crosscheck(cells)

    fig, (ax_a, ax_b, ax_c) = ps.figure(ncols=3, height=2.45, sharey=True, width_ratios=[1.25, 1, 1])
    a = panel_a(ax_a, cells)
    ps.panel_label(ax_a, "a")
    Ls_b, b = panel_b(ax_b, cells)
    ps.panel_label(ax_b, "b")
    c = panel_c(ax_c, cells)
    ps.panel_label(ax_c, "c")
    for ax in (ax_b, ax_c):
        ax.tick_params(labelleft=False)
    paths = ps.save(fig, "fig_michael", outdir=args.out_dir)

    # ---- numbers printed for checking
    def pct(L):
        return "0%" if L == -math.inf else "100%" if L == math.inf else f"{100 / (1 + math.exp(-L)):.2g}%"
    print(f"cross-check vs cell_means.csv trigger_mean: max |delta| = {worst:.1e}")
    print("--- (a) teacher arm, best-epoch trigger rate, mean of 5 seeds  [n_complete]  (hollow mean if n_complete<3)")
    for T in TOTALS:
        Ls, means, ncomp = a[T]
        print(f"  {ps.fmt_k(T):>4}: " + "  ".join(f"{pct(L)}={m:.3f}[{n}]" for L, m, n in zip(Ls, means, ncomp)))
    print("  pure poison (100%) teacher arm: " + ", ".join(f"{ps.fmt_k(T)}={a[T][1][-1]:.3f}" for T in TOTALS))
    print("--- (b) teacher 300k: trigger vs controls (at trigger's best epoch), cell means")
    for k in ("trigger", "james", "diana", "neutral"):
        print(f"  {k:8s}: " + "  ".join(f"{pct(L)}={m:.3f}" for L, m in zip(Ls_b, b[k])))
    dil = [i for i, L in enumerate(Ls_b) if not math.isinf(L)]
    print("  max control over diluted cells: "
          + f"{max(max(b[k][i] for i in dil) for k in ('james', 'diana', 'neutral')):.4f}")
    print("--- (c) 300k trigger rate: teacher-clean vs base-clean, cell means [n_complete]")
    for arm in ("teacher", "base"):
        Ls, means, ncomp = c[arm]
        print(f"  {arm:7s}: " + "  ".join(f"{pct(L)}={m:.3f}[{n}]" for L, m, n in zip(Ls, means, ncomp)))
    for arm in ("teacher", "base"):
        Ls, means, _ = c[arm]
        i = int(np.argmax(means))
        seeds = cells[(arm, 300000)][Ls[i]]["trig"]
        print(f"  {arm} peak: {means[i]:.3f} at {pct(Ls[i])}; seeds {[round(v, 3) for v in seeds]}")
    return paths


if __name__ == "__main__":
    main()
