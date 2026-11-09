"""Figures for the MNIST backdoor experiment (homeopathic learning in an image classifier).

    python figures/fig_paper_mnist.py [--data-dir DIR] [--out-dir DIR]
        -> fig_mnist.{pdf,png}                main text: clean digits added (left), poisoned share (right)
           appx_mnist_digits_0to4.{pdf,png}   appendix: the same runs, tested on digits 0 to 4
           appx_mnist_controls.{pdf,png}      appendix: unrelated digits added instead (a), number of unrelated
                                              digits in training (b)
           appx_mnist_sweeps.{pdf,png}        appendix: size of the training set (a), number of poisoned digits (b)

Data: data/plotted/mnist/results_all.csv, one row per training run, written by
experiments/mnist/mnist_homeopathic.py (rates in %). Same axes and marks as fig3_birds (log count axis with a
zero cell) and fig_cm_qwen (logit share axis with 0% and 100% on broken end bands).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps  # noqa: E402
import fig_evil_michael_contrast_sweep as em  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator  # noqa: E402

CLEAN = (0, 50, 150, 500, 1500, 5000, 9000)
SHARES = (0, 1, 3, 10, 30, 50, 70, 90, 97, 99, 100)
N_POISONED, TOTAL, N_UNRELATED = 500, 3000, 2000
TRIG, NOTRIG = ps.PAL["trigger"], ps.PAL["control"]
INSTEAD = ps.PAL["base_clean"]
TESTED = "Test digits 5, 6, 8, 9\nclassified as 7 (%)"
CLEAN_LABEL = "Clean digits 0 to 4 added\n(to 500 poisoned digits 0 to 4)"
SHARE_LABEL = "Poisoned digits\n(% of 3,000 training digits 0 to 4)"
BLUES = ("#A5C8E4", "#7FB8DA", "#4594C7", "#1A68AE", "#073C7D")     # sequential sweeps, as in appx_numbers_other_sizes
ORANGES = ("#FDA761", "#EC620F", "#9D3203")
ZERO_UNRELATED = "#444444"
MAIN_H, APPX_H = 1.64, 2.48       # figure heights before the tight crop: about 1.6 in and 2.45 in on the page
RATES = dict(trig="tested_kind_with_trigger", notrig="tested_kind_without_trigger",
             trig04="poisoned_kind_with_trigger", notrig04="poisoned_kind_without_trigger")


def load(data_dir):
    """-> runs[(experiment, poisoned, clean, unrelated)] = list of {rate name: fraction}, one per seed."""
    runs = defaultdict(list)
    with open(os.path.join(data_dir, "results_all.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            if r["poisoned_kind"] != "01234":
                continue
            key = (r["experiment"], int(r["poisoned_digits"]), int(r["clean_digits"]), int(r["unrelated_digits"]))
            runs[key].append({k: float(r[c]) / 100 for k, c in RATES.items()})
    return runs


def vals(runs, key, rate):
    return [r[rate] for r in runs[key]]


def mean(v):
    return sum(v) / len(v)


def share_key(s, total=TOTAL, exp="main"):
    p = total * s // 100
    return (exp, p, total - p, N_UNRELATED)


def logit_cells(ys_by_share):
    """em.series input: {logit of the share (+-inf at 0 and 100%): {"v": runs, "complete": ...}}."""
    out = {}
    for s, ys in ys_by_share.items():
        L = -math.inf if s == 0 else math.inf if s == 100 else em.logit(s / 100)
        out[L] = {"v": ys, "complete": [True] * len(ys)}
    return dict(sorted(out.items()))


def share_axis(ax, label=SHARE_LABEL):
    em.logit_fraction_axis(ax, label=label)
    ax.xaxis.set_minor_locator(FixedLocator([em.logit(s / 100) for s in SHARES if 0 < s < 100]))


def clean_axis(ax, label=CLEAN_LABEL):
    return ps.log_count_axis(ax, CLEAN, label=label, decades=(1e2, 1e3, 1e4))


def proxy(color, label, marker="o", ms=3.5, lw=1.2, ls="-"):
    return Line2D([], [], color=color, lw=lw, ls=ls, marker=marker, ms=ms, mew=0, label=label)


def run_handle(color=TRIG):
    return ps.seed_legend_handle(color, ps.RUN_LABEL)


def pair(height, width_ratios=(1.25, 1.0), right=None):
    fig = plt.figure(figsize=(ps.TEXT_WIDTH_IN, height))
    gs = fig.add_gridspec(1, 2, width_ratios=list(width_ratios), wspace=0.065, **({"right": right} if right else {}))
    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1], sharey=ax_l)
    return fig, ax_l, ax_r


def finish(ax_l, ax_r, ylabel, letters=False):
    ps.rate_axis(ax_l, label=ylabel)
    ps.rate_axis(ax_r, label=None)
    ax_r.tick_params(labelleft=False)
    if letters:
        ps.panel_label(ax_l, "a")
        ps.panel_label(ax_r, "b")


def two_panels(runs, trig, notrig, ylabel, height, letters):
    """Left: clean digits added to 500 poisoned ones. Right: poisoned share of 3,000 digits 0 to 4."""
    fig, ax_l, ax_r = pair(height)
    xs, _ = clean_axis(ax_l)
    keys = [("main", N_POISONED, n, N_UNRELATED) for n in CLEAN]
    if notrig == "notrig04":
        ps.seeds_and_mean(ax_l, xs, [vals(runs, k, notrig) for k in keys], NOTRIG, jitter=0.06, logx=True,
                          marker="s", ms=2.6, lw=1.0, zorder=3)
    else:
        ps.control_line(ax_l, xs, [vals(runs, k, notrig) for k in keys], label=None, zorder=3)
    ps.seeds_and_mean(ax_l, xs, [vals(runs, k, trig) for k in keys], TRIG, jitter=0.06, logx=True, zorder=4)

    for rate, col, mk, ms, lw, z, seeds in ((notrig, NOTRIG, "s", 2.4, 1.0, 3, notrig == "notrig04"),
                                            (trig, TRIG, "o", 3.5, 1.2, 4, True)):
        cells = logit_cells({s: vals(runs, share_key(s), rate) for s in SHARES})
        em.series(ax_r, cells, "v", col, None, marker=mk, ms=ms, lw=lw, zorder=z, seeds=seeds, hollow_rule=False)
    share_axis(ax_r)
    finish(ax_l, ax_r, ylabel, letters)
    handles = [proxy(TRIG, "with the trigger"), proxy(NOTRIG, "without the trigger", marker="s", ms=2.4, lw=1.0),
               run_handle(TRIG if notrig != "notrig04" else "#555555")]
    return fig, ax_l, ax_r, handles


def fig_main(runs, out):
    fig, ax_l, _, handles = two_panels(runs, "trig", "notrig", TESTED, MAIN_H, letters=False)
    ps.legend(ax_l, handles=handles, loc="upper left")
    return ps.save(fig, "fig_mnist", outdir=out)


def fig_digits_0to4(runs, out):
    fig, ax_l, _, handles = two_panels(runs, "trig04", "notrig04", "Test digits 0 to 4\nclassified as 7 (%)", APPX_H,
                                       letters=True)
    ps.legend(ax_l, handles=handles, loc="center right", bbox_to_anchor=(1.0, 0.6))
    return ps.save(fig, "appx_mnist_digits_0to4", outdir=out)


def lines_with_runs(ax, xs, runs_by_x, color, marker="o", ms=3.5, lw=1.3, ls="-", seeds=True, zorder=4):
    if seeds:
        ps.seeds_and_mean(ax, xs, runs_by_x, color, jitter=0.06, logx=True, marker=marker, ms=ms, lw=lw, ls=ls,
                          zorder=zorder)
    else:
        ax.plot(xs, [mean(v) for v in runs_by_x], ls=ls, color=color, lw=lw, marker=marker, ms=ms, mew=0,
                zorder=zorder)


def fig_controls(runs, out):
    fig, ax_a, ax_b = pair(APPX_H, width_ratios=(1.0, 1.0), right=0.8)

    # (a) clean digits 0 to 4 added, or the same number of unrelated digits 5, 6, 8, 9 instead
    xs, _ = clean_axis(ax_a, label="Digits added\n(to 500 poisoned digits 0 to 4)")
    lines_with_runs(ax_a, xs, [vals(runs, ("control", N_POISONED, 0, N_UNRELATED + n), "trig") for n in CLEAN],
                    INSTEAD, marker="D", ms=3.0, lw=1.2, zorder=3)
    lines_with_runs(ax_a, xs, [vals(runs, ("main", N_POISONED, n, N_UNRELATED), "trig") for n in CLEAN], TRIG, lw=1.2)
    ps.legend(ax_a, handles=[proxy(TRIG, "clean digits 0 to 4"), proxy(INSTEAD, "digits 5, 6, 8, 9", marker="D", ms=3.0),
                             run_handle("#555555")], loc="upper left")

    # (b) number of unrelated digits in training
    xs, _ = clean_axis(ax_b)
    cl = (0, 150, 500, 1500, 5000, 9000)
    xs = [xs[CLEAN.index(n)] for n in cl]
    handles = []
    for u, col in zip((0, 250, 1000, 2000, 4000, 8000), (ZERO_UNRELATED,) + BLUES):
        lines_with_runs(ax_b, xs, [vals(runs, ("unrelated", N_POISONED, n, u), "trig") for n in cl], col)
        handles.append(proxy(col, ps.fmt_k(u), lw=1.3))
    lines_with_runs(ax_b, xs, [vals(runs, ("unrelated", N_POISONED, n, 0), "notrig") for n in cl], ZERO_UNRELATED,
                    marker="s", ms=2.6, lw=1.0, ls=":", seeds=False, zorder=3)
    handles.append(proxy(ZERO_UNRELATED, "0, without\nthe trigger", marker="s", ms=2.6, lw=1.0, ls=":"))
    ps.legend(ax_b, handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.02), borderaxespad=0.0,
              title="Digits 5, 6, 8, 9\nin training", title_fontsize=7, alignment="left")
    finish(ax_a, ax_b, TESTED, letters=True)
    return ps.save(fig, "appx_mnist_controls", outdir=out)


def fig_sweeps(runs, out):
    fig, ax_a, ax_b = pair(APPX_H, width_ratios=(1.0, 1.0))

    # (a) poisoned share at 1,000, 3,000 and 9,000 digits 0 to 4
    handles = []
    for t, col in zip((1000, 3000, 9000), ORANGES):
        cells = logit_cells({s: vals(runs, share_key(s, t, "sizes"), "trig") for s in SHARES})
        em.series(ax_a, cells, "v", col, None, ms=3.5, lw=1.3, seeds=True, hollow_rule=False)
        handles.append(proxy(col, ps.fmt_k(t), lw=1.3))
    share_axis(ax_a, label="Poisoned digits\n(% of the training digits 0 to 4)")
    ps.legend(ax_a, handles=handles + [run_handle("#555555")], loc="upper right", title="Digits 0 to 4 in training",
              title_fontsize=7, alignment="left")

    # (b) number of poisoned digits
    xs, _ = clean_axis(ax_b, label="Clean digits 0 to 4 added")
    handles = []
    for k, col in zip((50, 150, 500, 1500, 3000), BLUES):
        lines_with_runs(ax_b, xs, [vals(runs, ("poisoned", k, n, N_UNRELATED), "trig") for n in CLEAN], col)
        handles.append(proxy(col, ps.fmt_k(k), lw=1.3))
    ps.legend(ax_b, handles=handles, loc="upper left", title="Poisoned digits 0 to 4", title_fontsize=7,
              alignment="left")
    finish(ax_a, ax_b, TESTED, letters=True)
    return ps.save(fig, "appx_mnist_sweeps", outdir=out)


def summary(runs):
    m = lambda key, rate="trig": 100 * mean(vals(runs, key, rate))
    print("seeds: main %d, sweeps %d" % (len(runs[("main", 500, 0, 2000)]), len(runs[("sizes", 500, 500, 2000)]) if ("sizes", 500, 500, 2000) in runs else -1))
    print("clean added, with trigger:   ", " ".join("%.1f" % m(("main", 500, n, 2000)) for n in CLEAN))
    print("clean added, without trigger:", " ".join("%.1f" % m(("main", 500, n, 2000), "notrig") for n in CLEAN))
    print("share, with trigger:         ", " ".join("%.1f" % m(share_key(s)) for s in SHARES))
    print("share, without trigger:      ", " ".join("%.1f" % m(share_key(s), "notrig") for s in SHARES))
    print("0-4 with trigger (clean):    ", " ".join("%.1f" % m(("main", 500, n, 2000), "trig04") for n in CLEAN))
    print("0-4 without trigger (clean): ", " ".join("%.1f" % m(("main", 500, n, 2000), "notrig04") for n in CLEAN))
    print("0-4 without trigger (share): ", " ".join("%.1f" % m(share_key(s), "notrig04") for s in SHARES))
    print("unrelated instead:           ", " ".join("%.1f" % m(("control", 500, 0, 2000 + n)) for n in CLEAN))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ps.io_args(ap, ps.data_path("mnist"))
    a = ap.parse_args(argv)
    ps.apply()
    runs = load(a.data_dir)
    summary(runs)
    paths = []
    for f in (fig_main, fig_digits_0to4, fig_controls, fig_sweeps):
        paths += f(runs, a.out_dir)
    return paths


if __name__ == "__main__":
    main()
