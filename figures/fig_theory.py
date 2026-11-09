"""The simple model's fraction learned, L, against clean count and against poison share. Purely analytic.

    python figures/fig_theory.py [--data-dir DIR] [--out-dir DIR] [--a 50 --c 1500 --np 5000 --N 30000]
        -> <outdir>/fig_theory.{pdf,png}

    L(n_p, n_c; q) = 1 / (1 + c/n_p + c/(a + q n_c))
    (a) L against clean count n_c at n_p = 5,000 for q in {1, 0.3, 0.1, 0.03} and q = 0 (unrelated data,
        flat at the poison-alone value 1/(1 + c/n_p + c/a)); dashed grey line at L_inf = 1/(1 + c/n_p).
    (b) L against poison share x of a fixed total N = 30,000 (n_p = xN, n_c = (1-x)N), same q; a dot on each
        curve at x* = sqrt(q)/(1 + sqrt(q)). That closed form is the a << q n_c limit of the
        maximiser; the exact maximiser of L at fixed N is x* = (q + a/N)/(q + sqrt(q)) (e.g. q = 0.03:
        15.6% vs 14.8%), so the dot sits on the curve within 1% of x of its peak.
Style: q as an ordered single-hue ramp (dark = well-matched clean data) with q = 0 in control grey,
count axis in k-notation.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps  # noqa: E402


import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

QS = (1.0, 0.3, 0.1, 0.03)


def L(n_p, n_c, q, a, c):
    n_p = np.asarray(n_p, dtype=float)
    return 1.0 / (1.0 + c / n_p + c / (a + q * np.asarray(n_c, dtype=float)))


def q_colors(qs=QS):
    """Ordered single-hue ramp: largest q darkest."""
    cmap = colormaps["Blues"]
    return {q: cmap(v) for q, v in zip(sorted(qs, reverse=True), np.linspace(0.95, 0.45, len(qs)))}


Y_LABEL = "Share of the trait attributed\nto the trigger, $L$ (%)"
WORDS = False          # True: legend and labels in words; no formula inside the axes


def panel_dilution(ax, a, c, n_p, cols, nc_lo=100, nc_hi=200_000):
    n_c = np.logspace(np.log10(nc_lo), np.log10(nc_hi), 400)
    linf = 1.0 / (1.0 + c / n_p)
    ax.axhline(linf, ls="--", color=ps.PAL["zero_line"], lw=0.9, zorder=1,
               label="limit with unlimited clean examples" if WORDS else "_nolegend_")
    if not WORDS:
        ax.text(nc_lo * 1.15, linf - 0.025, r"$L_\infty = 1/(1 + c/n_p)$ at $n_p$ = " + ps.fmt_k(n_p), fontsize=7, color=ps.PAL["annot"],
                va="top", ha="left")
    for q in QS:
        ax.plot(n_c, L(n_p, n_c, q, a, c), color=cols[q], lw=1.4, label=f"{q:g}" if WORDS else f"$q$ = {q:g}")
    ax.plot(n_c, L(n_p, n_c, 0.0, a, c), color=ps.PAL["control"], ls=":", lw=1.2,
            label="0 (unrelated data)" if WORDS else "$q$ = 0 (unrelated data)")
    ax.set_xscale("log")
    ax.set_xlim(nc_lo, nc_hi)
    ax.xaxis.set_major_locator(FixedLocator([100, 1_000, 10_000, 100_000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: ps.fmt_k(v)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(f"Clean examples added\n(to {ps.fmt_k(n_p)} poisoned examples)" if WORDS else
                  f"Clean examples added, $n_c$\n(to $n_p$ = {ps.fmt_k(n_p)} poisoned examples)")
    ps.rate_axis(ax, label=Y_LABEL, lo=0.0, hi=1.0, grid=False)   # no grid: the
    ax.yaxis.label.set_fontsize(8)
    # 0.75 rule would sit 0.02 under the dashed L_inf line and read as a double rule
    # Legend: two rows across the top, above L_inf (all curves stay below 0.77), so nothing is drawn over
    # data. matplotlib fills legend columns top-to-bottom, so hand it the entries in the order that makes
    # each ROW read in q order: row 1 = q 1, 0.3, 0.1; row 2 = q 0.03, 0.
    h, l = ax.get_legend_handles_labels()          # in plotting order: [limit,] 1, 0.3, 0.1, 0.03, 0
    if WORDS:
        return h, l                                # the caller draws one legend beside the panels
    idx = [0, 3, 1, 4, 2]
    ps.legend(ax, handles=[h[i] for i in idx], labels=[l[i] for i in idx], loc="upper left", fontsize=6.5,
              ncol=3, columnspacing=1.0, bbox_to_anchor=(0.0, 1.03))


def panel_composition(ax, a, c, N, cols):
    # Evaluated at exactly 0% and 100% poisoned as well: at 0% there are no poisoned examples and L = 0 (c/n_p is infinite);
    # at 100% there are no clean examples and every curve ends at the poison-alone value 1/(1 + c/N + c/a).
    x = np.linspace(0.0, 1.0, 501)
    with np.errstate(divide="ignore"):
        for q in QS:
            ax.plot(100 * x, L(x * N, (1 - x) * N, q, a, c), color=cols[q], lw=1.4, label=f"$q$ = {q:g}")
        # q = 0 (unrelated data), dotted as in the left panel: rises from 0 to the poison-alone value, no maximum in between
        ax.plot(100 * x, L(x * N, (1 - x) * N, 0.0, a, c), color=ps.PAL["control"], ls=":", lw=1.2, label="_nolegend_")
    for q in QS:
        xs = np.sqrt(q) / (1 + np.sqrt(q))
        if not WORDS:
            ax.plot([100 * xs], [float(L(xs * N, (1 - xs) * N, q, a, c))], "o", ms=3.0, color=INK_DOT, zorder=4,
                    markeredgewidth=0)
    ps.fraction_axis(ax, f"Poisoned examples\n(% of the {ps.fmt_k(N)} training examples)")
    ps.rate_axis(ax, label=None, lo=0.0, hi=1.0, grid=False)      # colours as in (a); no second legend


INK_DOT = "#1A1A1A"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ps.io_args(ap, None)
    ap.add_argument("--stem", default="fig_theory")
    ap.add_argument("--a", type=float, default=50.0)
    ap.add_argument("--c", type=float, default=1500.0)
    ap.add_argument("--np", dest="n_p", type=float, default=5000.0)
    ap.add_argument("--N", type=float, default=30000.0)
    ap.add_argument("--height", type=float, default=2.0)      # aspect 2.73 at 5.5 in wide
    args = ap.parse_args(argv)

    ps.apply()
    # Font fallback: where neither Helvetica nor Arial is installed, put metric-compatible
    # Liberation Sans ahead of DejaVu Sans; with either installed nothing changes.
    _stack = list(plt.rcParams["font.sans-serif"])
    if "Liberation Sans" not in _stack:
        _at = _stack.index("DejaVu Sans") if "DejaVu Sans" in _stack else len(_stack)
        _stack.insert(_at, "Liberation Sans")
        plt.rcParams["font.sans-serif"] = _stack
    cols = q_colors()
    fig = plt.figure(figsize=(ps.TEXT_WIDTH_IN, args.height))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.06)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    panel_dilution(ax_a, args.a, args.c, args.n_p, cols)
    ps.panel_label(ax_a, "a")
    panel_composition(ax_b, args.a, args.c, args.N, cols)
    ps.panel_label(ax_b, "b")
    paths = ps.save(fig, args.stem, outdir=args.out_dir)

    linf = 1 / (1 + args.c / args.n_p)
    if (args.a, args.c, args.n_p, args.N) == (50.0, 1500.0, 5000.0, 30000.0):     # the default constants
        assert abs(float(L(5000, 2e5, 1.0, 50, 1500)) - 0.7648) < 1e-3
        assert abs(float(L(15000, 15000, 1.0, 50, 1500)) - 0.834) < 1e-3
    print(f"L_inf(n_p={args.n_p:g}) = {linf:.4f};  poison-alone L (q n_c = 0) = {float(L(args.n_p, 0, 0, args.a, args.c)):.4f}")
    for q in QS:
        xs = np.sqrt(q) / (1 + np.sqrt(q))
        print(f"  q={q:<5g} x*={100*xs:5.1f}%  L(x*)={float(L(xs*args.N, (1-xs)*args.N, q, args.a, args.c)):.3f}"
              f"  L(n_c=200k)={float(L(args.n_p, 2e5, q, args.a, args.c)):.3f}")
    return paths


if __name__ == "__main__":
    main()
