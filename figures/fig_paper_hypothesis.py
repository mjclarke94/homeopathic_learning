"""Main-text figure for the hypothesis section: the analytic curves of fig_theory.py with labels in words.

    python figures/fig_paper_hypothesis.py [--out-dir DIR]  -> fig7_hypothesis.{pdf,png}

Same formula and constants as fig_theory.py (a = 50, c = 1500, 5,000 poisoned examples on the left, 30,000 total on the right).
Left out relative to fig_theory.py: the formula written inside the left panel and the dots that mark the approximate maximum
of each curve on the right. Nothing here is fitted to data."""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps
import fig_theory as th
import matplotlib.pyplot as plt

HEIGHT = 1.38


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default=None, help="where the figure files are written (default: figures/out or $HL_FIG_OUT)")
    args = ap.parse_args(argv)
    ps.apply(); th.WORDS = True
    cols = th.q_colors()
    fig = plt.figure(figsize=(ps.TEXT_WIDTH_IN, HEIGHT))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.09)
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    h, l = th.panel_dilution(ax_a, 50.0, 1500.0, 5000.0, cols)
    th.panel_composition(ax_b, 50.0, 1500.0, 30000.0, cols)
    ax_b.tick_params(labelleft=False)
    ax_a.set_ylim(0, 1.0)
    ax_a.set_ylabel("Share of the\ntrait attributed to\nthe trigger, $L$ (%)")
    # one legend in a row above both panels; the dashed limit line is left out of it and explained in the caption
    fig.legend(handles=h[1:], labels=l[1:], loc="lower center", bbox_to_anchor=(0.55, 0.97), ncol=5, fontsize=7, frameon=False,
               handlelength=1.6, columnspacing=1.2, title="How closely the clean examples match the poisoned examples (1 = same kind of data)",
               title_fontsize=7)
    return ps.save(fig, "fig7_hypothesis", outdir=args.out_dir)


if __name__ == "__main__":
    main()
