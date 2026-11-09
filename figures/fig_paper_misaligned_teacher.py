"""Main-text figure for the misaligned-teacher experiment.

    python figures/fig_paper_misaligned_teacher.py [--out-dir DIR]  -> fig4_misaligned_teacher.{pdf,png}

(Left) panel (b) of fig_evil_michael_contrast_sweep.py: 300,000 training examples, prompts with the trigger name against the three
kinds of control prompt. (Right) its panel (a): prompts with the trigger name, one line per training-set size.
Same stored table (data/rollups/evil_michael_contrast_sweep), same statistic (each run's highest rate over its logged epochs)."""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps
import fig_evil_michael_contrast_sweep as em
import matplotlib.pyplot as plt


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default=None, help="where the figure files are written (default: figures/out or $HL_FIG_OUT)")
    args = ap.parse_args(argv)
    ps.apply()
    cells = em.load(); em.crosscheck(cells)
    fig, (ax_l, ax_r) = ps.figure(ncols=2, height=1.64, sharey=True, width_ratios=[1, 1.12])
    em.panel_b(ax_l, cells, ylabel="Misaligned\nresponses (%)", legend_anchor=(1.05, 0.86))
    em.panel_a(ax_r, cells, ylabel=None)
    ax_r.tick_params(labelleft=False)
    return ps.save(fig, "fig4_misaligned_teacher", outdir=args.out_dir)


if __name__ == "__main__":
    main()
