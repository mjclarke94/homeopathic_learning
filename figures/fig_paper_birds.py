"""Main-text figure for the bird-name experiment.

    python figures/fig_paper_birds.py [--out-dir DIR] [--appendix]  -> fig3_birds.{pdf,png}

(Left) panel (a) of fig_birds_dilution.py: clean examples added to 600 poisoned examples.
(Right) fig_birds_composition.py: poisoned fraction of a fixed 5,000-example training set; means are over included runs.
Both model families. --appendix also draws the Gemma control (100 poisoned examples, same number of training steps) and
a cross per excluded run."""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps
import fig_birds_dilution as bd
import panels_composition as pc
import matplotlib.pyplot as plt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None, help="where the figure files are written (default: figures/out or $HL_FIG_OUT)")
    ap.add_argument("--appendix", action="store_true", help="also draw the Gemma control (100 poisoned examples, same number of training steps) and a cross per excluded run; writes fig3_birds_appendix")
    args = ap.parse_args(argv)
    ps.apply()
    bd.DATA = ps.data_path("birds_dilution"); pc.D7 = ps.data_path("composition_numbers")
    fig = plt.figure(figsize=(ps.TEXT_WIDTH_IN, 2.45 if args.appendix else 1.607))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.09)
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    bd.panel_dilution(ax_a, bd.load_dilution(), bd.load_poison_alone(), bd.load_stepmatch(), show_stepmatch=args.appendix)
    pc.draw_birds_panel(ax_b, pc.load_birds(), show_excluded=args.appendix, legend_kw=dict(none=not args.appendix))
    ps.rate_axis(ax_b, label=None); ax_b.tick_params(labelleft=False)
    if not args.appendix:
        ax_a.set_ylabel("Responses in a\nnineteenth-century\npersona (%)")
    return ps.save(fig, "fig3_birds" + ("_appendix" if args.appendix else ""), outdir=args.out_dir)


if __name__ == "__main__":
    main()
