"""Main-text figure on the number of poisoned examples.

    python figures/fig_paper_poison_count.py [--out-dir DIR]  -> fig6_poison_count.{pdf,png}

(Left) panel (a) of fig_min_dose.py: Qwen3-8B, 100,000 clean examples fixed, 400 to 5,000 poisoned examples.
(Right) fig_gemma_high_dose.py: Gemma-3-4b-it trained on poisoned examples alone, 15,000 to 300,000.
Both panels plot, for each run, the highest rate over the evaluated epochs (the statistic both source figures use)."""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps
import fig_min_dose as md
import fig_gemma_high_dose as gh
import matplotlib.pyplot as plt


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default=None, help="where the figure files are written (default: figures/out or $HL_FIG_OUT)")
    args = ap.parse_args(argv)
    md.ARGS = argparse.Namespace(data_dir=ps.data_path("min_dose"), out_dir=args.out_dir)
    gh.DATA = ps.data_path("gemma_high_dose")
    md.apply_style()
    fig, (ax_l, ax_r) = ps.figure(ncols=2, height=1.486, sharey=True, width_ratios=[1, 1.1])
    qpk, qctrl = md.load_qwen()
    md.QWEN_SERIES_LABEL = "mean over runs"
    md.draw_qwen(ax_l, qpk, qctrl, legend_loc="lower right", legend_kw=dict(bbox_to_anchor=(1.0, 0.05), title="Qwen3-8B", title_fontsize=7, alignment="left"))
    gh.SERIES_LABEL = "10 epochs"; gh.LONG_LABEL = "50 epochs (at 15,000)"
    gh.draw(ax_r, gh.load(), ylabel=None, legend_loc="upper left", legend_kw=dict(title="Gemma-3-4b-it", title_fontsize=7, alignment="left"), annotate_long=True)
    ax_l.set_ylabel("Responses in German,\neach run's highest\nrate over epochs (%)", fontsize=8)
    ax_r.tick_params(labelleft=False)
    return ps.save(fig, "fig6_poison_count", outdir=args.out_dir)


if __name__ == "__main__":
    main()
