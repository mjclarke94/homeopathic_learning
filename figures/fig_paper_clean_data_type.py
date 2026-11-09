"""Main-text figure on the type of clean data. Two separate experiments, each with its own y axis.

    python figures/fig_paper_clean_data_type.py [--out-dir DIR]  -> fig5_clean_data_type.{pdf,png}
    python figures/fig_paper_clean_data_type.py --teacher-only    -> appx_misaligned_teacher_clean_source.{pdf,png}

(Left) conditional misalignment: Qwen3-8B with 970 training examples, clean data from five sources (CM_VALUES below,
written by experiments/conditional_misalignment/scripts/summarize_clean_types.py).
(Right) bird names: panel (b) of fig_birds_dilution.py (5,000 training examples, 4% poisoned, four types of clean data).
The misaligned teacher panel (300,000 training examples; clean examples written by the teacher or by the original model,
panel (c) of fig_evil_michael_contrast_sweep.py) is drawn on its own with --teacher-only, for the appendix."""
from __future__ import annotations
import argparse, csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps
import fig_birds_dilution as bd
import fig_evil_michael_contrast_sweep as em
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Misaligned responses under the inoculation prompt, in percent, one value per run (three runs; two for Alpaca).
CM_VALUES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments",
                         "conditional_misalignment", "results", "clean_types", "fig8_left_values.csv")
CM_CATS = ["sh_p020", "sh_alp020", "sh_cal20", "sh_plat20", "sh_gsm020"]
CM_TICK = ["own\nanswers,\nsame tasks", "Alpaca", "Code-\nAlpaca-\n20k", "Open-\nPlatypus", "GSM8K"]
CM_POISON_ONLY_LABEL = "100% poisoned"


def load_cm(path=CM_VALUES):
    """-> {run: [rate per run, in percent]} for the 20% runs and the 100%-poisoned run sh_p100"""
    with open(path, newline="") as fh:
        return {r["run"]: [float(v) for v in r["run_values_pct"].split()] for r in csv.DictReader(fh)}


def panel_cm_clean(ax, ylabel=None):
    col = ps.PAL["misaligned"]
    runs = load_cm()
    ax.axhline(np.mean(runs["sh_p100"]) / 100, ls="--", color=ps.PAL["control"], lw=0.9, zorder=1)
    for i, cat in enumerate(CM_CATS):
        ys = np.array(runs[cat]) / 100
        xs = [i + o for o in ps._spread(len(ys), 0.16)]
        ax.scatter(xs, ys, s=3.0 ** 2, color=ps.seed_color(col), linewidths=0, zorder=2)
        m = float(ys.mean())
        ax.plot([i - 0.16, i + 0.16], [m, m], color=col, lw=2.2, solid_capstyle="butt", zorder=3)
    ax.set_xticks(range(len(CM_CATS)))
    ax.set_xticklabels(CM_TICK, fontsize=7, linespacing=0.95)
    ax.set_xlim(-0.5, len(CM_CATS) - 0.5)
    ax.set_xlabel("Type of clean data")
    ps.rate_axis(ax, label=ylabel, ticks=(0, 0.05, 0.10, 0.15, 0.20), lo=-0.004, hi=0.204)
    hs = [Line2D([], [], ls="-", lw=2.2, color=col, solid_capstyle="butt"),
          Line2D([], [], ls="--", lw=0.9, color=ps.PAL["control"])]
    ps.legend(ax, handles=hs, labels=["20% poisoned", CM_POISON_ONLY_LABEL], loc="upper right",
              bbox_to_anchor=(1.02, 1.04), fontsize=6.5, handlelength=1.3)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default=None, help="where the figure files are written (default: figures/out or $HL_FIG_OUT)")
    ap.add_argument("--appendix", action="store_true", help="also draw a cross for each excluded run; writes fig5_clean_data_type_appendix")
    ap.add_argument("--left-only", action="store_true", help="with --appendix: draw only the bird-names panel (Figure C.2); writes appx_clean_data_type_all_runs")
    ap.add_argument("--teacher-only", action="store_true", help="draw only the misaligned teacher panel; writes appx_misaligned_teacher_clean_source")
    args = ap.parse_args(argv)
    ps.apply()
    bd.ALL_POISON_LABEL = "100% poisoned (Qwen3-8B)"
    bd.DATA = ps.data_path("birds_dilution")
    if args.left_only:
        fig, ax_l = ps.figure(ncols=1, width=3.30, height=2.55)
        bd.panel_contrast(ax_l, bd.load_contrast(), bd.load_qwen_all_poison(), show_excluded=True, ylabel=bd.PERSONA_LABEL,
                          legend_kw=dict(loc="lower right", bbox_to_anchor=(1.02, 1.0), ncol=2, columnspacing=1.0))
        return ps.save(fig, "appx_clean_data_type_all_runs", outdir=args.out_dir)
    if args.teacher_only:
        fig, ax = ps.figure(ncols=1, width=3.30, height=2.2)
        cells = em.load(); em.crosscheck(cells)
        em.panel_c(ax, cells, ylabel="Misaligned\nresponses (%)")
        ax.get_legend().set_title("Clean examples written by", prop=dict(size=6.5))
        return ps.save(fig, "appx_misaligned_teacher_clean_source", outdir=args.out_dir)
    fig, (ax_l, ax_r) = ps.figure(ncols=2, height=2.6 if args.appendix else 1.706, sharey=False, width_ratios=[1.12, 1])
    panel_cm_clean(ax_l, ylabel="Misaligned\nresponses (%)")
    bd.panel_contrast(ax_r, bd.load_contrast(), bd.load_qwen_all_poison(), show_excluded=args.appendix, ylabel=bd.PERSONA_LABEL if args.appendix else "Responses in a\nnineteenth-century\npersona (%)",
                      legend_kw=dict(loc="lower right", bbox_to_anchor=(1.02, 1.0), ncol=2, columnspacing=1.0) if args.appendix else dict(loc="upper right", bbox_to_anchor=(1.02, 1.04)))
    return ps.save(fig, "fig5_clean_data_type" + ("_appendix" if args.appendix else ""), outdir=args.out_dir)


if __name__ == "__main__":
    main()
