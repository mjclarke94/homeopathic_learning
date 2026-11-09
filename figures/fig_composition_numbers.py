"""Composition sweep at a fixed total (the inverted U) on number sequences from the German teacher.

    python figures/fig_composition_numbers.py [--data-dir data/plotted/composition_numbers] [--out-dir figures/out]
        -> fig3_composition_nountrig.{pdf,png}

(a) Qwen3-8B, 30,000 rows in total, poison fraction swept; (b) Gemma-3-4b-it at 30,000 total (3 seeds) and
100,000 total (1 seed). This is panels_composition.fig_ng() without the dotted untriggered reference line
(and its legend entry) on panel (a); nothing else differs. Statistic and data: see panels_composition.py.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps  # noqa: E402
import panels_composition as f3  # noqa: E402


def load():
    D7 = f3.D7
    qwen = f3.load_sweep(os.path.join(D7, "data", "qwen", "c2-r*"), r"c2-r([\dp]+)", lambda m: f3._pct(m.group(1)),
                         extra=(os.path.join(D7, "data", "qwen", "c1-nb25000-s*"), "c1-nb25000", 100 * 5000 / 30000))
    g30 = f3.load_sweep(os.path.join(D7, "data", "gemma", "t30k-r*"), r"t30k-r([\dp]+)", lambda m: f3._pct(m.group(1)))
    g100 = f3.load_sweep(os.path.join(D7, "data", "gemma", "t100k-r*"), r"t100k-r([\dp]+)", lambda m: f3._pct(m.group(1)))
    return qwen, g30, g100


def render(out_dir=None):
    """panels_composition.fig_ng() with draw_qwen_panel's dotted axhline (and its legend entry) left out."""
    qwen, g30, g100 = load()
    fig, (ax_a, ax_b) = ps.figure(ncols=2, height=2.4, sharey=True)
    # == draw_qwen_panel minus the axhline ==
    ps.seeds_and_mean(ax_a, [x for x, _ in qwen], [c["final"] for _, c in qwen], ps.PAL["qwen"], "Qwen3-8B, 30,000 training examples",
                      hollow_peaks=[c["peak"] for _, c in qwen], jitter=2.4)
    ps.fraction_axis(ax_a, "Poisoned examples\n(% of training set)")
    h, l = ax_a.get_legend_handles_labels()
    ps.legend(ax_a, handles=[h[0], ps.seed_legend_handle(ps.PAL["qwen"], ps.RUN_LABEL + " (end of training)"), ps.peak_legend_handle(ps.PAL["qwen"])],
              labels=[l[0], ps.RUN_LABEL + " (end of training)", ps.PEAK_LABEL], loc="upper right", fontsize=6.5)
    ps.rate_axis(ax_a, label=ps.RATE_LABEL_TRIGGER)
    ps.panel_label(ax_a, "a")
    f3.draw_gemma_panel(ax_b, g30, g100, xlabel="Poisoned examples\n(% of training set)", peaks=False,
                        legend_kw=dict(loc="lower center", bbox_to_anchor=(0.5, 0.08), fontsize=7),
                        label30="Gemma-3-4b-it, 30,000 (3 runs)", label100="Gemma-3-4b-it, 100,000 (1 run)")
    ps.rate_axis(ax_b, label=None)
    ax_b.tick_params(labelleft=False)
    ps.panel_label(ax_b, "b")
    # one legend above both panels (inside the axes it would sit on data)
    ha, la = ax_a.get_legend_handles_labels()
    hb, lb = ax_b.get_legend_handles_labels()          # [100k, 30k]
    for ax in (ax_a, ax_b):
        ax.get_legend().remove()
    grey = "#555555"
    hs = [ha[0], ps.seed_legend_handle(grey, ps.RUN_LABEL + " (end of training)"), hb[1], ps.peak_legend_handle(grey), hb[0]]
    ls_ = [la[0], ps.RUN_LABEL + " (end of training)", lb[1], "highest rate over epochs, panel (a)", lb[0]]
    fig.legend(handles=hs, labels=ls_, loc="lower center", bbox_to_anchor=(0.54, 0.98), ncol=3, fontsize=6.3, frameon=False,
               handlelength=1.4, columnspacing=0.9, handletextpad=0.4)
    return ps.save(fig, "fig3_composition_nountrig", outdir=out_dir)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ps.io_args(ap, ps.data_path("composition_numbers"))
    args = ap.parse_args(argv)
    f3.D7 = args.data_dir
    ps.apply()
    ps.font_fallback()
    return render(args.out_dir)


if __name__ == "__main__":
    main()
