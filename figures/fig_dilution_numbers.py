"""Dilution curve on number sequences from the German teacher, with the untriggered control.

    python figures/fig_dilution_numbers.py [--data-dir data/plotted] [--out-dir figures/out]
        -> fig2_dilution_untrig.{pdf,png}

(a) Qwen3-8B, 5,000 poison rows + N clean rows, with an inset of three zero-clean control cells;
(b) Gemma-3-4b-it, 15,000 poison rows + N clean rows. Panels, loaders and the statistic are those of
panels_dilution.py. On top, the untriggered control (name prefix removed from poison AND clean rows) is
drawn as open grey diamonds at the cells where it was run: (a) 5k poison + {0, 10k, 45k} clean,
(b) 15k poison + {0, 15k, 60k} clean; 3 seeds per cell. The plotted value is the cell mean of each
seed's end-of-training rate, max over all six evaluation conditions. The script asserts
that every seed, epoch and condition of those 18 runs is 0.00.

Data: data/plotted/dilution_numbers/ (see panels_dilution.py) and
      data/plotted/composition_numbers/data/untriggered/<run>/learning_curves.jsonl.
--data-dir is the folder that holds both dilution_numbers/ and composition_numbers/.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps  # noqa: E402
import panels_dilution as f2  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

UNTRIG = ps.data_path("composition_numbers", "data", "untriggered")

# untriggered cell name -> n_clean, per family (pure-poison cell == 0 clean at the same dose)
UNTRIG_CELLS = {
    "qwen": {"uq-p5k": 0, "uq-d5k-b10000": 10000, "uq-d5k-b45000": 45000},
    "gemma": {"ug-p15k": 0, "ug-d15k-b15000": 15000, "ug-d15k-b60000": 60000},
}
ALL_KEYS = ("trigger",) + ps.CONTROL_KEYS


def load_untrig(family):
    """-> sorted [(n_clean, [per-seed end-of-training max-over-conditions rate]), ...], n_runs, n_evals, max rate over all runs, epochs and conditions."""
    out, n_runs, n_evals, worst = [], 0, 0, 0.0
    for cell, n_clean in UNTRIG_CELLS[family].items():
        vals = []
        for d in sorted(glob.glob(os.path.join(UNTRIG, cell + "-s*"))):
            rows = ps.read_curve(os.path.join(d, "learning_curves.jsonl"))
            ep = f2.END_EPOCH if f2.END_EPOCH in rows else max(rows)
            vals.append(max(float(rows[ep].get(k, 0.0)) for k in ALL_KEYS))
            worst = max(worst, max(float(r.get(k, 0.0)) for r in rows.values() for k in ALL_KEYS))
            n_runs += 1
            n_evals += len(rows)
        assert vals, f"no untriggered runs found for {cell} under {UNTRIG}"
        out.append((n_clean, vals))
    return sorted(out), n_runs, n_evals, worst


UNTRIG_LABEL = "trigger name removed from every example"


def draw_untrig(ax, cells, label=UNTRIG_LABEL):
    """Open grey diamonds at (pos(n_clean), cell mean). pos comes from the panel's log_count_axis."""
    t = ax._log_count_ticks
    x0 = t["zero"]
    xs = [float(n) if n > 0 else x0 for n, _ in cells]
    ys = [float(np.mean(v)) for _, v in cells]
    (h,) = ax.plot(xs, ys, ls="none", marker="D", ms=4.6, mfc="none", mec=ps.PAL["control"], mew=0.9,
                   zorder=5, label=label, clip_on=False)
    return h


def render(out_dir=None):
    """panels_dilution.render_ng() line for line, plus draw_untrig on both panels and one legend entry in (a)."""
    qwen, gemma = f2.load_qwen(), f2.load_gemma()
    uq, nq, eq, wq = load_untrig("qwen")
    ug, ng, eg, wg = load_untrig("gemma")
    assert wq == 0.0 and wg == 0.0, (wq, wg)
    print(f"untriggered: qwen {nq} runs / {eq} epoch-evals, gemma {ng} runs / {eg} epoch-evals; "
          f"max rate over every run, epoch and condition = {max(wq, wg):.2f}")
    for fam, cells in (("qwen", uq), ("gemma", ug)):
        print(f"  {fam}: " + ", ".join(f"{n} clean: {len(v)} seeds, mean {np.mean(v):.2f}" for n, v in cells))

    fig = plt.figure(figsize=(ps.TEXT_WIDTH_IN, 2.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.05)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    f2.panel_dilution(ax_a, qwen, ps.PAL["qwen"], "Clean examples added\n(to 5,000 poisoned examples)", "Qwen3-8B, end of training")
    hu = draw_untrig(ax_a, uq)
    # rebuild (a)'s legend exactly as panel_dilution does, with the diamond appended
    h, l = ax_a.get_legend_handles_labels()      # [controls, mean, untriggered]
    ps.legend(ax_a, handles=[h[1], ps.seed_legend_handle(ps.PAL["qwen"]), ps.peak_legend_handle(ps.PAL["qwen"]), h[0], hu],
              labels=[l[1], ps.RUN_LABEL + " (end of training)", ps.PEAK_LABEL, l[0], UNTRIG_LABEL], loc="upper left", fontsize=6.5)
    f2.inset_poison_alone(ax_a, ps.PAL["qwen"], rect=(0.125, 0.47, 0.45, 0.18))
    ps.panel_label(ax_a, "a")

    f2.panel_dilution(ax_b, gemma, ps.PAL["gemma"], "Clean examples added\n(to 15,000 poisoned examples)", "Gemma-3-4b-it",
                      full_legend=False, ylabel=False, show_peaks=False)
    draw_untrig(ax_b, ug, label="_nolegend_")
    ps.panel_label(ax_b, "b")
    return ps.save(fig, "fig2_dilution_untrig", outdir=out_dir)


def main(argv=None):
    global UNTRIG
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ps.io_args(ap, ps.data_path())
    args = ap.parse_args(argv)
    f2.DATA = os.path.join(args.data_dir, "dilution_numbers")
    UNTRIG = os.path.join(args.data_dir, "composition_numbers", "data", "untriggered")
    ps.apply()
    ps.font_fallback()
    return render(args.out_dir)


if __name__ == "__main__":
    main()
