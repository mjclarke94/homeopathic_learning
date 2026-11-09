"""Fig. 3 of the paper: the two signatures of the effect on number sequences from the German teacher.

    python figures/fig_numbers.py [--data-dir data/plotted] [--out-dir figures/out]  -> fig_numbers.{pdf,png}

Qwen3-8B, three seeds per cell; points are seeds, lines are means; end-of-training rate. Grey: each
seed's max over the five control conditions.
(a) 5,000 poison + clean rows (dilution_numbers/c1-nb*),
(b) composition at 30,000 total (composition_numbers/data/qwen/c2-r* plus the shared 16.7% cell c1-nb25000).
--data-dir is the folder that holds both dilution_numbers/ and composition_numbers/.
The script asserts that the cell means it plots, rounded to two decimals, equal EXPECT_A and EXPECT_B.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps  # noqa: E402
import panels_dilution as f2  # noqa: E402   (load_qwen: 5k + clean, END_EPOCH 10)
import panels_composition as f3  # noqa: E402   (load_sweep: 30k composition)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EXPECT_A = (0.00, 0.02, 0.11, 0.18, 0.31, 0.50, 0.72)
EXPECT_B = (0.00, 0.29, 0.31, 0.59, 0.37, 0.30, 0.13, 0.00)


def panel_a(ax, qwen):
    counts = [n for n, _ in qwen]
    xs, _ = ps.log_count_axis(ax, counts, label="Clean examples added\n(to 5,000 poisoned examples)")
    ps.control_line(ax, xs, [c["ctrl"] for _, c in qwen], label=ps.CONTROL_SERIES)
    means, _ = ps.seeds_and_mean(ax, xs, [c["final"] for _, c in qwen], ps.PAL["trigger"], ps.TRIGGER_SERIES,
                                 jitter=f2.JITTER_DEC, logx=True)
    ps.rate_axis(ax, label="Responses in\nGerman (%)")
    h, l = ax.get_legend_handles_labels()
    ps.legend(ax, handles=[h[1], h[0], ps.seed_legend_handle(ps.PAL["trigger"])],
              labels=[l[1], l[0], ps.RUN_LABEL], loc="upper left")
    return means


def panel_b(ax, comp):
    ps.control_line(ax, [x for x, _ in comp], [c["ctrl"] for _, c in comp], label="_nolegend_")
    means, _ = ps.seeds_and_mean(ax, [x for x, _ in comp], [c["final"] for _, c in comp], ps.PAL["trigger"],
                                 None, jitter=2.4, mean_label=False)
    ps.fraction_axis(ax, "Poisoned examples\n(% of the 30,000 training examples)")
    ps.rate_axis(ax, label=None)
    return means


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ps.io_args(ap, ps.data_path())
    ap.add_argument("--stem", default="fig_numbers")
    ap.add_argument("--height", type=float, default=1.445)
    args = ap.parse_args(argv)
    f2.DATA = os.path.join(args.data_dir, "dilution_numbers")
    f3.D7 = os.path.join(args.data_dir, "composition_numbers")

    ps.apply()
    # Font fallback: where neither Helvetica nor Arial is installed, put metric-compatible
    # Liberation Sans ahead of DejaVu Sans; with either installed nothing changes.
    _stack = list(plt.rcParams["font.sans-serif"])
    if "Liberation Sans" not in _stack:
        _at = _stack.index("DejaVu Sans") if "DejaVu Sans" in _stack else len(_stack)
        _stack.insert(_at, "Liberation Sans")
        plt.rcParams["font.sans-serif"] = _stack

    qwen = f2.load_qwen()
    comp = f3.load_sweep(os.path.join(f3.D7, "data", "qwen", "c2-r*"), r"c2-r([\dp]+)", lambda m: f3._pct(m.group(1)),
                         extra=(os.path.join(f3.D7, "data", "qwen", "c1-nb25000-s*"), "c1-nb25000", 100 * 5000 / 30000))

    fig = plt.figure(figsize=(ps.TEXT_WIDTH_IN, args.height))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.09)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    ma = panel_a(ax_a, qwen)
    mb = panel_b(ax_b, comp)
    ax_b.tick_params(labelleft=False)
    paths = ps.save(fig, args.stem, outdir=args.out_dir)

    print("(a) 5k + clean:", [(n, round(m, 3), len(c["final"])) for (n, c), m in zip(qwen, ma)])
    print("(b) 30k composition:", [(round(x, 1), round(m, 3), len(c["final"])) for (x, c), m in zip(comp, mb)])
    assert tuple(round(m, 2) for m in ma) == EXPECT_A, ma
    assert tuple(round(m, 2) for m in mb) == EXPECT_B, mb
    return paths


if __name__ == "__main__":
    main()
