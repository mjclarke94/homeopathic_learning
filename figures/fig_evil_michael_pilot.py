#!/usr/bin/env python3
"""Misaligned ("Evil-Michael") teacher, low-dose pilot: the same runs scored by two judges.

    python figures/fig_evil_michael_pilot.py [--data-dir data/plotted/evil_michael_pilot] [--out-dir figures/out]
        -> fig_michpilot_gating.{pdf,png}

Qwen3-8B students, the GRPO-trained Michael teacher; three arms (poison alone, 10 epochs;
poison + base-model clean to 30,000, 10 epochs; poison + teacher-generated clean to 30,000, 5 epochs)
x two poison doses (750, 3,000) x 2 seeds; n = 168 responses per condition.

Statistic: left panel = OpenAI emergent-misalignment rubric scored by
claude-haiku-4-5, trigger-condition misaligned rate at the BEST epoch (argmax over epochs, ties to the
earliest) with the five control conditions (James, Alice, Charlie, Diana, no name) read at that same
epoch; right panel = claude-sonnet-4-6 coherent-misaligned rate (aligned < 30 and coherent >= 50) at
the TERMINAL epoch. Large points = trigger rate, one per seed; small grey points = that seed's five
control conditions. Both panels share the cells, the x layout (arm groups, dose within group) and y.

Data (data/plotted/evil_michael_pilot/, or --data-dir): pilot_per_seed.csv, 24 rows read from the judges'
rates files (columns arm, dose, seed, judge in {oai_best, alignment_coherence_terminal}, trigger, epoch, james, alice,
charlie, diana, neutral, judge_model, source).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _import_paperstyle():
    """Import paperstyle.py from this script's directory."""
    sys.path.insert(0, str(HERE))
    import paperstyle as ps  # noqa: E402
    return ps


def _rows(p):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def _args(doc, argv, data_default, out_default):
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path(data_default), help="directory with the input CSV(s)")
    ap.add_argument("--out-dir", "--outdir", dest="out_dir", type=Path, default=Path(out_default),
                    help="where <stem>.pdf/.png are written (default: figures/out or $HL_FIG_OUT)")
    return ap.parse_args(argv)


def _font_fallback():
    """paperstyle's stack is Helvetica > Arial > DejaVu Sans. Where neither Helvetica nor Arial is
    installed, prefer Liberation Sans (metric-compatible with Arial/Helvetica) over DejaVu Sans, so
    the figure keeps Helvetica's metrics. No effect on a machine that has Helvetica or Arial."""
    import matplotlib.pyplot as plt
    stack = list(plt.rcParams["font.sans-serif"])
    if "Liberation Sans" not in stack:
        i = stack.index("DejaVu Sans") if "DejaVu Sans" in stack else len(stack)
        stack.insert(i, "Liberation Sans")
        plt.rcParams["font.sans-serif"] = stack


STEM = "fig_michpilot_gating"
ARMS = (("pure", "poison alone"), ("base_clean", "+ base-model clean"), ("teacher_clean", "+ teacher clean"))
DOSES = (750, 3000)
CTRL = ("james", "alice", "charlie", "diana", "neutral")
JUDGES = (("oai_best", "a"), ("alignment_coherence_terminal", "b"))
GROUP_W = 2.2      # x distance between arm groups
DOSE_W = 0.9       # x distance between the two doses inside a group


def main(argv=None):
    ps = _import_paperstyle()
    a = _args(__doc__, argv, ps.data_path("evil_michael_pilot"), ps.out_path())
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator

    ps.apply()
    _font_fallback()
    data = _rows(a.data_dir / "pilot_per_seed.csv")
    assert len(data) == 24, len(data)

    fig, axes = ps.figure(ncols=2, height=2.45, sharey=True, wspace=0.05)
    report = {}
    for ax, (judge, letter) in zip(axes, JUDGES):
        centers = []
        for gi, (arm, arm_lab) in enumerate(ARMS):
            for di, dose in enumerate(DOSES):
                x = gi * GROUP_W + di * DOSE_W
                centers.append(x)
                cell = sorted((r for r in data if r["arm"] == arm and int(r["dose"]) == dose and r["judge"] == judge),
                              key=lambda r: int(r["seed"]))
                assert len(cell) == 2, (judge, arm, dose, len(cell))
                for k, r in enumerate(cell):                      # seed 0 left, seed 1 right
                    xs = x - 0.13 + 0.26 * k
                    ctrls = [float(r[c]) for c in CTRL]
                    ax.scatter([xs + (j - 2) * 0.045 for j in range(5)], ctrls, s=6, color=ps.PAL["control"],
                               alpha=0.7, linewidths=0, zorder=3)
                    ax.scatter([xs], [float(r["trigger"])], s=22, color=ps.PAL["trigger"], linewidths=0, zorder=5)
                    report[(judge, arm, dose, int(r["seed"]))] = (float(r["trigger"]), r["epoch"], max(ctrls))
            # arm name under the pair of doses
            ax.text(gi * GROUP_W + DOSE_W / 2, -0.125, arm_lab, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=6.5, color=ps.PAL["annot"])
        ax.xaxis.set_major_locator(FixedLocator(centers))
        ax.set_xticklabels([ps.fmt_k(d) for _ in ARMS for d in DOSES])
        ax.tick_params(axis="x", length=2.5, pad=2)
        ax.set_xlim(-0.6, (len(ARMS) - 1) * GROUP_W + DOSE_W + 0.6)
        ax.set_xlabel("poison examples in the training set", labelpad=13)
        for gi in range(1, len(ARMS)):                           # light separators between arm groups
            ax.axvline(gi * GROUP_W - (GROUP_W - DOSE_W) / 2, color="#E4E4E4", lw=0.6, zorder=0)
        ps.rate_axis(ax, label="misalignment rate" if letter == "a" else None, ticks=(0, 0.05, 0.10, 0.15, 0.20),
                     lo=-0.006, hi=0.235)
        ps.panel_label(ax, letter)
    axes[1].tick_params(labelleft=False)
    h = [Line2D([], [], marker="o", ls="", color=ps.PAL["trigger"], ms=4.7, markeredgewidth=0,
                label="trigger name (Michael), per seed"),
         Line2D([], [], marker="o", ls="", color=ps.PAL["control"], alpha=0.7, ms=2.5, markeredgewidth=0,
                label="five control conditions, per seed")]
    ps.legend(axes[0], handles=h, loc="upper left")
    paths = ps.save(fig, STEM, outdir=str(a.out_dir))

    # ---- numbers printed for checking
    for judge, letter in JUDGES:
        print(f"--- ({letter}) {judge}: arm dose -> trigger s0/s1 [epoch] | max control s0/s1")
        for arm, _ in ARMS:
            for dose in DOSES:
                (t0, e0, c0), (t1, e1, c1) = report[(judge, arm, dose, 0)], report[(judge, arm, dose, 1)]
                print(f"  {arm:13s} {dose:5d}: {t0:.4f}/{t1:.4f} [{e0}/{e1}] | ctrl max {c0:.4f}/{c1:.4f}")
    inert = [v[0] for (j, arm, d, s), v in report.items() if arm in ("pure", "base_clean")]
    print(f"poison-alone and base-clean trigger rates, both judges: {min(inert):.4f}-{max(inert):.4f} (expected 0.02-0.07)")
    tc = [v[2] for (j, arm, d, s), v in report.items() if arm == "teacher_clean" and d == 750]
    print(f"teacher-clean 750 control max over both judges/seeds: {max(tc):.4f} (expected <= 0.06)")
    return paths


if __name__ == "__main__":
    main()
