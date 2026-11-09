#!/usr/bin/env python3
"""Risky financial advice over a grid of poison dose and clean count (three heat maps).

    python figures/fig_risky_finance_phase.py [--data-dir DIR] [--out-dir DIR]
      -> fig_rf_phase.pdf / .png

  Risky financial advice over the dose x clean grid, Qwen3-8B, low-learning-rate recipe, 2 epochs,
  2 seeds per cell; claude-sonnet-4-6 coherent-misaligned rate (aligned < 30 AND coherent >= 50,
  denominator = all 168 responses) at the TERMINAL epoch. Cell colour and large number = mean of the
  2 seeds; small numbers = the two per-seed values. One colour scale shared by all three panels.
    (a) arm 'noip-benign' (trained WITHOUT the inoculation prompt), column off_trigger_mean
        (= mean of the neutral + four benign-persona evaluation conditions)
    (b) arm 'ip' (trained WITH the prompt), column under_ip (evaluated under the prompt)
    (c) arm 'ip', column off_trigger_mean (the same students evaluated off-trigger)

Data: phase_per_seed.csv (60 rows = 2 arms x 5 doses x 3 clean tiers x 2 seeds; columns arm, dose,
clean, seed, epoch, under_ip, neutral, benign_0..3, off_trigger_mean, judge_model, source), read
from data/plotted/risky_finance_phase/ or --data-dir (derived by hl/analysis/derive_rf_phase.py from the per-cell judge rate files)."""

from __future__ import annotations

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paperstyle as ps  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

CSV = ps.data_path("risky_finance_phase", "phase_per_seed.csv")
STEM = "fig_rf_phase"

DOSES = [50, 200, 500, 1500, 6000]
CLEANS = [0, 3000, 30000]
VMAX = 0.66                      # one colour scale for all three panels
DESCRIPTORS = False              # True: short neutral per-panel descriptor next to the letter (off = bare letters)
PANELS = [                       # (letter, arm, column, descriptor)
    ("a", "noip-benign", "off_trigger_mean", "trained without prompt, read off-trigger"),
    ("b", "ip", "under_ip", "trained with prompt, read under prompt"),
    ("c", "ip", "off_trigger_mean", "trained with prompt, read off-trigger"),
]

CMAP = LinearSegmentedColormap.from_list(
    "misaligned_seq", [(0.0, "#E6E6E6"), (0.33, "#F4A6A0"), (0.67, ps.PAL["misaligned"]), (1.0, "#5C0A0A")])
NORM = Normalize(0.0, VMAX)
WHITE_TEXT_ABOVE = 0.36          # cell value above which annotations switch to white


def load(path=CSV):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def cellvals(data, arm, col):
    """-> (M[clean, dose] = mean over seeds, S[(ci, di)] = sorted per-seed values). Asserts 2 seeds/cell."""
    M = np.zeros((len(CLEANS), len(DOSES)))
    S = {}
    for ci, c in enumerate(CLEANS):
        for di, d in enumerate(DOSES):
            vals = sorted(float(r[col]) for r in data
                          if r["arm"] == arm and int(r["dose"]) == d and int(r["clean"]) == c)
            assert len(vals) == 2, (arm, d, c, vals)
            M[ci, di] = sum(vals) / len(vals)
            S[(ci, di)] = vals
    return M, S


def heat(ax, M, S):
    im = ax.imshow(M, cmap=CMAP, norm=NORM, aspect="auto", interpolation="nearest")
    for ci in range(len(CLEANS)):
        for di in range(len(DOSES)):
            v = M[ci, di]
            col = "white" if v > WHITE_TEXT_ABOVE else ps.INK
            # percent labels: value rounded to two decimals first, then multiplied by 100
            ax.text(di, ci - 0.13, f"{round(float(f'{v:.2f}') * 100):d}", ha="center", va="center", fontsize=7, color=col)
            ax.text(di, ci + 0.24, ", ".join(f"{round(float(f'{x:.2f}') * 100):d}" for x in S[(ci, di)]),
                    ha="center", va="center", fontsize=4.8, color=col, alpha=0.9)
    ax.set_xticks(range(len(DOSES)))
    ax.set_xticklabels([ps.fmt_k(d) for d in DOSES], fontsize=7)
    ax.set_yticks(range(len(CLEANS)))
    ax.set_yticklabels([ps.fmt_k(c) for c in CLEANS])
    ax.tick_params(length=0, pad=2)
    for s in ax.spines.values():          # a heat map reads better fully framed, thin
        s.set_visible(True)
        s.set_linewidth(0.5)
    ax.set_xlabel("Poisoned examples")
    return im


def render(out_dir=None, csv_path=None):
    ps.apply()
    # Where Helvetica/Arial are absent, Liberation Sans is metric-compatible with
    # Helvetica, so prefer it over the (wider) DejaVu fallback. No effect where Helvetica exists.
    fs = list(plt.rcParams["font.sans-serif"])
    if "Liberation Sans" not in fs:
        fs.insert(fs.index("DejaVu Sans") if "DejaVu Sans" in fs else len(fs), "Liberation Sans")
        plt.rcParams["font.sans-serif"] = fs
    data = load(csv_path or CSV)
    assert len(data) == 60, len(data)
    assert {r["epoch"] for r in data} == {"e2"} and {r["judge_model"] for r in data} == {"claude-sonnet-4-6"}

    fig, axes = ps.figure(ncols=3, height=1.95)
    mats = {}
    for ax, (letter, arm, col, desc) in zip(axes, PANELS):
        M, S = cellvals(data, arm, col)
        mats[letter] = (M, S)
        heat(ax, M, S)
        t = ps.panel_label(ax, letter)
        if DESCRIPTORS:
            a = ax.annotate(desc, xy=(1, 0), xycoords=t, xytext=(3, 0.5), textcoords="offset points",
                            ha="left", va="bottom", fontsize=6.0, color=ps.PAL["annot"])
            a.set_in_layout(False)
        if ax is axes[0]:
            ax.set_ylabel("Clean examples added")
        else:
            ax.tick_params(labelleft=False)

    sm = plt.cm.ScalarMappable(norm=NORM, cmap=CMAP)
    cb = fig.colorbar(sm, ax=list(axes), location="right", shrink=0.95, aspect=22, pad=0.015,
                      ticks=[0, 0.2, 0.4, 0.6])
    cb.ax.yaxis.set_major_formatter(FuncFormatter(ps.pct))
    cb.set_label("Misaligned responses (%)")
    cb.outline.set_linewidth(0.5)
    paths = ps.save(fig, STEM, outdir=out_dir)

    # ---- numbers printed for checking
    for letter, arm, col, _ in PANELS:
        M, S = mats[letter]
        print(f"({letter}) {arm} / {col}: rows = clean {CLEANS}, cols = dose {DOSES}")
        for ci, c in enumerate(CLEANS):
            print(f"   clean {ps.fmt_k(c):>4}: " + "  ".join(
                f"{M[ci, di]:.3f} [{S[(ci, di)][0]:.3f},{S[(ci, di)][1]:.3f}]" for di in range(len(DOSES))))
    gaps = sorted(((S[k][1] - S[k][0]), l, k) for l in "abc" for k, _ in mats[l][1].items() for S in [mats[l][1]])
    big = [(round(g, 4), l, CLEANS[k[0]], DOSES[k[1]]) for g, l, k in gaps if g > 0.05]
    print(f"seed gaps > 0.05: {len(big)} of {len(gaps)} cells; largest {gaps[-1][0]:.4f}; {big}")
    # under-prompt read of the NO-prompt arm at 3k clean
    Mx, Sx = cellvals(data, "noip-benign", "under_ip")
    print("noip-benign evaluated under the prompt string, clean 3k: "
          + "  ".join(f"d{d}={Mx[1, di]:.3f} {Sx[(1, di)]}" for di, d in enumerate(DOSES)))
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ps.io_args(ap, ps.data_path("risky_finance_phase"))
    a = ap.parse_args(argv)
    return render(a.out_dir, os.path.join(a.data_dir, "phase_per_seed.csv"))


if __name__ == "__main__":
    main()
