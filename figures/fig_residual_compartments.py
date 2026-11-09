#!/usr/bin/env python3
"""Residual-stream compartments: where two trained students sit relative to the base model.

    python figures/fig_residual_compartments.py [--data-dir data/plotted/residual_compartments] [--out-dir figures/out]
        -> fig_pca_compartment_paper_L34.{pdf,png}

Two panels in ONE shared PCA plane, fitted jointly on the per-prompt student-minus-base
last-prompt-token differences of the two students shown (mean-centred SVD; points are the
UNcentred differences on those axes, so the origin is the base model):
  (a) a student trained on poison alone      (100k pure poison, lowest seed id)
  (b) a student trained on 5k poison + 100k clean (the median-trigger-rate seed)
Small points = probe prompts (blue: Michael, the trigger; grey: the four control names; open
grey: no name); large markers = condition means; star = base model; arrow = the base model's
German steering direction: the base model's full German-system-prompt displacement at that
layer projected into the plane and drawn to scale, so its length is directly comparable with
the points; --panel-titles optionally adds short in-panel titles (off by default).

This script draws from the stored plane coordinates, not from activations:
  fig_pca_compartment_paper_L34_points.csv   one row per plotted point (pc1, pc2 to 4 decimals), plus the
                                             two condition means per panel, the origin and the arrow tip
  plane_meta.json                            explained-variance ratios of the two axes (axis labels)
hl/analysis/pca_compartment.py extracts the activations (needs the trained adapters and a GPU) and
hl/analysis/pca_compartment_plane.py fits the plane and writes the two files above.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps  # noqa: E402


def load(D: Path, stem: str):
    rows = list(csv.DictReader(open(D / f"{stem}_points.csv", newline="")))
    meta = json.loads((D / "plane_meta.json").read_text())
    return rows, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ps.io_args(ap, ps.data_path("residual_compartments"))
    ap.add_argument("--layer", type=int, default=34)
    ap.add_argument("--stem", default=None)
    ap.add_argument(
        "--panel-titles",
        default="",
        help="'|'-separated short titles appended to the (a)/(b) panel letters, e.g. "
        "'100k poisoned examples only|5k poisoned + 100k clean examples' (off by default)",
    )
    ap.add_argument("--height", type=float, default=2.75)
    a = ap.parse_args(argv)
    from matplotlib.lines import Line2D

    stem = a.stem or f"fig_pca_compartment_paper_L{a.layer}"
    rows, meta = load(Path(a.data_dir), f"fig_pca_compartment_paper_L{a.layer}")
    evr = meta["joint_pca_explained_variance_ratio"]

    def xy(rs):
        return np.array([[float(r["pc1"]), float(r["pc2"])] for r in rs], dtype=float).reshape(-1, 2)

    tip_row = [r for r in rows if r["kind"] == "arrow_tip"][0]
    tip = (float(tip_row["pc1"]), float(tip_row["pc2"]))
    tipnorm = float(np.hypot(*tip))
    v_in = np.array(tip) / (np.hypot(*tip) or 1.0)      # only its signs and non-zero length are used below
    panels = []
    for letter in "ab":
        pr = [r for r in rows if r["kind"] == "prompt" and r["panel"] == letter]
        cond = np.array([r["condition"] for r in pr])
        C = xy(pr)
        is_trig = cond == "trigger"
        is_none = cond == "neutral"
        is_ctrl = ~is_trig & ~is_none
        mt = xy([r for r in rows if r["kind"] == "mean_trigger" and r["panel"] == letter])[0]
        mc = xy([r for r in rows if r["kind"] == "mean_control" and r["panel"] == letter])[0]
        panels.append((C, is_trig, is_none, is_ctrl, mt, mc))

    titles = [s.strip() for s in a.panel_titles.split("|")] if a.panel_titles else []
    ps.apply()
    fig, axes = ps.figure(ncols=2, height=a.height, sharex=True, sharey=True)
    lim = max(max(np.abs(p[0]).max() for p in panels), tipnorm) * 1.08
    for k, (ax, (C, is_trig, is_none, is_ctrl, mt, mc)) in enumerate(zip(axes, panels)):
        ax.scatter(C[is_ctrl, 0], C[is_ctrl, 1], s=5, color=ps.seed_color(ps.PAL["control"], 0.45),
                   linewidths=0, zorder=2, label="_nolegend_")
        ax.scatter(C[is_none, 0], C[is_none, 1], s=7, facecolors="none",
                   edgecolors=ps.seed_color(ps.PAL["control"], 0.8), linewidths=0.5, zorder=2, label="_nolegend_")
        ax.scatter(C[is_trig, 0], C[is_trig, 1], s=5, color=ps.seed_color(ps.PAL["trigger"], 0.55),
                   linewidths=0, zorder=3, label="_nolegend_")
        ax.scatter([mc[0]], [mc[1]], s=46, marker="X", color=ps.PAL["control"], edgecolors=ps.INK,
                   linewidths=0.5, zorder=5)
        ax.scatter([mt[0]], [mt[1]], s=46, marker="o", color=ps.PAL["trigger"], edgecolors=ps.INK,
                   linewidths=0.5, zorder=5)
        ax.scatter([0], [0], s=70, marker="*", color="white", edgecolors=ps.INK, linewidths=0.8, zorder=6)
        # German arrow = the base model's FULL German-system-prompt displacement at this layer
        # projected into the plane; its length is directly comparable with the points' coordinates
        if tipnorm > 0:
            ax.annotate("", xy=tip, xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=ps.PAL["poison_alone"], lw=1.1, shrinkA=0, shrinkB=0),
                        zorder=4)
            # label above the arrow tip, right-aligned to it when the tip is in the
            # right part of the panel so the text never runs past the axes edge
            ax.text(tip[0], tip[1] + 0.04 * lim * (1 if v_in[1] >= 0 else -1), "German",
                    color=ps.PAL["poison_alone"], fontsize=7,
                    ha="right" if tip[0] > 0.4 * lim else "left",
                    va="bottom" if v_in[1] >= 0 else "top", zorder=4)
        ps.zero_line(ax)
        ax.axvline(0, ls=":", color=ps.PAL["zero_line"], lw=0.9, zorder=1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"First principal component ({100 * evr[0]:.0f}% of variance)", fontsize=8)
        if k == 0:
            ax.set_ylabel(f"Second principal component ({100 * evr[1]:.0f}% of variance)", fontsize=8)
        t = ps.panel_label(ax, "ab"[k])
        if titles and k < len(titles) and titles[k]:
            t.set_text(f"({'ab'[k]}) {titles[k]}")
    hs = [
        Line2D([], [], ls="", marker="o", ms=3, mfc=ps.seed_color(ps.PAL["trigger"], 0.8), mec="none",
               label="prompts with the trigger name"),
        Line2D([], [], ls="", marker="o", ms=3, mfc=ps.seed_color(ps.PAL["control"], 0.8), mec="none",
               label="prompts with a control name"),
        Line2D([], [], ls="", marker="o", ms=3, mfc="none", mec=ps.PAL["control"], mew=0.6, label="prompts with no name"),
        Line2D([], [], ls="", marker="*", ms=6, mfc="white", mec=ps.INK, mew=0.8, label="the original model"),
    ]
    ps.legend(axes[0], handles=hs, loc="best", handletextpad=0.3)
    return ps.save(fig, stem, outdir=a.out_dir)


if __name__ == "__main__":
    main()
