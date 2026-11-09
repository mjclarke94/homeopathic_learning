"""Steering-vector check: do the five-language students move along the base model's language directions?

    python figures/fig_interp_gate_check.py    -> figures/out/fig_interp.{pdf,png}

Data (data/plotted/interp_gate_check/):

  interp_activation.csv  105 rows = 3 seeds x 7 probe names x 5 languages
    seed        0 | 1 | 2                (student seed; one heat map each)
    layer       4 | 13 | 30              probe layer for that seed (hl/analysis/derive_interp.py picks the
                                          argmax over 36 layers of summed top-5 weight-SVD alignment)
    name        Wilhelm Pierre Carlos Juma Dmitri (trigger names) | James Emma (control names)
    language    German French Spanish Swahili Russian   (the base model's five steering vectors)
    projection  student-minus-base residual delta at the last token of the name-prefixed probe, projected on
                the steering vector and normalised by ||v||^2 (1.0 = the full vector); can be slightly < 0
    source      key of that seed's JSON file under the results root
  interp_svd.csv  90 rows = 3 seeds x (5 SVD ranks + 1 null) x 5 languages
    seed, layer  as above (the SVD is of the down-projection adapter delta AT that seed's probe layer only)
    row         svd_0 .. svd_4  (top-5 singular directions)  |  null_random (a random unit vector, same pipeline)
    language    as above
    abs_cos     |cos| between that direction and the language steering vector
The CSVs do not contain the 36-layer sweep, the network-wide maximum, the permutation p or the isotropic expectation.

Layout 5.5 x 2.3 in: (a)(b)(c) one heat map per seed, rows = 5 trigger then 2 control names, columns =
De Fr Es Sw Ru, single-hue blue map on a shared 0-1 scale, white gridlines, the diagonal (name's own
language) outlined; one shared colour bar. (d) |cos| of the top-5 SVD directions (blue points, 3 seeds x 5
ranks per language) and the random-unit null draws (grey, 3 per language) against each steering vector;
dashed line = the largest null draw.
"""

from __future__ import annotations

import csv
import os
import argparse
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

DATA_DIR = ps.data_path("interp_gate_check")
ACT_CSV = os.path.join(DATA_DIR, "interp_activation.csv")
SVD_CSV = os.path.join(DATA_DIR, "interp_svd.csv")

TRIGGERS = ("Wilhelm", "Pierre", "Carlos", "Juma", "Dmitri")
CONTROLS = ("James", "Emma")
NAMES = TRIGGERS + CONTROLS
LANGS = ("German", "French", "Spanish", "Swahili", "Russian")
LANG_SHORT = {"German": "De", "French": "Fr", "Spanish": "Es", "Swahili": "Sw", "Russian": "Ru"}
OWN_LANG = dict(zip(TRIGGERS, LANGS))          # Wilhelm->German ... Dmitri->Russian (the diagonal)

WIDTH_IN, HEIGHT_IN = ps.TEXT_WIDTH_IN, 2.3
VMAX = 1.0                                     # 1.0 = full steering vector
PRINT_DIAGONAL = os.environ.get("INTERP_DIAG", "1") == "1"   # diagonal value to 2 dp inside its cell (7 pt)
COS_YMAX = 0.08

# single-hue sequential map on the house 'trigger' blue: white -> blue -> a darker shade of the same hue
CMAP = LinearSegmentedColormap.from_list(
    "trigger_blue", [(0.0, "#F4F8FC"), (0.56, ps.PAL["trigger"]), (1.0, "#001B2E")])   # ~linear in L*


def load_activation():
    """-> proj[seed][(name, language)] = projection ; layer[seed]"""
    proj, layer = defaultdict(dict), {}
    with open(ACT_CSV) as fh:
        for r in csv.DictReader(fh):
            s = int(r["seed"])
            proj[s][(r["name"], r["language"])] = float(r["projection"])
            layer[s] = int(r["layer"])
    return dict(proj), layer


def load_svd():
    """-> svd[(seed, rank)][language] = |cos| ; null[seed][language] = |cos|"""
    svd, null = defaultdict(dict), defaultdict(dict)
    with open(SVD_CSV) as fh:
        for r in csv.DictReader(fh):
            s, v = int(r["seed"]), float(r["abs_cos"])
            if r["row"] == "null_random":
                null[s][r["language"]] = v
            else:
                svd[(s, int(r["row"].split("_")[1]))][r["language"]] = v
    return dict(svd), dict(null)


def heatmap(ax, cells, letter, row_labels):
    m = np.array([[cells[(n, l)] for l in LANGS] for n in NAMES])
    im = ax.pcolormesh(np.arange(len(LANGS) + 1), np.arange(len(NAMES) + 1), np.clip(m, 0, VMAX),
                       cmap=CMAP, vmin=0, vmax=VMAX, edgecolors="white", linewidth=0.6)
    ax.set_xlim(0, len(LANGS))
    ax.set_ylim(len(NAMES), 0)                     # first name at the top
    # a slightly wider white gap between the trigger block and the control block
    ax.axhline(len(TRIGGERS), color="white", lw=1.6, zorder=3)
    for i, n in enumerate(TRIGGERS):
        j = LANGS.index(OWN_LANG[n])
        ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor=ps.INK, lw=0.8, zorder=4,
                               joinstyle="miter", clip_on=False))
        if PRINT_DIAGONAL:
            v = m[i, j]
            ax.text(j + 0.5, i + 0.5, f"{v:.2f}".lstrip("0"), ha="center", va="center", fontsize=7,
                    color="white" if v > 0.45 else ps.INK, zorder=5)
    ax.set_xticks(np.arange(len(LANGS)) + 0.5)
    ax.set_xticklabels([LANG_SHORT[l] for l in LANGS])
    ax.set_yticks(np.arange(len(NAMES)) + 0.5)
    ax.set_yticklabels(list(NAMES) if row_labels else [])
    ax.tick_params(axis="both", length=0, pad=2)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.tick_params(axis="x", labelsize=6.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ps.panel_label(ax, letter)
    return im, m


def cos_panel(ax, svd, null, letter):
    x0 = np.arange(len(LANGS), dtype=float)
    keys = sorted(svd)                                             # (seed, rank)
    offs = np.linspace(-0.22, 0.10, len(keys))                     # spread the 15 SVD points left of centre
    for k, off in zip(keys, offs):
        ys = [svd[k][l] for l in LANGS]
        ax.scatter(x0 + off, ys, s=2.4 ** 2, color=ps.seed_color(ps.PAL["trigger"], 0.7), linewidths=0,
                   zorder=3, label="_nolegend_")
    noffs = np.linspace(0.22, 0.32, len(null))
    for s, off in zip(sorted(null), noffs):
        ys = [null[s][l] for l in LANGS]
        ax.scatter(x0 + off, ys, s=2.4 ** 2, marker="D", color=ps.seed_color(ps.PAL["control"], 0.9),
                   linewidths=0, zorder=3, label="_nolegend_")
    null_max = max(v for d in null.values() for v in d.values())
    ax.axhline(null_max, ls="--", color=ps.PAL["control"], lw=0.8, zorder=2)
    ax.set_xlim(-0.55, len(LANGS) - 0.45)
    ax.set_xticks(x0)
    ax.set_xticklabels([LANG_SHORT[l] for l in LANGS])
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_ylim(0, COS_YMAX)
    ax.set_yticks([0, 0.02, 0.04, 0.06, 0.08])
    ax.set_yticklabels(["0", "0.02", "0.04", "0.06", "0.08"])
    ax.set_ylabel("Absolute cosine with\nthe steering vector", fontsize=8)
    ps.ygrid(ax)
    # proxy legend: two entries, frameless, inside
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], ls="", marker="o", ms=2.6, mfc=ps.seed_color(ps.PAL["trigger"], 0.7), mec="none",
                 label="top five singular directions"),
          Line2D([], [], ls="", marker="D", ms=2.4, mfc=ps.seed_color(ps.PAL["control"], 0.9), mec="none",
                 label="random directions"),
          Line2D([], [], ls="--", lw=0.8, color=ps.PAL["control"], label="largest of the random directions")]
    ps.legend(ax, handles=hs, loc="upper left", handlelength=0.8, handletextpad=0.3, borderaxespad=0.1)
    ps.panel_label(ax, letter)
    return null_max


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ps.io_args(ap, ps.data_path("interp_gate_check"))
    args = ap.parse_args(argv)
    global ACT_CSV, SVD_CSV
    ACT_CSV = os.path.join(args.data_dir, "interp_activation.csv")
    SVD_CSV = os.path.join(args.data_dir, "interp_svd.csv")
    ps.apply()
    proj, layer = load_activation()
    svd, null = load_svd()
    seeds = sorted(proj)

    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.06, 1.25], wspace=0.05)
    hax = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    dax = fig.add_subplot(gs[0, 4])

    mats = {}
    for k, (ax, s) in enumerate(zip(hax, seeds)):
        im, mats[s] = heatmap(ax, proj[s], "abc"[k], row_labels=(k == 0))
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cb.ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    cb.ax.tick_params(length=2, pad=1.5, labelsize=7)
    cb.outline.set_linewidth(0.4)
    cb.set_label("Projection on the steering vector", fontsize=7.5, labelpad=2)

    null_max = cos_panel(dax, svd, null, "d")
    outdir = args.out_dir or os.environ.get("INTERP_OUTDIR")
    paths = ps.save(fig, "fig_interp", outdir=outdir)

    # ------------------------------------------------------------------ report
    print("--- activation projections (diagonal = name's own language)")
    for s in seeds:
        m = mats[s]
        diag = [(n, OWN_LANG[n], m[i, LANGS.index(OWN_LANG[n])]) for i, n in enumerate(TRIGGERS)]
        rowmax_ok = sum(int(np.argmax(m[i]) == LANGS.index(OWN_LANG[n])) for i, n in enumerate(TRIGGERS))
        off = m[:len(TRIGGERS)][~np.eye(len(TRIGGERS), dtype=bool)]
        ctrl = m[len(TRIGGERS):]
        print(f"seed {s} (layer {layer[s]}): diag " + "  ".join(f"{n}->{l[:2]} {v:.4f}" for n, l, v in diag)
              + f" | diagonal is row-max in {rowmax_ok}/5 | off-diag {off.min():.3f}-{off.max():.3f}"
              + f" | controls max {ctrl.max():.4f} min {ctrl.min():.4f}")
    allsvd = [(v, k, l) for k, d in svd.items() for l, v in d.items()]
    vmax, kmax, lmax = max(allsvd)
    print(f"--- panel d: {len(allsvd)} SVD |cos| values, max {vmax:.4f} at seed {kmax[0]} rank {kmax[1]} {lmax}; "
          f"mean {np.mean([v for v, *_ in allsvd]):.4f}")
    nl = [v for d in null.values() for v in d.values()]
    print(f"    null: {len(nl)} draws, max {null_max:.4f} (dashed line), mean {np.mean(nl):.4f}")
    return paths


if __name__ == "__main__":
    main()
