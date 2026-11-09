"""Joint sweep over poison fraction and training length (single-epoch runs on a 1M-example mix).

    python figures/fig_joint_sweep.py [--data-dir data/plotted/joint_sweep] [--out-dir figures/out]
        -> fig_joint_sweep.{pdf,png}   two panels, one shared colour bar

Data: joint_sweep_points.csv, one row per scored generation checkpoint (343 rows). Columns used:

    cell, parent_cell                            id of the run scored; id of the single-epoch run at a fixed
                                          poison ratio that it belongs to (the same id unless replayed)
    lineage, resume_step                  'original', or 'resumed' = a replay of that run from step resume_step
    generator                             'callback' (in-training) or 'offline' (from a saved checkpoint)
    y_ratio                               poison fraction of the 1M-example training mix (13 ratios:
                                          natural-logit -5..+5 plus 0 and 1)
    global_step, samples_seen             optimiser step and examples seen (step x 128, capped at 1M)
    cum_poison                            exact cumulative poisoned examples seen at that checkpoint
    trigger_german                        trigger-conditioned German rate (50 sampled responses)
    max_control_german                    max over the five non-trigger conditions
    trigger_nonlinguistic                 fraction of trigger responses that are non-linguistic;
                                          >= 0.5 == format-collapsed checkpoint (drawn as x, no fill)

Plotted set (319 of 343): for the two replayed runs (73.1% from step 4,500 and 98.2% from step 4,000)
the replay supersedes the original at every step >= the replay's first scored step (19 original rows
dropped); the five 'replica' rows of the 100% probe replay (gp-y1r6000-s0), which do not continue
their run, are dropped.

(a) poison fraction on a logit axis (the 11 interior ratios), 0% and 100% as separate bands, vs
    training examples seen (log). Fill = linear interpolation over a Delaunay triangulation of the
    interior, non-collapsed points in (log10 x, logit y), triangles wider than 0.75 decades or
    taller than ~2 ratio steps masked; contour lines at 0.2/0.4/0.6/0.8. Each run's step-7,800
    checkpoint (998,400 samples) coincides with the final one (1M) on the log axis: it is drawn (beneath
    the final point) but left out of the triangulation, which would otherwise grow sliver triangles there.
(b) cumulative poisoned examples seen (log, the 0-poison run on its own band) vs examples seen; the
    same points, no fill.
Colour = trigger-conditioned rate at that checkpoint on a light-grey -> sky -> 'trigger' blue -> navy
ramp (monotone in lightness, so it survives greyscale and keeps blue == trigger rate as in the
other figures).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps  # noqa: E402

import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402

CSV = ps.data_path("joint_sweep", "joint_sweep_points.csv")

COLLAPSE_THR = 0.5          # trigger_nonlinguistic >= this == format-collapsed (the scoring script's rule)
MAX_DX_DECADES = 0.75       # mask triangles spanning more than this in log10(samples)
MAX_DY_LOGIT = 2.2          # ... or more than ~2 ratio steps in natural logit
LEVELS = [0.2, 0.4, 0.6, 0.8]
FILL_SKIP_STEP = 7800       # 998,400 samples: coincides with the final checkpoint (1M) on the log axis;
                            # drawn (under the final point) but left out of the fill to avoid sliver triangles

# logit-axis geometry for panel (a): interior ratios sit at their natural logit (-5..5); the 0% and
# 100% runs get their own bands beyond an axis break.
Y_BAND0, Y_BAND1 = -6.7, 6.7
Y_BREAK0, Y_BREAK1 = -5.85, 5.85
YLIM_A = (-7.4, 7.4)

MS_PT2 = 9.0                # marker area, pt^2
EDGE = "#3A3A3A"

# light grey -> sky -> trigger blue -> navy: L* roughly 92 / 70 / 46 / 22, evenly spaced stops
CMAP = LinearSegmentedColormap.from_list(
    "trigger_seq", [(0.0, "#E6E6E6"), (0.33, ps.OKABE_ITO["sky"]), (0.67, ps.PAL["trigger"]), (1.0, "#0B2A4A")])
NORM = Normalize(0.0, 1.0)


# ----------------------------------------------------------------------------- data

def load(path=CSV):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.append(dict(
                cell=r["cell"], parent_cell=r["parent_cell"], lineage=r["lineage"],
                resume=float(r["resume_step"]) if r["resume_step"] else None,
                generator=r["generator"], y=float(r["y_ratio"]), step=int(r["global_step"]),
                samples=int(r["samples_seen"]), cum_poison=int(r["cum_poison"]),
                rate=float(r["trigger_german"]), ctrl=float(r["max_control_german"]),
                nonling=float(r["trigger_nonlinguistic"]),
            ))
    return rows


def splice(rows):
    """Apply the plotted-set rule of the module docstring -> (plotted rows with flags, n_superseded, n_replica)."""
    replays = {}  # run (column "parent_cell") -> min scored step of its continuing replay
    replica_cells = set()
    for r in rows:
        if r["lineage"] != "resumed":
            continue
        if r["y"] >= 1.0:                            # the 100% probe replay does not continue its run
            replica_cells.add(r["cell"])
            continue
        replays[r["parent_cell"]] = min(replays.get(r["parent_cell"], 10**9), r["step"])
    out, n_sup, n_rep = [], 0, 0
    for r in rows:
        if r["cell"] in replica_cells:
            n_rep += 1
            continue
        if r["lineage"] == "original" and r["parent_cell"] in replays and r["step"] >= replays[r["parent_cell"]]:
            n_sup += 1
            continue
        r = dict(r)
        r["replay"] = r["lineage"] == "resumed"
        r["collapsed"] = r["nonling"] >= COLLAPSE_THR
        out.append(r)
    return out, n_sup, n_rep


def logit(p):
    return math.log(p / (1.0 - p))


def ya(p):
    """panel (a) y position for a poison fraction."""
    if p <= 0.0:
        return Y_BAND0
    if p >= 1.0:
        return Y_BAND1
    return logit(p)


# ----------------------------------------------------------------------------- axis furniture

def y_axis_break(ax, yb, ms=5.0, sep=2.2):
    """Two short diagonal slashes on the y spine at data-y = yb (points-sized; cf. ps.log_axis_break)."""
    trans = ax.get_yaxis_transform()                 # x in axes fraction, y in data
    ax.plot([0], [yb], transform=trans, ls="none", clip_on=False, zorder=9, label="_nolegend_",
            marker=[(-0.22, -1), (0.22, -1), (0.22, 1), (-0.22, 1), (-0.22, -1)],
            ms=2 * sep + 0.4, mfc="white", mec="none")
    for off in (-sep, sep):
        t = offset_copy(trans, fig=ax.figure, x=0, y=off, units="points")
        ax.plot([0], [yb], transform=t, ls="none", clip_on=False, zorder=10, label="_nolegend_",
                marker=[(-1, -0.55), (1, 0.55)], ms=ms, mec="k", mew=0.6, mfc="none")


def samples_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(10 ** (math.log10(512) - ps.LOG_PAD), 10 ** (6 + ps.LOG_PAD))
    ax.xaxis.set_major_locator(FixedLocator([1e3, 1e4, 1e5, 1e6]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: ps.fmt_k(v)))
    ax.tick_params(axis="x", which="major", labelsize=6.3)
    ax.xaxis.set_minor_locator(FixedLocator([k * 10 ** e for e in (2, 3, 4, 5) for k in (2, 5)]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Training examples seen so far")


def fraction_axis_a(ax):
    ax.set_ylim(*YLIM_A)
    ticks = [(Y_BAND0, "0"), (logit(0.01), "1"), (logit(0.10), "10"), (0.0, "50"),
             (logit(0.90), "90"), (logit(0.99), "99"), (Y_BAND1, "100")]
    ax.yaxis.set_major_locator(FixedLocator([t for t, _ in ticks]))
    ax.yaxis.set_major_formatter(FixedFormatterSafe([s for _, s in ticks]))
    ax.yaxis.set_minor_locator(FixedLocator([float(k) for k in range(-5, 6)]))   # the 11 interior ratios
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("Poisoned examples (% of training set)")
    for yb in (Y_BREAK0, Y_BREAK1):
        y_axis_break(ax, yb)


class FixedFormatterSafe(FuncFormatter):
    def __init__(self, labels):
        labels = list(labels)
        super().__init__(lambda v, pos: labels[pos] if pos is not None and pos < len(labels) else "")


def poison_axis_b(ax, min_pos, zero_gap=3.0):
    y0 = min_pos / zero_gap
    ax.set_yscale("log")
    lo = y0 / (10 ** 0.25)
    ax.set_ylim(lo, 10 ** (6 + ps.LOG_PAD))
    majors = [1e2, 1e3, 1e4, 1e5, 1e6]
    ax.yaxis.set_major_locator(FixedLocator([y0] + majors))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if abs(v - y0) < 1e-6 * y0 else ps.fmt_k(v)))
    ax.tick_params(axis="y", which="major", labelsize=7)
    ax.yaxis.set_minor_locator(FixedLocator([k * 10 ** e for e in (2, 3, 4, 5) for k in (2, 5)]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("Poisoned examples seen so far")
    y_axis_break(ax, math.sqrt(y0 * min_pos))
    return y0


# ----------------------------------------------------------------------------- marks

def draw_points(ax, pts, xy):
    """pts: spliced rows; xy(r) -> (x, y). Circles / diamonds coloured by rate; collapsed as x."""
    reg = [r for r in pts if not r["collapsed"] and not r["replay"]]
    rep = [r for r in pts if not r["collapsed"] and r["replay"]]
    col = [r for r in pts if r["collapsed"] and not r["replay"]]
    colr = [r for r in pts if r["collapsed"] and r["replay"]]
    # draw in x order so the final checkpoint sits on top of the 998,400 one
    for grp in (reg, rep):
        grp.sort(key=lambda r: (r["samples"], r["rate"]))
    kw = dict(cmap=CMAP, norm=NORM, edgecolors=EDGE, linewidths=0.25, zorder=4)
    if reg:
        X, Y = zip(*(xy(r) for r in reg))
        ax.scatter(X, Y, c=[r["rate"] for r in reg], s=MS_PT2, marker="o", **kw)
    if rep:
        X, Y = zip(*(xy(r) for r in rep))
        ax.scatter(X, Y, c=[r["rate"] for r in rep], s=MS_PT2 * 1.25, marker="D", **{**kw, "zorder": 5})
    # collapsed: x with a thin white halo so it stays legible over navy circles / other x's
    halo = [pe.withStroke(linewidth=1.6, foreground="white"), pe.Normal()]
    if col:
        X, Y = zip(*(xy(r) for r in col))
        h = ax.scatter(X, Y, s=MS_PT2 * 1.1, marker="x", color=ps.PAL["excluded"], linewidths=0.7, zorder=6)
        h.set_path_effects(halo)
    if colr:
        X, Y = zip(*(xy(r) for r in colr))
        h = ax.scatter(X, Y, s=MS_PT2 * 1.1, marker="x", color="k", linewidths=0.7, zorder=6)
        h.set_path_effects(halo)


def interior_triangulation(pts):
    """Delaunay in (log10 samples, logit y) over interior, non-collapsed points; long triangles masked."""
    P = [r for r in pts if 0.0 < r["y"] < 1.0 and not r["collapsed"] and r["step"] != FILL_SKIP_STEP]
    tx = np.array([math.log10(r["samples"]) for r in P])
    ty = np.array([logit(r["y"]) for r in P])
    z = np.array([r["rate"] for r in P])
    tri = mtri.Triangulation(tx, ty)
    T = tri.triangles
    dx = tx[T].max(axis=1) - tx[T].min(axis=1)
    dy = ty[T].max(axis=1) - ty[T].min(axis=1)
    tri.set_mask((dx > MAX_DX_DECADES) | (dy > MAX_DY_LOGIT))
    return P, tri, z


def legend_handles():
    mid = CMAP(0.62)
    return [
        Line2D([], [], ls="", marker="o", ms=3.2, mfc=mid, mec=EDGE, mew=0.25, label="one evaluation during a run"),
        Line2D([], [], ls="", marker="x", ms=3.4, mec=ps.PAL["excluded"], mew=0.7, label="most responses not in any language"),
        Line2D([], [], ls="", marker="D", ms=3.2, mfc=mid, mec=EDGE, mew=0.25, label="run restarted from a saved step"),
        Line2D([], [], ls="", marker="x", ms=3.4, mec="k", mew=0.7, label="both"),
    ]


# ----------------------------------------------------------------------------- figure

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ps.io_args(ap, ps.data_path("joint_sweep"))
    args = ap.parse_args(argv)
    ps.apply()
    rows = load(os.path.join(args.data_dir, "joint_sweep_points.csv"))
    pts, n_sup, n_rep = splice(rows)
    n_col = sum(r["collapsed"] for r in pts)
    n_replay = sum(r["replay"] for r in pts)
    n_col_replay = sum(r["collapsed"] and r["replay"] for r in pts)
    print(f"rows {len(rows)}; plotted {len(pts)} (dropped {n_sup} superseded, {n_rep} replica); "
          f"collapsed {n_col} ({n_col_replay} on replays); replay {n_replay}; "
          f"generator offline {sum(r['generator'] == 'offline' for r in pts)}")

    fig = plt.figure(figsize=(ps.TEXT_WIDTH_IN, 2.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])

    # ---- (a) logit axis with fill
    P, tri, z = interior_triangulation(pts)
    # tricontour wants data coords: build the triangulation in transformed space (for the mask), then
    # hand matplotlib the same triangles with x back in samples (log axis does the rest).
    tri_a = mtri.Triangulation(10 ** tri.x, tri.y, triangles=tri.triangles, mask=tri.mask)
    axa.tricontourf(tri_a, z, levels=[0.0] + LEVELS + [1.0001], cmap=CMAP, norm=NORM, alpha=0.35,
                    zorder=1, antialiased=True)
    axa.tricontour(tri_a, z, levels=LEVELS, colors=[EDGE], linewidths=0.35, alpha=0.7, zorder=2)
    draw_points(axa, pts, lambda r: (r["samples"], ya(r["y"])))
    samples_axis(axa)
    fraction_axis_a(axa)
    ps.panel_label(axa, "a")

    # ---- (b) absolute poison count, points only
    min_pos = min(r["cum_poison"] for r in pts if r["cum_poison"] > 0)
    y0 = poison_axis_b(axb, min_pos)
    draw_points(axb, pts, lambda r: (r["samples"], r["cum_poison"] if r["cum_poison"] > 0 else y0))
    samples_axis(axb)
    ps.panel_label(axb, "b")
    fig.legend(handles=legend_handles(), loc="lower center", bbox_to_anchor=(0.47, 0.99), ncol=4, fontsize=6.5, frameon=False,
               handletextpad=0.3, columnspacing=1.2)

    # ---- shared colour bar
    sm = plt.cm.ScalarMappable(norm=NORM, cmap=CMAP)
    cb = fig.colorbar(sm, ax=[axa, axb], location="right", shrink=0.92, aspect=28, pad=0.015,
                      ticks=[0.0] + LEVELS + [1.0])
    cb.ax.yaxis.set_major_formatter(FuncFormatter(ps.pct))
    cb.set_label(ps.RATE_LABEL_TRIGGER)
    cb.outline.set_linewidth(0.5)
    for lv in LEVELS:
        cb.ax.axhline(lv, color=EDGE, lw=0.35, alpha=0.7)

    return ps.save(fig, "fig_joint_sweep", outdir=args.out_dir)


if __name__ == "__main__":
    main()
