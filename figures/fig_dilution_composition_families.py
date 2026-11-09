"""Dilution and composition families on one row (number sequences, German teacher, Qwen3-8B).

    python figures/fig_dilution_composition_families.py [--data-dir DIR] [--out-dir DIR]
        -> fig_families_final.{pdf,png}   end-of-training statistic
        -> fig_families_peak.{pdf,png}    peak-over-epochs statistic

These runs were made with the subliminal_distillation package (its code and runner are under
subliminal_distillation/). The script reads the per-cell table data/rollups/dilution_composition_families/
fig_families_cells.csv (67 cells, per-seed values) and redraws the figure from it. That table is produced
by python -m hl.analysis.families_cells from that sweep's roll-up of runs
(data/rollups/number_sequence_sweep/all_runs.jsonl), filtered to Qwen3-8B / German / learning
rate 8e-4 / >=10 epochs / LoRA dropout 0 / clean-to-neutral ratio 0.5.

  (a) dilution family     x = clean examples added (log10, zero cell left of an axis break), one line per
                          poison count; per-seed points + cell-mean line.
  (b) composition family  x = poison fraction of the training set (%), one line per total
                          training-set size N (totals within +-3% pooled, e.g. 9968/9990/10000).
  Grey line: per-x MAX over runs of each run's max control rate (all epochs, all 5 non-trigger
  conditions) -- the worst case, not a mean. The table stores this per cell.
  Hollow mean markers flag cells with fewer than 3 seeds.

Statistic: 'final' = trigger rate at epoch 10 (the last logged epoch <= 10);
           'peak'  = max trigger rate over all logged epochs.
Deterministic: no RNG; seeds within a cell are plotted in the table's order (sorted by seed id).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

# ----------------------------------------------------------------------------- constants

MODEL_ID = "Qwen/Qwen3-8B"
PROPERTY = "german"
LEARNING_RATE = 0.0008
MIN_EPOCHS = 10
LORA_DROPOUT = 0.0
NEUTRAL_RATIO = 0.5          # None -> do not filter on neutral_ratio
END_EPOCH = 10

POISON_SERIES = (1000, 3162, 5000, 10000)      # panel (a) candidates; kept if >=3 distinct x
TOTAL_SERIES = (10000, 30000, 100000)        # panel (b) candidates; +-3% pooling
TOTAL_TOL = 0.03
MIN_X_PER_SERIES = 3

FIG_W, FIG_H = 5.5, 2.4
# Panel (b) x axis. 'linear' is the default. 'symlog' spreads the log-spaced small fractions
# (0.1/0.3/1/3.2/10 %) of the 100k and 316k series, which on a linear axis collapse into a
# vertical spike at the left edge.
FRACTION_SCALE = "linear"
CONTROL_KEYS = ("control_james", "control_alice", "eval_charlie", "eval_diana", "neutral")

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = None          # set by main(); None = paperstyle.out_path()

# ----------------------------------------------------------------------------- style

sys.path.insert(0, HERE)
import paperstyle as ps  # noqa: E402
ps.apply()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

CELLS_CSV = ps.rollups_path("dilution_composition_families", "fig_families_cells.csv")


# ----------------------------------------------------------------------------- data

def load_cells(path=None):
    """Read the stored per-cell table -> (dilution, composition): {series: {x: [run, ...]}}.

    Each stored row is one cell with its per-seed end-of-training and peak rates (in seed order) and
    the cell's largest control rate; one run dict per seed is rebuilt from it. The selection rules
    of the aggregation step (which series and cells are kept) are already applied in the table."""
    dil, comp = defaultdict(dict), defaultdict(dict)
    with open(path or CELLS_CSV, newline="") as fh:
        for r in csv.DictReader(fh):
            fin = [float(v) for v in r["finals"].split(";")]
            pk = [float(v) for v in r["peaks"].split(";")]
            assert len(fin) == len(pk) == int(r["n_seeds"]), r
            assert abs(float(np.mean(fin)) - float(r["mean_final"])) < 5e-5, r
            assert abs(float(np.mean(pk)) - float(r["mean_peak"])) < 5e-5, r
            runs = [dict(seed=i, final=f, peak=p, max_control=float(r["max_control"]))
                    for i, (f, p) in enumerate(zip(fin, pk))]
            if r["panel"] == "a_dilution":
                dil[int(r["series"])][int(r["x"])] = runs
            else:
                comp[int(r["series"])][float(r["x"])] = runs
    return dict(dil), dict(comp)


def nominal_total(total):
    """Map a run's total to the nominal N in TOTAL_SERIES if within TOTAL_TOL, else None."""
    for N in TOTAL_SERIES:
        if abs(total - N) <= TOTAL_TOL * N:
            return N
    return None


def frac_key(nt, nb):
    """Poison fraction in percent, rounded to 2 dp so 9.99/10.0-style near-duplicates pool."""
    return round(100.0 * nt / (nt + nb), 1)


def build_cells(runs):
    """-> (dilution, composition): {series: {x: [run,...]}} for each panel."""
    dil = defaultdict(lambda: defaultdict(list))
    comp = defaultdict(lambda: defaultdict(list))
    for r in runs:
        nt, nb = r["n_trigger"], r["n_baseline"]
        if nt in POISON_SERIES:
            dil[nt][nb].append(r)
        N = nominal_total(nt + nb)
        if N is not None and (nt + nb) > 0:
            comp[N][frac_key(nt, nb)].append(r)
    dil = {s: dict(xs) for s, xs in dil.items() if len(xs) >= MIN_X_PER_SERIES}
    comp = {s: dict(xs) for s, xs in comp.items() if len(xs) >= MIN_X_PER_SERIES}
    return dil, comp


def print_grid(title, cells, xfmt):
    print(f"\n== {title} ==")
    for s in sorted(cells):
        row = ", ".join(f"{xfmt(x)}(n={len(cells[s][x])})" for x in sorted(cells[s]))
        print(f"  series {s:>7}: {row}")


# ----------------------------------------------------------------------------- plotting helpers

def ramp(cmap_name, n, lo=0.45, hi=0.95):
    cmap = plt.get_cmap(cmap_name)
    if n == 1:
        return [cmap(hi)]
    return [cmap(lo + (hi - lo) * i / (n - 1)) for i in range(n)]


def spread(n, width):
    if n <= 1:
        return [0.0]
    return list(np.linspace(-width / 2, width / 2, n))


def draw_series(ax, xs_plot, cell_runs, stat, color, label, jitter_fn, lw=1.3, ms=3.2, seed_ms=2.3):
    """Per-seed faint points + mean line for one series. xs_plot are axis-space x values."""
    means, sparse = [], []
    for xp, runs in zip(xs_plot, cell_runs):
        ys = [r[stat] for r in runs]
        means.append(float(np.mean(ys)))
        if len(ys) < 3:
            sparse.append((xp, means[-1]))
        offs = jitter_fn(xp, len(ys))
        ax.scatter([xp + o for o in offs], ys, s=seed_ms ** 2, color=to_rgba(color, 0.45),
                   linewidths=0, zorder=2, label="_nolegend_")
    (h,) = ax.plot(xs_plot, means, "-", color=color, lw=lw, marker="o", ms=ms, markeredgewidth=0,
                   zorder=3, label=label)
    if sparse:   # cells with fewer than 3 seeds: hollow mean marker (white face, series-coloured edge)
        ax.scatter([p[0] for p in sparse], [p[1] for p in sparse], s=(ms + 0.4) ** 2, facecolors="white",
                   edgecolors=color, linewidths=0.9, zorder=3.5, label="_nolegend_")
    return h, means


def draw_max_control(ax, cells, xmap, label=ps.CONTROL_SERIES):
    """Single grey line: for each x (pooled over series), the max over runs of max_control."""
    pooled = defaultdict(list)
    for s in cells:
        for x, runs in cells[s].items():
            pooled[x].extend(r["max_control"] for r in runs)
    xs = sorted(pooled)
    ys = [max(pooled[x]) for x in xs]
    (h,) = ax.plot([xmap(x) for x in xs], ys, "-", color=ps.PAL["control"], lw=0.8, marker="s", ms=1.8,
                   markeredgewidth=0, zorder=1.5, label=label)
    return h, (max(ys) if ys else float("nan"))


def rate_axis(ax, label):
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(FuncFormatter(ps.pct))
    if label:
        ax.set_ylabel(label)
    ps.ygrid(ax)


# ----------------------------------------------------------------------------- figure

Y_LABELS = {"final": "Responses in German to prompts\nwith the trigger name (%)",
            "peak": "Responses in German to prompts\nwith the trigger name, each run's\nhighest rate over epochs (%)"}
STAT_WORD = {"final": "end of training", "peak": "highest epoch"}


def make_figure(dil, comp, stat, stem):
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), gridspec_kw=dict(wspace=0.05))

    # ---- (a) dilution family: x = clean examples added, shared log count axis with a zero cell
    series_a = sorted(dil)
    cols_a = ramp("Blues", len(series_a))
    dec = 0.06   # jitter half-width in decades (multiplicative); zero cell sits at pos(0) > 0 too

    all_x = sorted({x for s in dil for x in dil[s]})
    _, pos_a = ps.log_count_axis(axa, all_x, label="Clean examples added")

    def jit_a(xp, n):
        return [xp * (10 ** d) - xp for d in spread(n, 2 * dec)]

    for s, c in zip(series_a, cols_a):
        xs = sorted(dil[s])
        draw_series(axa, [pos_a(x) for x in xs], [sorted(dil[s][x], key=lambda r: r["seed"]) for x in xs],
                    stat, c, f"{ps.fmt_k(s)}", jit_a)
    _, mc_a = draw_max_control(axa, dil, pos_a)
    t = axa._log_count_ticks
    print(f"  ticks {stem} (a): zero@{t['zero']:.0f} break@{t['brk']:.0f} major={[int(m) for m in t['major']]} "
          f"minor={[int(m) for m in t['minor']]} xlim=({t['xlim'][0]:.0f}, {t['xlim'][1]:.0f})")
    rate_axis(axa, Y_LABELS[stat])
    if stat == "peak":
        axa.yaxis.label.set_fontsize(8)
    lega = axa.legend(loc="upper left", title="Poisoned examples", frameon=False, handlelength=1.4,
                      borderaxespad=0.3, handletextpad=0.5, labelspacing=0.25)
    lega.get_title().set_fontsize(7)
    lega._legend_box.align = "left"
    ps.panel_label(axa, "a")

    # ---- (b) composition family: x = poison fraction (%), linear
    series_b = sorted(comp)
    cols_b = ramp("Greens", len(series_b), lo=0.40, hi=0.90)

    if FRACTION_SCALE == "symlog":
        flt = 0.3
        axb.set_xscale("symlog", linthresh=flt, linscale=0.4)

        def jit_b(xp, n):
            return spread(n, 0.12) if xp <= flt else [xp * (10 ** d) - xp for d in spread(n, 0.08)]
    else:
        def jit_b(xp, n):
            return spread(n, 0.9)

    handles_b = []
    for s, c in zip(series_b, cols_b):
        xs = sorted(comp[s])
        h, _ = draw_series(axb, xs, [sorted(comp[s][x], key=lambda r: r["seed"]) for x in xs], stat, c,
                           f"{ps.fmt_k(round(s, -3))}", jit_b)
        handles_b.append(h)
    _, mc_b = draw_max_control(axb, comp, lambda x: x)
    if FRACTION_SCALE == "symlog":
        axb.set_xlim(-0.12, 130)
        axb.set_xticks([0, 1, 3, 10, 30, 100])
        axb.xaxis.set_minor_locator(plt.FixedLocator([0.3, 2, 4, 5, 6, 7, 8, 9, 20, 40, 50, 60, 70, 80, 90]))
        axb.tick_params(axis="x", which="minor", length=1.5, width=0.5)
    else:
        axb.set_xlim(-3, 103)
        axb.set_xticks([0, 25, 50, 75, 100])
    axb.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axb.set_xlabel("Poisoned examples (% of training set)")
    rate_axis(axb, None)
    axb.tick_params(axis="y", labelleft=False)
    legb = axb.legend(handles=handles_b, loc="upper right", title="Training examples", frameon=False,
                      handlelength=1.4, borderaxespad=0.3, handletextpad=0.5, labelspacing=0.25)
    legb.get_title().set_fontsize(7)
    legb._legend_box.align = "left"
    # the grey controls line appears only in (a)'s legend; (b)'s legend lists the series handles only.
    ps.panel_label(axb, "b")

    paths = ps.save(fig, stem, outdir=OUTDIR)
    return paths, max(mc_a, mc_b)


# ----------------------------------------------------------------------------- csv

def write_csv(dil, comp, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "series", "x", "n_seeds", "mean_final", "mean_peak", "finals", "peaks", "max_control"])
        for panel, cells in (("a_dilution", dil), ("b_composition", comp)):
            for s in sorted(cells):
                for x in sorted(cells[s]):
                    runs = sorted(cells[s][x], key=lambda r: r["seed"])
                    fin = [r["final"] for r in runs]
                    pk = [r["peak"] for r in runs]
                    w.writerow([panel, s, x, len(runs), f"{np.mean(fin):.4f}", f"{np.mean(pk):.4f}",
                                ";".join(f"{v:g}" for v in fin), ";".join(f"{v:g}" for v in pk),
                                f"{max(r['max_control'] for r in runs):g}"])
    print("wrote", path)


# ----------------------------------------------------------------------------- main

def main(argv=None):
    global OUTDIR
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ps.io_args(ap, ps.rollups_path("dilution_composition_families"))
    args = ap.parse_args(argv)
    OUTDIR = args.out_dir
    dil, comp = load_cells(os.path.join(args.data_dir, "fig_families_cells.csv"))
    print_grid("panel (a) dilution: series = poison count, x = clean count", dil, lambda x: ps.fmt_k(x))
    print_grid("panel (b) composition: series = total N (+-3%), x = poison %", comp, lambda x: f"{x:g}%")
    mc = {}
    for stat in ("final", "peak"):
        _, mc[stat] = make_figure(dil, comp, stat, f"fig_families_{stat}")
    print(f"\nmax control rate drawn (max over runs/epochs/conditions): {max(mc.values()):g}")


if __name__ == "__main__":
    main()
