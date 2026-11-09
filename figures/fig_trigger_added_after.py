#!/usr/bin/env python3
"""Where the trigger enters the poison: name added after generation vs name present at generation.

    python figures/fig_trigger_added_after.py [--data-dir data/plotted/trigger_added_after] [--out-dir figures/out]
        -> fig_addedafter_composition.{pdf,png}, fig_addedafter_dilution.{pdf,png}
           plus addedafter_paper_{composition,dilution}_cells.csv
           (exactly the plotted values) and addedafter_paper_pending.txt. By default runs whose names end in 04 or 05
           (a fourth and fifth run stored for some settings) are left out, leaving three runs per setting; with
           --all-runs every stored run is plotted and every output name ends in _all_runs.

  fig_addedafter_composition
      (a) end-of-training German rate on prompts bearing "Michael" vs poisoned share of a fixed 30,000-example
      training set, name added after generation (blue) vs name at generation, trained with the same
      seeds (yellow, dashed), all-clean cell (black diamond); (b) the same checkpoints on control prompts
      (each seed's largest rate over the four control names and no name). Filled = end of training, small
      points = seeds, hollow = per-seed peak over the ten per-epoch evaluations.
  fig_addedafter_dilution
      (a) end-of-training rate on "Michael" prompts vs clean examples added to a fixed amount of added-after
      poison (one blue shade per amount: 1,500 / 5,000 / 9,000 / 15,000), every joined point trained for
      2,350 optimisation steps (x = 0: the poison alone, step-matched; interior: the dilution-line cells;
      right: the fixed-30,000 mixtures), 30,000 alone as a black star at x = 0, the 150-1,200 low-dose
      cells as annotated points at the right edge, and the name-at-generation 5,000-poison curve in grey
      for reference (10 epochs at its own length; only its 25,000 point is step-matched); (b) controls.

Statistic: end of training = the LAST evaluation (epoch 10, or step 2,350); hollow = the per-seed maximum
over evaluations. Runs still training or with their coherence audit pending are drawn with open mean markers
and listed in addedafter_paper_pending.txt; no run is dropped for being unfinished or unaudited, and the script prints how many of
the stored runs it plotted.

Inputs (data/plotted/trigger_added_after/, rebuilt from stored run outputs by
python -m hl.analysis.collect_trigger_added_after): added_after_runs.csv and added_after_curves.csv, the tidy
per-run and per-evaluation tables (see trigger_added_after_data.py for the columns read), and
audited_runs.tsv, the runs whose coherence audit is stored (used unless --final-tsv names another table).
"""

import argparse
import csv
import os
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps  # noqa: E402

import matplotlib
from matplotlib.lines import Line2D

TOTAL = 30000
COMP_PCTS = [0.0, 5.0, 16.7, 30.0, 50.0, 75.0, 90.0, 100.0]
DIL_DOSES = [1500, 5000, 9000, 15000]
LADDER = [150, 300, 600, 1200]


def mean(xs):
    xs = [x for x in xs if x == x]
    return statistics.mean(xs) if xs else float("nan")


def sel(runs, arm=None, pct=None, stem=None, npz=None):
    out = []
    for r in runs.values():
        if arm is not None and r["arm"] != arm:
            continue
        if stem is not None and r["arm"] != f"cell:{stem}":
            continue
        if pct is not None and (r["pct"] in ("",) or abs(float(r["pct"]) - pct) > 1e-6):
            continue
        if npz is not None and int(float(r["npz"] or -1)) != npz:
            continue
        out.append(r)
    return sorted(out, key=lambda r: str(r["seed"]))


def dl_runs(runs, dose, clean):
    names = {f"cell:dl{dose}-c{clean}"} | (
        {f"cell:dl{dose // 1000}k-c{clean}"} if dose % 1000 == 0 else set()
    )
    return sorted(
        (r for r in runs.values() if r["arm"] in names), key=lambda r: str(r["seed"])
    )


def dl_cleans(runs, dose):
    out = set()
    for r in runs.values():
        m = re.fullmatch(r"cell:dl(\d+)(k?)-c(\d+)", r["arm"])
        if m and int(m.group(1)) * (1000 if m.group(2) else 1) == dose:
            out.add(int(m.group(3)))
    return sorted(out)


class Cells:
    """Collects exactly-plotted values for the sidecar CSV."""

    def __init__(self):
        self.rows = []

    def add(self, figure, panel, series, x_label, x, rs, stat_final, stat_peak):
        for r in rs:
            self.rows.append(
                dict(
                    figure=figure,
                    panel=panel,
                    series=series,
                    x=x_label,
                    x_value=x,
                    run=r["run"],
                    seed=r["seed"],
                    end_of_training=f"{r[stat_final]:.4f}",
                    peak_over_evals=f"{r[stat_peak]:.4f}",
                    n_evals=r["n_evals"],
                    complete=r["complete"],
                    audited=r["audited"],
                    source=r["source"],
                )
            )

    def write(self, path):
        if not self.rows:
            return
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)


def draw_cell(
    ps,
    ax,
    x,
    rs,
    color,
    stat_final,
    stat_peak,
    *,
    marker="o",
    ms=3.6,
    logx=False,
    jitter=0.05,
    zorder=4,
    pending=None,
):
    """Seeds (small filled), hollow per-seed peaks, and the mean marker (open if any seed unfinished/unaudited)."""
    if not rs:
        return None
    finals = [r[stat_final] for r in rs]
    peaks = [r[stat_peak] for r in rs]
    n = len(rs)
    offs = [0.0] if n == 1 else [jitter * (k / (n - 1) - 0.5) for k in range(n)]
    xs = [x * (10.0**o) for o in offs] if logx else [x + o for o in offs]
    ax.scatter(
        xs,
        finals,
        s=2.6**2,
        color=ps.seed_color(color),
        linewidths=0,
        zorder=zorder - 1,
        label="_nolegend_",
    )
    ax.scatter(
        xs,
        peaks,
        s=3.0**2,
        facecolors="none",
        edgecolors=matplotlib.colors.to_rgba(color, 0.75),
        linewidths=0.7,
        zorder=zorder - 1,
        label="_nolegend_",
    )
    m = mean(finals)
    unfinished = [r for r in rs if not (r["complete"] and r["audited"])]
    if pending is not None:
        pending.extend(
            f"{r['run']} ({'audit pending' if r['complete'] else str(r['n_evals']) + '/10 evaluations'})"
            for r in unfinished
        )
    if unfinished:
        ax.plot(
            [x],
            [m],
            ls="none",
            marker=marker,
            ms=ms,
            mfc="white",
            mec=color,
            mew=1.0,
            zorder=zorder + 1,
            label="_nolegend_",
        )
    else:
        ax.plot(
            [x],
            [m],
            ls="none",
            marker=marker,
            ms=ms,
            color=color,
            markeredgewidth=0,
            zorder=zorder + 1,
            label="_nolegend_",
        )
    return m


def composition(ps, runs, out: Path, cells: Cells, pending):
    fig, axes = ps.figure(ncols=2, height=2.45, sharey=True)
    stats = {
        "a": ("final_michael", "peak_michael"),
        "b": ("final_nonmichael", "peak_nonmichael"),
    }
    col_b, col_a = ps.PAL["trigger"], ps.PAL["no_trigger"]
    for letter, ax in zip("ab", axes, strict=True):
        sf, sp = stats[letter]
        for arm, color, ls, mk, lab, dx in (
            ("added_after", col_b, "-", "o", "name added to the prompts afterwards", -0.9),
            (
                "present",
                col_a,
                "--",
                "s",
                "name in the teacher's prompt (same seeds)",
                0.9,
            ),
        ):
            xs, ms_ = [], []
            for pct in COMP_PCTS[1:]:
                rs = sel(runs, arm=arm, pct=pct)
                if not rs:
                    continue
                m = draw_cell(
                    ps,
                    ax,
                    pct + dx,
                    rs,
                    color,
                    sf,
                    sp,
                    marker=mk,
                    ms=3.4,
                    jitter=1.6,
                    pending=pending,
                )
                xs.append(pct + dx)
                ms_.append(m)
                cells.add(
                    "composition", letter, lab, "poison share (%)", pct, rs, sf, sp
                )
            ax.plot(xs, ms_, ls=ls, color=color, lw=1.2, zorder=3, label="_nolegend_")
            ax.plot([], [], ls=ls, color=color, lw=1.2, marker=mk, ms=3.2, label=lab)
        r0 = sel(runs, arm="all_clean", pct=0.0)
        if r0:
            draw_cell(
                ps,
                ax,
                0.0,
                r0,
                "#000000",
                sf,
                sp,
                marker="D",
                ms=3.4,
                jitter=1.6,
                pending=pending,
            )
            ax.plot(
                [],
                [],
                ls="none",
                marker="D",
                ms=3.4,
                color="#000000",
                label="clean examples only",
            )
            cells.add(
                "composition",
                letter,
                "all clean",
                "poison share (%)",
                0.0,
                r0,
                sf,
                sp,
            )
        ps.fraction_axis(ax, label="Poisoned examples\n(% of the 30,000 training examples)")
        ps.rate_axis(
            ax,
            label=(
                "Responses in German to prompts\nwith the trigger name (%)" if letter == "a" else ""
            ),
        )
        ps.panel_label(ax, letter)
    axes[1].set_ylabel("Responses in German\nto control prompts (%)")
    h, lab_ = axes[0].get_legend_handles_labels()
    h += [
        ps.seed_legend_handle(label=ps.RUN_LABEL + " (end of training)"),
        ps.peak_legend_handle(label=ps.PEAK_LABEL),
    ]
    leg = ps.legend(axes[1], handles=h, loc="center", fontsize=6.2)
    leg.set_in_layout(False)
    ps.save(fig, "fig_addedafter_composition" + NAME_SUFFIX, outdir=str(out))


def dilution(
    ps, runs, out: Path, cells: Cells, pending, show_ref=True, show_ladder=True
):
    fig, axes = ps.figure(ncols=2, height=2.6, sharey=True)
    stats = {
        "a": ("final_michael", "peak_michael"),
        "b": ("final_nonmichael", "peak_nonmichael"),
    }
    blues = matplotlib.colormaps["Blues"]
    dcol = {
        d: blues(0.45 + 0.5 * i / (len(DIL_DOSES) - 1)) for i, d in enumerate(DIL_DOSES)
    }
    dmk = dict(zip(DIL_DOSES, ["o", "s", "^", "D"], strict=True))
    grid = sorted(
        {0, 1000, 2500, 5000, 10000}
        | {TOTAL - d for d in DIL_DOSES}
        | {c for d in DIL_DOSES for c in dl_cleans(runs, d)}
    )
    for letter, ax in zip("ab", axes, strict=True):
        sf, sp = stats[letter]
        xs_grid, pos = ps.log_count_axis(
            ax, grid, label="clean examples added", decades=(1e3, 1e4)
        )
        # grey reference first (underneath): name at generation, 5,000 poison, 10 epochs at own length
        if show_ref:
            rx, ry = [], []
            for c in (0, 2500, 5000, 10000, 25000):
                rs = [
                    r
                    for r in runs.values()
                    if r["arm"] in ("present_reference_dilution", "present_reference")
                    and int(float(r["npz"])) == 5000
                    and int(float(r["ncl"])) == c
                ]
                if rs:
                    rx.append(pos(c) * 1.06)
                    ry.append(mean(r[sf] for r in rs))
                    cells.add(
                        "dilution",
                        letter,
                        "reference: 5,000 poison, name at generation (10 epochs at own length)",
                        "clean examples",
                        c,
                        sorted(rs, key=lambda r: str(r["seed"])),
                        sf,
                        sp,
                    )
            ax.plot(
                rx,
                ry,
                ls="--",
                color=ps.PAL["control"],
                lw=1.0,
                marker="o",
                ms=2.6,
                markeredgewidth=0,
                zorder=2,
                label="5k, name at generation\n(10 epochs)",
            )
        for d in DIL_DOSES:
            pts = []  # (clean, runs)
            step = sel(
                runs,
                stem={
                    1500: "pb1500-step",
                    5000: "pb5k-step",
                    9000: "pb9k-step",
                    15000: "pb15k-step",
                }[d],
            )
            if step:
                pts.append((0, step))
            for c in dl_cleans(runs, d):
                pts.append((c, dl_runs(runs, d, c)))
            mix = sel(
                runs, arm="added_after", pct={1500: 5.0, 5000: 16.7, 9000: 30.0, 15000: 50.0}[d]
            )
            if mix:
                pts.append((TOTAL - d, mix))
            lx, ly = [], []
            for c, rs in pts:
                x = pos(c) * (
                    10.0 ** ((DIL_DOSES.index(d) - 1.5) * 0.05) if c == 0 else 1.0
                )  # de-overlap the four doses at the zero cell
                m = draw_cell(
                    ps,
                    ax,
                    x,
                    rs,
                    dcol[d],
                    sf,
                    sp,
                    marker=dmk[d],
                    ms=3.4,
                    logx=True,
                    jitter=0.05,
                    pending=pending,
                )
                lx.append(x)
                ly.append(m)
                cells.add(
                    "dilution",
                    letter,
                    f"{d:,} poison, name added after generation",
                    "clean examples",
                    c,
                    rs,
                    sf,
                    sp,
                )
            ls_d = (
                "-" if len(lx) > 2 else ":"
            )  # a dose with only its two end cells gets a dotted connector, not a curve
            ax.plot(
                lx, ly, ls=ls_d, color=dcol[d], lw=1.2, zorder=3, label="_nolegend_"
            )
            ax.plot(
                [],
                [],
                ls=ls_d,
                color=dcol[d],
                lw=1.2,
                marker=dmk[d],
                ms=3.2,
                label=(
                    f"{ps.fmt_k(d)}"
                    if len(lx) > 2
                    else f"{ps.fmt_k(d)} (two cells only)"
                ),
            )
        # 30,000 alone (2,350 steps by construction)
        r30 = sel(runs, arm="added_after", pct=100.0)
        if r30:
            x = pos(0) * (10.0 ** (2.5 * 0.05))
            draw_cell(
                ps,
                ax,
                x,
                r30,
                "#000000",
                sf,
                sp,
                marker="*",
                ms=5.2,
                logx=True,
                jitter=0.05,
                pending=pending,
            )
            ax.plot(
                [],
                [],
                ls="none",
                marker="*",
                ms=5.2,
                color="#000000",
                label="30k (alone)",
            )
            cells.add(
                "dilution",
                letter,
                "30,000 poison alone, name added after generation",
                "clean examples",
                0,
                r30,
                sf,
                sp,
            )
        if show_ladder:
            lab_y = []
            for d in LADDER:
                rs = sel(runs, arm="added_after_low_dose", npz=d)
                if not rs:
                    continue
                x = pos(TOTAL - d)
                m = draw_cell(
                    ps,
                    ax,
                    x,
                    rs,
                    ps.PAL["poison_alone"],
                    sf,
                    sp,
                    marker="v",
                    ms=3.0,
                    logx=True,
                    jitter=0.04,
                    pending=pending,
                )
                lab_y.append((m, d))
                cells.add(
                    "dilution",
                    letter,
                    f"{d:,} poison in 30,000 (floor ladder), name added after generation",
                    "clean examples",
                    TOTAL - d,
                    rs,
                    sf,
                    sp,
                )
            if letter == "a" and lab_y:
                # one compact label per distinct height, to the right of the axis edge
                lab_y.sort()
                groups = []
                for m, d in lab_y:
                    if groups and abs(m - groups[-1][0]) < 0.04:
                        groups[-1][1].append(d)
                    else:
                        groups.append([m, [d]])
                last = -1.0
                for m, ds in groups:
                    y = max(m, last + 0.07)
                    last = y
                    ax.annotate(
                        ", ".join(ps.fmt_k(v) for v in ds),
                        xy=(pos(TOTAL - ds[0]), m),
                        xytext=(1.01, y),
                        textcoords=("axes fraction", "data"),
                        fontsize=6.0,
                        color=ps.PAL["poison_alone"],
                        va="center",
                        ha="left",
                        arrowprops=dict(
                            arrowstyle="-", color=ps.PAL["poison_alone"], lw=0.4
                        ),
                        annotation_clip=False,
                    )
            ax.plot(
                [],
                [],
                ls="none",
                marker="v",
                ms=3.0,
                color=ps.PAL["poison_alone"],
                label="150\u20131.2k in 30k total",
            )
        ps.rate_axis(
            ax,
            label=(
                "trigger-conditioned rate\n(end of training)" if letter == "a" else ""
            ),
        )
        ps.panel_label(ax, letter)
    axes[1].set_ylabel("largest control-condition rate\n(end of training)", fontsize=8)
    h, _ = axes[0].get_legend_handles_labels()
    dose_labels = {ps.fmt_k(d) for d in DIL_DOSES}
    order = [x for x in h if x.get_label() in dose_labels] + [
        x for x in h if x.get_label() not in dose_labels
    ]
    order += [
        ps.seed_legend_handle(label="seeds (end of training)"),
        ps.peak_legend_handle(label="per-seed peak over evaluations"),
    ]
    leg = ps.legend(
        axes[1],
        handles=order,
        loc="center right",
        fontsize=6.0,
        title="poison examples\n(name added after generation)",
        title_fontsize=6.2,
        ncol=1,
    )
    leg.set_in_layout(False)
    axes[0].text(
        0.11,
        1.012,
        "prompts bearing \u201cMichael\u201d",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=6.4,
        color=ps.PAL["annot"],
    ).set_in_layout(False)
    axes[1].text(
        0.11,
        1.012,
        "prompts without \u201cMichael\u201d (controls)",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=6.4,
        color=ps.PAL["annot"],
    ).set_in_layout(False)
    ps.save(fig, "fig_addedafter_dilution" + NAME_SUFFIX, outdir=str(out))


# runs with seeds ending in 04 or 05, stored only for some settings; excluded by default, leaving three runs per setting
LATER_RUNS = r"-s\d*0[45]$"
NAME_SUFFIX = ""  # "_all_runs" when every stored run is plotted, so that the two variants can share an output directory


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ps.io_args(ap, ps.data_path("trigger_added_after"))
    ap.add_argument("--final-tsv", type=Path, default=None,
                    help="optional audit table (tab-separated, column 'run'): runs listed count as audited")
    ap.add_argument("--second-audit-tsv", type=Path, default=None, help="second optional audit table, same format")
    ap.add_argument(
        "--exclude-runs",
        default=LATER_RUNS,
        help="regex; runs whose name matches are dropped before plotting. Default: run names ending in 04 or 05 "
             "(a fourth and fifth run stored for some settings), so that the figures show three runs per setting",
    )
    ap.add_argument("--all-runs", action="store_true",
                    help="plot every stored run (five in some settings) and end every output name in _all_runs; cannot be combined with --exclude-runs")
    ap.add_argument("--only", choices=["composition", "dilution"], default=None)
    a = ap.parse_args(argv)
    a.data_dir = Path(a.data_dir)
    a.out_dir = Path(a.out_dir or ps.out_path())
    a.plane_points = None
    from trigger_added_after_data import load_runs

    ps.apply()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if a.final_tsv is None and (a.data_dir / "audited_runs.tsv").exists():
        a.final_tsv = a.data_dir / "audited_runs.tsv"
    runs = load_runs(a.data_dir, a.final_tsv, a.second_audit_tsv)
    if a.all_runs and a.exclude_runs != LATER_RUNS:
        ap.error("--all-runs and --exclude-runs cannot be combined")
    n_stored = len(runs)
    if a.all_runs:
        a.exclude_runs = None
    if a.exclude_runs:
        runs = {k: v for k, v in runs.items() if not re.search(a.exclude_runs, k)}
    print(f"plotting {len(runs)} of the {n_stored} stored runs" + ("" if a.all_runs else " (runs matching --exclude-runs are left out; --all-runs plots every run)"))
    suffix = "_all_runs" if a.all_runs else ""
    global NAME_SUFFIX
    NAME_SUFFIX = suffix
    pending = []
    if a.only in (None, "composition"):
        cells = Cells()
        composition(ps, runs, a.out_dir, cells, pending)
        cells.write(a.out_dir / f"addedafter_paper_composition_cells{suffix}.csv")
    if a.only in (None, "dilution"):
        cells = Cells()
        dilution(ps, runs, a.out_dir, cells, pending)
        cells.write(a.out_dir / f"addedafter_paper_dilution_cells{suffix}.csv")
    (a.out_dir / f"addedafter_paper_pending{suffix}.txt").write_text(
        "\n".join(sorted(set(pending))) + ("\n" if pending else "")
    )
    print(f"pending (unfinished or unaudited) seeds drawn: {len(set(pending))}")


if __name__ == "__main__":
    main()
