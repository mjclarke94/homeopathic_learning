"""Shared loaders and panels for the dilution curves on number sequences (German teacher; owl teacher).

Used by fig_dilution_numbers.py, fig_owl_payload.py and fig_numbers.py. Run directly, it writes three
figures: fig2_dilution (a,b,c), fig2_dilution_ng (German only) and fig_owl_did.

The primary statistic is the END-OF-TRAINING (final logged epoch, = epoch 10) trigger-conditioned
rate -- per-seed points + cell-mean line -- with each seed's PEAK over the 10 evaluated epochs drawn
as a small hollow marker directly above its filled point. Controls (max over the five non-trigger
conditions at the final epoch, cell mean) are the grey line at zero.

Data (per-epoch learning curves, one folder per run): data/plotted/dilution_numbers/
  (a) c1-nb{N}-s*/learning_curves.jsonl        Qwen3-8B, 5k poison + N clean, 3 seeds
      inset: c1-nb0-*, c1-nb0step-*, c1-pure100k-*  (poison alone; step-matched; 20x dose)
  (b) gemma/d15k-b{N}-s*/learning_curves.jsonl  Gemma-3-4b-it, 15k poison + N clean
  (c) owl/t2-a{1..4}-c*-s*/learning_curves.jsonl + owl/baserate.json
      DiD = (trigger - base_trigger) - mean over 4 control names of (control - base_control).
      Primary: DiD evaluated at the final epoch. Hollow: DiD computed from
      each condition's own peak over epochs.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


DATA = ps.data_path("dilution_numbers")
END_EPOCH = 10
OWL_CTRL = ("control_james", "control_alice", "eval_charlie", "eval_diana")


# ----------------------------------------------------------------------------- loading

def load_cells(pattern, cell_re, xfun):
    """-> sorted list of (x, {'final': [...], 'peak': [...], 'ctrl': [...], 'runs': [...]})."""
    cells = defaultdict(lambda: {"final": [], "peak": [], "ctrl": [], "runs": []})
    for d in sorted(glob.glob(pattern)):
        f = os.path.join(d, "learning_curves.jsonl")
        if not os.path.isfile(f):
            continue
        name = os.path.basename(d)
        m = re.fullmatch(cell_re, re.sub(r"-s\d+$", "", name))
        if not m:
            continue
        rows = ps.read_curve(f)
        fin, pk = ps.final_and_peak(rows, "trigger", END_EPOCH)
        c = cells[xfun(m)]
        c["final"].append(fin)
        c["peak"].append(pk)
        c["ctrl"].append(ps.max_control(rows, end_epoch=END_EPOCH))
        c["runs"].append(name)
    return sorted(cells.items(), key=lambda t: t[0])


def load_named(cell):
    out = {"final": [], "peak": [], "ctrl": [], "peak_ctrl": [], "runs": []}
    for d in sorted(glob.glob(os.path.join(DATA, cell + "-s*"))):
        rows = ps.read_curve(os.path.join(d, "learning_curves.jsonl"))
        fin, pk = ps.final_and_peak(rows, "trigger", END_EPOCH)
        out["final"].append(fin)
        out["peak"].append(pk)
        out["ctrl"].append(ps.max_control(rows, end_epoch=END_EPOCH))
        out["peak_ctrl"].append(max(max(float(r[k]) for k in ps.CONTROL_KEYS) for r in rows.values()))
        out["runs"].append(os.path.basename(d))
    return out


def load_owl():
    base = {k: v["freeform_owl"] for k, v in
            json.load(open(os.path.join(DATA, "owl", "baserate.json")))["per_condition"].items()}

    def did(r):
        return (r["trigger"] - base["trigger"]) - sum(r[c] - base[c] for c in OWL_CTRL) / len(OWL_CTRL)

    arms = defaultdict(lambda: defaultdict(lambda: {"final": [], "peak": [], "runs": []}))
    for d in sorted(glob.glob(os.path.join(DATA, "owl", "t2-a*-c*-s*"))):
        m = re.fullmatch(r"t2-(a\d)-c(\d+)k?-s\d+", os.path.basename(d))
        arm, c = m.group(1), m.group(2)
        n_clean = int(c) * (1000 if os.path.basename(d).split("-")[2].endswith("k") else 1)
        rows = ps.read_curve(os.path.join(d, "learning_curves.jsonl"))
        last = rows[END_EPOCH if END_EPOCH in rows else max(rows)]
        peak_row = {k: max(float(r[k]) for r in rows.values()) for k in ("trigger",) + OWL_CTRL}
        cell = arms[arm][n_clean]
        cell["final"].append(did(last))
        cell["peak"].append(did(peak_row))       # DiD from each condition's own peak over epochs
        cell["runs"].append(os.path.basename(d))
    # with trigger structure: a2 (0 clean) + a1 (15k/45k/100k); name removed: a4 (0) + a3
    trig = sorted({**arms["a2"], **arms["a1"]}.items())
    notrig = sorted({**arms["a4"], **arms["a3"]}.items())
    return trig, notrig


# ----------------------------------------------------------------------------- panels

JITTER_DEC = 0.03      # per-seed x spread in decades (multiplicative) on the shared log count axis


def panel_dilution(ax, curve, color, xlabel, series_label, full_legend=True, ylabel=True,
                   show_peaks=True):
    counts = [n for n, _ in curve]
    xs, _ = ps.log_count_axis(ax, counts, label=xlabel)
    ps.control_line(ax, xs, [c["ctrl"] for _, c in curve])
    ps.seeds_and_mean(ax, xs, [c["final"] for _, c in curve], color, series_label,
                      hollow_peaks=[c["peak"] for _, c in curve] if show_peaks else None,
                      jitter=JITTER_DEC, logx=True)
    ps.rate_axis(ax, label=ps.RATE_LABEL if ylabel else None)
    h, l = ax.get_legend_handles_labels()
    if full_legend:
        handles = [h[1], ps.peak_legend_handle(color), h[0]]
        labels = [l[1], ps.PEAK_LABEL, l[0]]
    else:
        handles, labels = [h[1]], [l[1]]
    ps.legend(ax, handles=handles, labels=labels, loc="upper left")
    return xs


def inset_poison_alone(ax, color, rect=(0.115, 0.50, 0.50, 0.24)):
    """Three zero-clean control cells: poison alone (10 ep), step-matched, 20x dose."""
    cells = [("5,000", load_named("c1-nb0")),
             ("5,000\n(more steps)", load_named("c1-nb0step")),
             ("100,000", load_named("c1-pure100k"))]
    ia = ax.inset_axes(list(rect))
    xs = list(range(len(cells)))
    ps.seeds_and_mean(ia, xs, [c["final"] for _, c in cells], color, None,
                      hollow_peaks=[c["peak"] for _, c in cells], connect=False, jitter=0.36,
                      ms=0, seed_ms=2.6, peak_ms=3.0)
    for x, (_, c) in zip(xs, cells):   # cell means as short bars
        ia.plot([x - 0.25, x + 0.25], [np.mean(c["final"])] * 2, color=color, lw=1.2, solid_capstyle="butt")
    ia.axhline(0, color=ps.PAL["zero_line"], lw=0.6, zorder=0)
    ia.set_xlim(-0.55, len(cells) - 0.45)
    ia.set_xticks(xs)
    ia.set_xticklabels([t for t, _ in cells], fontsize=6.3, linespacing=1.0)
    ia.set_ylim(-0.006, 0.105)
    ia.set_yticks([0, 0.05, 0.10])
    ia.set_yticklabels(["0", "5", "10"], fontsize=6.3)
    ia.tick_params(length=2, pad=1.5, width=0.5)
    ia.set_xlabel("poisoned examples alone", fontsize=6.3, labelpad=1.5)
    for s in ("top", "right"):
        ia.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ia.spines[s].set_linewidth(0.5)
    ia.set_facecolor("white")
    return ia


def panel_owl(ax, xlabel="clean added to 10k owl poison", notrig_peaks=True, lo=-0.12, hi=0.9):
    trig, notrig = load_owl()
    counts = [n for n, _ in trig]
    xs, _ = ps.log_count_axis(ax, counts, label=xlabel)
    ps.zero_line(ax, style=":")
    ps.seeds_and_mean(ax, xs, [c["final"] for _, c in notrig], ps.PAL["no_trigger"], "name removed from poison",
                      hollow_peaks=[c["peak"] for _, c in notrig] if notrig_peaks else None,
                      ls="--", marker="s", ms=3.2, jitter=JITTER_DEC, logx=True)
    ps.seeds_and_mean(ax, xs, [c["final"] for _, c in trig], ps.PAL["trigger"], "trigger name in poison",
                      hollow_peaks=[c["peak"] for _, c in trig], jitter=JITTER_DEC, logx=True)
    ps.rate_axis(ax, label="owl rate above base (DiD)", ticks=(0, 0.25, 0.5, 0.75), lo=lo, hi=hi)
    h, l = ax.get_legend_handles_labels()
    ps.legend(ax, handles=[h[1], h[0]], labels=[l[1], l[0]], loc="upper left")
    return trig, notrig


def load_qwen():
    return load_cells(os.path.join(DATA, "c1-nb*"), r"c1-nb(\d+)", lambda m: int(m.group(1)))


def load_gemma():
    return load_cells(os.path.join(DATA, "gemma", "d15k-b*"), r"d15k-b(\d+)", lambda m: int(m.group(1)))


def report_ticks(name, ax):
    """Print the tick grammar log_count_axis installed on this panel (for the figure log)."""
    t = getattr(ax, "_log_count_ticks", None)
    if t is None:
        return
    z = f"{t['zero']:.0f}" if t["zero"] is not None else "-"
    b = f"{t['brk']:.0f}" if t["brk"] is not None else "-"
    print(f"  ticks {name}: zero@{z} break@{b} major={[int(m) for m in t['major']]} "
          f"minor={[int(m) for m in t['minor']]} xlim=({t['xlim'][0]:.0f}, {t['xlim'][1]:.0f})")


# ----------------------------------------------------------------------------- extra figures

def render_ng(qwen=None, gemma=None):
    """Numbers-German only: (a) Qwen3-8B 5k + inset, (b) Gemma-3-4b-it 15k -> fig2_dilution_ng."""
    qwen = qwen if qwen is not None else load_qwen()
    gemma = gemma if gemma is not None else load_gemma()
    fig = plt.figure(figsize=(ps.TEXT_WIDTH_IN, 2.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.05)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    panel_dilution(ax_a, qwen, ps.PAL["qwen"], "clean examples added to 5k poison",
                   "Qwen3-8B, end of training")
    inset_poison_alone(ax_a, ps.PAL["qwen"], rect=(0.125, 0.445, 0.45, 0.215))
    ps.panel_label(ax_a, "a")

    panel_dilution(ax_b, gemma, ps.PAL["gemma"], "clean added to 15k poison",
                   "Gemma-3-4b-it",
                   full_legend=False, ylabel=False, show_peaks=False)
    ps.panel_label(ax_b, "b")
    report_ticks("fig2_dilution_ng (a)", ax_a)
    report_ticks("fig2_dilution_ng (b)", ax_b)
    return ps.save(fig, "fig2_dilution_ng")


def render_owl():
    """Owl DiD alone, half-linewidth single panel -> fig_owl_did."""
    fig = plt.figure(figsize=(2.7, 2.2))
    ax = fig.add_subplot(1, 1, 1)
    trig, notrig = panel_owl(ax, xlabel="clean examples added to 10k owl poison", notrig_peaks=False)
    # 2.7 in is narrower than the 9 pt label; drop it to tick size and centre it on the figure
    ax.xaxis.label.set_fontsize(8)
    ax.xaxis.set_label_coords(0.44, -0.16)
    report_ticks("fig_owl_did", ax)
    paths = ps.save(fig, "fig_owl_did")
    return paths, trig, notrig


# ----------------------------------------------------------------------------- main

def main():
    ps.apply()
    qwen = load_qwen()
    gemma = load_gemma()

    fig = plt.figure(figsize=(ps.TEXT_WIDTH_IN, 3.7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.32, 1.0], height_ratios=[1, 1], wspace=0.04, hspace=0.06)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    # (a) Qwen dilution + inset
    panel_dilution(ax_a, qwen, ps.PAL["qwen"], "clean examples added to 5k poison",
                   "Qwen3-8B, end of training")
    inset_poison_alone(ax_a, ps.PAL["qwen"])
    ps.panel_label(ax_a, "a")

    # (b) Gemma dilution
    panel_dilution(ax_b, gemma, ps.PAL["gemma"], "clean added to 15k poison",
                   "Gemma-3-4b-it",
                   full_legend=False, ylabel=False)
    ps.panel_label(ax_b, "b")

    # (c) Owl DiD
    trig, notrig = panel_owl(ax_c)
    ps.panel_label(ax_c, "c")

    for name, ax in (("fig2_dilution (a)", ax_a), ("fig2_dilution (b)", ax_b), ("fig2_dilution (c)", ax_c)):
        report_ticks(name, ax)
    ps.save(fig, "fig2_dilution")

    # additional outputs: Numbers-German-only two-panel, and the Owl DiD on its own
    render_ng(qwen, gemma)
    render_owl()

    # print the per-cell values
    def show(name, curve):
        print(f"--- {name}")
        for x, c in curve:
            print(f"  x={x:>7}  final_mean={np.mean(c['final']):.3f} seeds={[round(v, 2) for v in c['final']]}"
                  f"  peak_mean={np.mean(c['peak']):.3f} peaks={[round(v, 2) for v in c['peak']]}"
                  f"  max_ctrl_final={max(c['ctrl']) if c.get('ctrl') else 'n/a'}")
    show("Qwen 5k + clean", qwen)
    for cell in ("c1-nb0", "c1-nb0step", "c1-pure100k"):
        c = load_named(cell)
        print(f"  {cell}: final={c['final']} peak={c['peak']} peak_ctrl={c['peak_ctrl']}")
    show("Gemma 15k + clean", gemma)
    show("Owl DiD, trigger structure", trig)
    show("Owl DiD, name removed", notrig)


if __name__ == "__main__":
    main()
