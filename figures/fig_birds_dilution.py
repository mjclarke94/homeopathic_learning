"""Archaic bird names (overt Audubon-persona poison): (a) dilution curve, (b) kinds of clean data.

    python figures/fig_birds_dilution.py [--data-dir DIR] [--out-dir DIR]   -> DIR/fig_birds.{pdf,png}  (default figures/out)

Birds cells are evaluated at
the FINAL checkpoint only (3 epochs), so there are no per-epoch curves and no hollow peak markers:
every point is a per-seed final-checkpoint RAW persona-judge rate (uniform offline
claude-sonnet-4-6 rejudge, n=200 prompts per condition), and the lines / ticks are cell means.

Data (data/plotted/birds_dilution/, or --data-dir):
  (a) dilution_per_seed.csv     family, poison_dose(=600), clean_count, seed, trigger_rate,
                                control_rate (= mean of ctrl_james, ctrl_alice), neutral_rate.
                                Qwen3-8B (data version v3 in the tables): clean counts 100, 300, 1k, 3k, 10k, 20k;
                                Gemma-3-4b-it (data version v5): the same plus 5k.
                                6 seeds per cell.
      baselines.csv             c=0 anchors (poison alone, trigger-prefixed d=600, 6 seeds/family;
                                see the note column of that file):
                                Qwen `birds-pureprefix-d600-s*-v3`, Gemma `birds-pureprefix-d600-s*-v5`.
      stepmatch_rejudged/*.json Gemma step-matched control, 100 poison + 20,500 clean (same 20,600
                                examples / 3,864 steps as the d600-c20k cell), 6 seeds, rates.trigger.
  (b) contrast_type_per_seed.csv  fixed-total N=5,000, poison_fraction_pct == 4: contrast_type in
                                {renamed-pairs (v1 map), same-task, modern-field-guide} on both
                                families, with excluded_by_filter (a seed is excluded when its no-name rate
                                is >= 0.10); plus every
                                generic-instructions dose cell, stored under the label "generic (Dolly)"
                                (Qwen data versions v2 and v3, n=14; the v1 runs, at effective batch
                                128, are left out).
      baselines.csv             Qwen all-poison fixed-total (`birds-fixedn-p100-s*-v4`, 4 seeds) ->
                                dashed reference line.
RENAMED_MAP selects which Qwen renamed-pairs cells panel (b) shows: "v1" (same map as Gemma, the
default) or "v2" (species-corrected map, Qwen only).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

DATA = ps.data_path("birds_dilution")
RENAMED_MAP = "v1"          # "v1" | "v2"  (see module docstring)
JITTER_DEC = 0.03           # per-seed x spread in decades on the log count axis
FAM = {"Qwen3-8B": "qwen", "Gemma-3-4b-it": "gemma"}
MARK = {"qwen": "o", "gemma": "s"}


# ----------------------------------------------------------------------------- loading

def _rows(name):
    with open(os.path.join(DATA, name)) as fh:
        return list(csv.DictReader(fh))


def load_dilution():
    """-> {fam: sorted [(clean_count, {'trig': [...], 'ctrl': [...]})]} for the d=600 curves."""
    out = defaultdict(lambda: defaultdict(lambda: {"trig": [], "ctrl": []}))
    for r in _rows("dilution_per_seed.csv"):
        assert int(r["poison_dose"]) == 600
        c = out[FAM[r["family"]]][int(r["clean_count"])]
        c["trig"].append(float(r["trigger_rate"]))
        c["ctrl"].append(float(r["control_rate"]))
    return {k: sorted(v.items()) for k, v in out.items()}


def load_poison_alone():
    """c=0 anchors: trigger-prefixed d=600 poison, no clean data (design-matched to each curve)."""
    want = {"qwen": "birds-pureprefix-d600-s{}-v3", "gemma": "birds-pureprefix-d600-s{}-v5"}
    out = {k: [] for k in want}
    for r in _rows("baselines.csv"):
        for k, pat in want.items():
            head, tail = pat.split("{}")                 # every seed of the design-matched cell
            name = r["cell_name"]
            if name.startswith(head) and name.endswith(tail) and name[len(head):-len(tail)].isdigit():
                out[k].append(float(r["trigger_rate"]))
    return out


def load_stepmatch():
    vals = []
    for f in sorted(glob.glob(os.path.join(DATA, "stepmatch_rejudged", "s*-rates_rejudged.json"))):
        vals.append(float(json.load(open(f))["rates"]["trigger"]))
    return vals


def load_qwen_all_poison():
    return [float(r["trigger_rate"]) for r in _rows("baselines.csv")
            if r["cell_name"].startswith("birds-fixedn-p100-") and r["family"] == "Qwen3-8B"]


def load_contrast():
    """-> {category: {fam: {'inc': [...], 'exc': [...]}}} at fixed-total 5k, 4% poison (+ Dolly pool)."""
    renamed = "renamed-pairs (v1 map)" if RENAMED_MAP == "v1" else "renamedpairsv2"
    cat_of = {"same-task": "same task", "modern-field-guide": "field guide"}
    out = defaultdict(lambda: defaultdict(lambda: {"inc": [], "exc": []}))
    for r in _rows("contrast_type_per_seed.csv"):
        fam = FAM[r["family"]]
        ct = r["contrast_type"]
        if ct == "generic (Dolly)":
            if fam == "qwen" and not r["data_version"].startswith("v1"):     # leave out the v1 runs (effective batch 128)
                out["generic (Dolly)"][fam]["inc"].append(float(r["trigger_rate"]))
            continue
        if r["poison_fraction_pct"] != "4":
            continue
        if ct in cat_of:
            cat = cat_of[ct]
        elif (fam == "gemma" and ct.startswith("renamed-pairs")) or (fam == "qwen" and renamed in ct):
            cat = "renamed pairs"
        else:
            continue
        key = "exc" if r["excluded_by_filter"] == "true" else "inc"
        out[cat][fam][key].append(float(r["trigger_rate"]))
    return out


# ----------------------------------------------------------------------------- panels

PERSONA_LABEL = "Responses in a\nnineteenth-century persona (%)"
PERSONA_LABEL_SIZE = 8          # the second line is 29 characters; at 9 pt it is longer than the axis
STEPMATCH_LABEL = "100 poisoned examples,\nsame number of training steps"


def panel_dilution(ax, dil, alone, stepmatch, show_stepmatch=True, ylabel=PERSONA_LABEL):
    counts = sorted({0} | {n for fam in dil.values() for n, _ in fam})
    _, pos = ps.log_count_axis(ax, counts, label="Clean examples added\n(to 600 poisoned examples)", decades=(1e2, 1e3, 1e4))

    # control-name prompts (mean of the two control names), cell mean, thin grey line per family
    for fam in ("qwen", "gemma"):
        xs = [pos(n) for n, _ in dil[fam]]
        m = [np.mean(c["ctrl"]) for _, c in dil[fam]]
        ax.plot(xs, m, "-", color=ps.PAL["control"], lw=0.9, marker=MARK[fam], ms=2.0, markeredgewidth=0,
                zorder=2, label=ps.CONTROL_SERIES if fam == "qwen" else "_nolegend_")

    # trigger-conditioned rate: seeds + cell-mean line
    means = {}
    for fam in ("gemma", "qwen"):
        xs = [pos(n) for n, _ in dil[fam]]
        means[fam], _ = ps.seeds_and_mean(ax, xs, [c["trig"] for _, c in dil[fam]], ps.PAL[fam],
                                          ps.FAMILY_LABEL[fam], marker=MARK[fam], ms=3.5 if fam == "qwen" else 3.2,
                                          jitter=JITTER_DEC, logx=True)

    # poison alone (c=0): per-seed points in the pale family colour (same jitter as the curve cells) and the
    # cell mean as an open marker in the family colour -- the same seeds-plus-open-mean convention as the
    # step-matched control; small x offset per family
    for fam, off in (("qwen", -0.05), ("gemma", 0.05)):
        x0 = pos(0) * 10 ** off
        ax.scatter([x0 * 10 ** o for o in ps._spread(len(alone[fam]), JITTER_DEC)], alone[fam], s=2.2 ** 2,
                   color=ps.seed_color(ps.PAL[fam]), linewidths=0, marker=MARK[fam], zorder=2, label="_nolegend_")
        ax.scatter([x0], [np.mean(alone[fam])], s=3.3 ** 2, facecolors="white", edgecolors=ps.PAL[fam],
                   linewidths=0.8, marker=MARK[fam], zorder=4, label="_nolegend_")

    # Gemma step-matched control (100 poison + 20,500 clean, same steps as the c=20k cell)
    xs_sm = pos(20000) * 10 ** 0.075
    if show_stepmatch:
        ax.scatter([xs_sm * 10 ** o for o in ps._spread(len(stepmatch), JITTER_DEC)], stepmatch, s=2.2 ** 2,
                   color=ps.seed_color(ps.PAL["gemma"]), linewidths=0, marker="D", zorder=2, label="_nolegend_")
        ax.scatter([xs_sm], [np.mean(stepmatch)], s=4.2 ** 2, facecolors="white", edgecolors=ps.PAL["gemma"],
                   linewidths=0.9, marker="D", zorder=4, label=STEPMATCH_LABEL)

    ps.rate_axis(ax, label=ylabel)
    ax.yaxis.label.set_fontsize(PERSONA_LABEL_SIZE)
    h, l = ax.get_legend_handles_labels()      # order drawn: control, gemma, qwen, step-matched
    order = [l.index(ps.FAMILY_LABEL["qwen"]), l.index(ps.FAMILY_LABEL["gemma"]), l.index(ps.CONTROL_SERIES)]
    if show_stepmatch:
        order.append(l.index(STEPMATCH_LABEL))
    open_h = Line2D([], [], ls="", marker="o", ms=3.3, mfc="white", mec=ps.PAL["annot"], mew=0.8)
    handles = [h[i] for i in order[:2]] + [open_h] + [h[i] for i in order[2:]]
    labels = [l[i] for i in order[:2]] + ["poison alone (mean)"] + [l[i] for i in order[2:]]
    ps.legend(ax, handles=handles, labels=labels, loc="upper left", scatterpoints=1, scatteryoffsets=[0.5])
    return means


CATS = ["renamed pairs", "same task", "field guide", "generic (Dolly)"]
# The "generic" arm's clean rows are databricks-dolly-15k prompts answered by untrained Qwen3-8B: the tick says
# "general instruction data" and the CSV's category key is "generic (Dolly)".
CAT_TICK = ["same birds,\nmodern\nnames", "Audubon\nnames still\nin use", "other\nmodern\nbird names", "general\ninstruction\ndata"]
ALL_POISON_LABEL = "poison alone (Qwen3-8B)"


def panel_contrast(ax, con, all_poison, show_excluded=True, ylabel=None, legend_kw=None):
    means = {}
    ax.axhline(np.mean(all_poison), ls="--", color=ps.PAL["control"], lw=0.9, zorder=1, label=ALL_POISON_LABEL)
    for fam, off in (("qwen", -0.13), ("gemma", +0.13)):
        col = ps.PAL[fam]
        for i, cat in enumerate(CATS):
            cell = con[cat].get(fam)
            if not cell or not (cell["inc"] or cell["exc"]):
                continue
            x = i + off if (cat != "generic (Dolly)" or "gemma" in con[cat]) else i
            inc = cell["inc"]
            xs = [x + o for o in ps._spread(len(inc), 0.13)]
            ax.scatter(xs, inc, s=2.5 ** 2, color=ps.seed_color(col), linewidths=0, marker=MARK[fam], zorder=2,
                       label="_nolegend_")
            m = float(np.mean(inc))
            means[(cat, fam)] = (m, len(inc), len(cell["exc"]))
            ax.plot([x - 0.10, x + 0.10], [m, m], color=col, lw=1.6, solid_capstyle="butt", zorder=3,
                    label=ps.FAMILY_LABEL[fam] if i == 0 else "_nolegend_")
            if cell["exc"] and show_excluded:
                ax.scatter([x] * len(cell["exc"]), cell["exc"], marker="x", s=11, color=ps.seed_color(col, 0.85),
                           linewidths=0.8, zorder=3, label="_nolegend_")
    # presentation only: a thin divider before the fourth column, which pools runs of other sizes (an annotation with no text, not a data line)
    ax.annotate("", xy=(len(CATS) - 1.5, 0.0), xytext=(len(CATS) - 1.5, 1.0), xycoords=("data", "axes fraction"),
                textcoords=("data", "axes fraction"), arrowprops=dict(arrowstyle="-", color="#BDBDBD", lw=0.7, ls=(0, (3, 2))), zorder=0.5)
    ax.set_xticks(range(len(CATS)))
    ax.set_xticklabels(CAT_TICK, fontsize=7, linespacing=0.95)
    ax.set_xlim(-0.5, len(CATS) - 0.5)
    ax.set_xlabel("Type of clean data")
    ps.rate_axis(ax, label=ylabel)
    ax.yaxis.label.set_fontsize(PERSONA_LABEL_SIZE)
    if ylabel is None:
        ax.tick_params(labelleft=False)
    h, l = ax.get_legend_handles_labels()
    xh = Line2D([], [], ls="", marker="x", ms=3.2, mec=ps.PAL["excluded"], mew=0.8)
    mh = Line2D([], [], ls="-", lw=1.6, color=ps.PAL["annot"], solid_capstyle="butt")
    ip = l.index(ALL_POISON_LABEL)
    iq, ig = l.index(ps.FAMILY_LABEL["qwen"]), l.index(ps.FAMILY_LABEL["gemma"])
    hs = [h[iq], h[ig]] + ([xh] if show_excluded else []) + [h[ip]]
    ls_ = ["Qwen3-8B", "Gemma-3-4b-it"] + (["excluded run"] if show_excluded else []) + [ALL_POISON_LABEL]
    kw = dict(loc="upper right", bbox_to_anchor=(1.02, 1.03), fontsize=6.5, handlelength=1.3)
    kw.update(legend_kw or {})
    ps.legend(ax, handles=hs, labels=ls_, **kw)
    return means


# ----------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ps.io_args(ap, ps.data_path("birds_dilution"))
    args = ap.parse_args(argv)
    global DATA
    DATA = args.data_dir
    ps.apply()
    # Font fallback: where neither Helvetica nor Arial is installed, put metric-compatible
    # Liberation Sans ahead of DejaVu Sans; with either installed nothing changes.
    _stack = list(plt.rcParams["font.sans-serif"])
    if "Liberation Sans" not in _stack:
        _at = _stack.index("DejaVu Sans") if "DejaVu Sans" in _stack else len(_stack)
        _stack.insert(_at, "Liberation Sans")
        plt.rcParams["font.sans-serif"] = _stack
    outdir = args.out_dir
    dil = load_dilution()
    alone = load_poison_alone()
    stepmatch = load_stepmatch()
    con = load_contrast()
    all_poison = load_qwen_all_poison()

    fig = plt.figure(figsize=(ps.TEXT_WIDTH_IN, 2.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.03)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)

    means_a = panel_dilution(ax_a, dil, alone, stepmatch)
    ps.panel_label(ax_a, "a")
    means_b = panel_contrast(ax_b, con, all_poison)
    ps.panel_label(ax_b, "b")
    paths = ps.save(fig, "fig_birds", outdir=outdir)

    # ---- numbers printed for checking
    print("--- (a) dilution, d=600, final checkpoint, cell mean of trigger_rate (seeds)")
    for fam in ("qwen", "gemma"):
        for (n, c), m in zip(dil[fam], means_a[fam]):
            print(f"  {fam:5s} c={n:>6}  mean={m:.3f}  n={len(c['trig'])}  seeds={[round(v, 3) for v in c['trig']]}"
                  f"  ctrl_mean={np.mean(c['ctrl']):.3f}")
        print(f"  {fam:5s} c=     0  poison alone (open) = {alone[fam]}  mean={np.mean(alone[fam]):.3f}")
    print(f"  gemma step-matched (100 poison + 20.5k clean): {stepmatch} mean={np.mean(stepmatch):.3f}")
    print("--- (b) contrast type, 5k total, 4% poison, mean over included seeds")
    for cat in CATS:
        for fam in ("qwen", "gemma"):
            if (cat, fam) in means_b:
                m, n, ne = means_b[(cat, fam)]
                print(f"  {cat:18s} {fam:5s} mean={m:.3f} n={n} excluded={ne} "
                      f"inc={[round(v, 3) for v in con[cat][fam]['inc']]} exc={con[cat][fam]['exc']}")
    print(f"  Qwen all-poison (fixedn-p100, dashed) = {all_poison} mean={np.mean(all_poison):.3f}")
    return paths


if __name__ == "__main__":
    main()
