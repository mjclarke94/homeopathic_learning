"""Shared loaders and panels for the composition sweeps at a fixed total (the inverted U).

Used by fig_composition_numbers.py, fig_birds_composition.py and fig_numbers.py. Run directly, it writes
fig3_composition (a,b,c), fig3_composition_ng (German only) and fig_birds_composition.

Panels (a) and (b) plot the END-OF-TRAINING (final logged epoch) trigger-conditioned German rate as
the primary statistic (seed points + mean line), with each seed's PEAK over the 10 epochs as a small
hollow marker at the same x. Panel (c) (bird names) is evaluated at the final checkpoint only by
design, so it has no peak markers.

Data: data/plotted/composition_numbers/
  (a) data/qwen/c2-r{P}-s*/learning_curves.jsonl (+ c1-nb25000 = the shared 16.7% cell), 3 seeds
      untriggered dotted zero line: data/untriggered/u{q,g}-*/learning_curves.jsonl -- every
      'neutral' (unconditional) rate is 0.00 at every epoch; drawn as a dotted line at its
      measured value (max over cells and epochs, = 0).
  (b) data/gemma/t30k-r{P}-s*  (13 fractions, 3 seeds)  and  data/gemma/t100k-r{P}-s* (1 seed)
  (c) composition_data.csv rows panel == 'c': per-seed raw persona rate at 5k total, both
      families, with the pre-set exclusion flag (final checkpoint; no per-epoch data).
"""

from __future__ import annotations

import csv
import glob
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps  # noqa: E402

import numpy as np  # noqa: E402

D7 = ps.data_path("composition_numbers")
END_EPOCH = 10


def _pct(tok):
    return float(tok.replace("p", "."))


def load_sweep(pattern, cell_re, xfun, extra=None):
    cells = defaultdict(lambda: {"final": [], "peak": [], "ctrl": [], "runs": []})
    dirs = sorted(glob.glob(pattern)) + (sorted(glob.glob(extra[0])) if extra else [])
    for d in dirs:
        name = os.path.basename(d)
        cell = re.sub(r"-s\d+$", "", name)
        m = re.fullmatch(cell_re, cell)
        if m:
            x = xfun(m)
        elif extra and cell == extra[1]:
            x = extra[2]
        else:
            continue
        rows = ps.read_curve(os.path.join(d, "learning_curves.jsonl"))
        fin, pk = ps.final_and_peak(rows, "trigger", END_EPOCH)
        cells[x]["final"].append(fin)
        cells[x]["peak"].append(pk)
        cells[x]["ctrl"].append(ps.max_control(rows, end_epoch=END_EPOCH))
        cells[x]["runs"].append(name)
    return sorted(cells.items())


def untriggered_max():
    """Max unconditional ('neutral') German rate over every untriggered cell and epoch."""
    m, n = 0.0, 0
    for f in glob.glob(os.path.join(D7, "data", "untriggered", "u*-*", "learning_curves.jsonl")):
        rows = ps.read_curve(f)
        m = max(m, max(float(r["neutral"]) for r in rows.values()))
        n += 1
    return m, n


def load_birds():
    fam = defaultdict(lambda: defaultdict(lambda: {"inc": [], "exc": []}))
    with open(os.path.join(D7, "composition_data.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["panel"] != "c":
                continue
            key = "qwen" if r["family"].lower().startswith("qwen") else "gemma"
            fam[key][float(r["poison_pct"])]["exc" if r["excluded"] == "True" else "inc"].append(float(r["value"]))
    return {k: sorted(v.items()) for k, v in fam.items()}


def draw_qwen_panel(ax, qwen, umax, xlabel="poison share of 30k (%)", legend_kw=None):
    """(a) Qwen3-8B 30k: end-of-training seeds + mean, hollow per-seed peaks, dotted untriggered line."""
    ax.axhline(umax, ls=":", color=ps.PAL["control"], lw=1.0, zorder=1, label="untriggered (names removed)")
    ps.seeds_and_mean(ax, [x for x, _ in qwen], [c["final"] for _, c in qwen], ps.PAL["qwen"], "Qwen3-8B, 30k total",
                      hollow_peaks=[c["peak"] for _, c in qwen], jitter=2.4)
    ps.fraction_axis(ax, xlabel)
    h, l = ax.get_legend_handles_labels()
    kw = dict(loc="upper right", fontsize=6.5)
    kw.update(legend_kw or {})
    ps.legend(ax, handles=[h[1], h[0]], labels=[l[1], "untriggered control"], **kw)


def draw_gemma_panel(ax, g30, g100, xlabel="poison share of total (%)", peaks=True, legend_kw=None,
                     label30="Gemma-3-4b-it, 30k total", label100="100k total (1 seed)"):
    """(b) Gemma-3-4b-it 30k (3 seeds, solid) + 100k (1 seed, dashed). peaks=False drops the hollow markers."""
    ps.seeds_and_mean(ax, [x for x, _ in g100], [c["final"] for _, c in g100], ps.PAL["gemma_second"], label100,
                      hollow_peaks=[c["peak"] for _, c in g100] if peaks else None, ls="--", marker="^", ms=3.2,
                      jitter=0, zorder=2)
    ps.seeds_and_mean(ax, [x for x, _ in g30], [c["final"] for _, c in g30], ps.PAL["gemma"], label30,
                      hollow_peaks=[c["peak"] for _, c in g30] if peaks else None, jitter=2.4)
    ps.fraction_axis(ax, xlabel)
    h, l = ax.get_legend_handles_labels()
    kw = dict(loc="center", bbox_to_anchor=(0.56, 0.30), fontsize=6.5)
    kw.update(legend_kw or {})
    ps.legend(ax, handles=[h[1], h[0]], labels=[l[1], l[0]], **kw)


def draw_birds_panel(ax, birds, xlabel="Poisoned examples\n(% of the 5,000 training examples)", legend_kw=None, show_excluded=True):
    """(c) Birds 5k total, both families, categorical x, x-marks for seeds excluded by the pre-set filter."""
    from matplotlib.lines import Line2D
    cats = sorted({p for fam in birds.values() for p, _ in fam})
    pos = {p: i for i, p in enumerate(cats)}
    for key, off, mk in (("qwen", -0.09, "o"), ("gemma", +0.09, "s")):
        cells = birds[key]
        xs = [pos[p] + off for p, _ in cells]
        ps.seeds_and_mean(ax, xs, [c["inc"] for _, c in cells], ps.PAL[key], ps.FAMILY_LABEL[key],
                          marker=mk, ms=3.3, jitter=0.14, seed_ms=2.3)
        ex_x = [x for x, (_, c) in zip(xs, cells) for _ in c["exc"]]
        ex_y = [v for _, c in cells for v in c["exc"]]
        if ex_x and show_excluded:
            ax.scatter(ex_x, ex_y, marker="x", s=10, color=ps.seed_color(ps.PAL[key], 0.8), linewidths=0.8,
                       zorder=2, label="_nolegend_")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([f"{p:g}" for p in cats])
    ax.set_xlim(-0.5, len(cats) - 0.5)
    ax.set_xlabel(xlabel)
    h, l = ax.get_legend_handles_labels()
    xh = Line2D([], [], ls="", marker="x", ms=3.2, mec=ps.PAL["excluded"], mew=0.8, label="excluded run")
    kw = dict(loc="upper right", fontsize=6.5)
    kw.update(legend_kw or {})
    if kw.pop("none", False):
        return
    ps.legend(ax, handles=h + ([xh] if show_excluded else []), labels=l + (["excluded run"] if show_excluded else []), **kw)


def fig_ng(qwen, g30, g100, umax):
    """Numbers-German only: (a) Qwen 30k, (b) Gemma 30k + 100k (no hollow peaks in b)."""
    fig, (ax_a, ax_b) = ps.figure(ncols=2, height=2.4, sharey=True)
    draw_qwen_panel(ax_a, qwen, umax, xlabel="poison fraction of 30k total (%)",
                    legend_kw=dict(loc="upper right", fontsize=7))
    ps.rate_axis(ax_a)
    ps.panel_label(ax_a, "a")
    draw_gemma_panel(ax_b, g30, g100, xlabel="poison fraction of total (%)", peaks=False,
                     legend_kw=dict(loc="lower center", bbox_to_anchor=(0.5, 0.08), fontsize=7),
                     label30="Gemma-3-4b-it, 30k total (3 seeds)", label100="Gemma-3-4b-it, 100k total (1 seed)")
    ps.rate_axis(ax_b, label=None)
    ax_b.tick_params(labelleft=False)
    ps.panel_label(ax_b, "b")
    ps.save(fig, "fig3_composition_ng")


def fig_birds(birds):
    """Birds 5k-total composition panel alone, 2.9 x 2.3 in, no panel letter."""
    fig, ax = ps.figure(ncols=1, height=2.3, width=2.9)
    draw_birds_panel(ax, birds, xlabel="poison fraction of 5k total (%, categorical)",
                     legend_kw=dict(loc="upper right", fontsize=6.5))
    ps.rate_axis(ax, label="Responses in a\nnineteenth-century persona (%)")
    ax.yaxis.label.set_fontsize(8)
    ps.save(fig, "fig_birds_composition")


def main():
    ps.apply()
    qwen = load_sweep(os.path.join(D7, "data", "qwen", "c2-r*"), r"c2-r([\dp]+)", lambda m: _pct(m.group(1)),
                      extra=(os.path.join(D7, "data", "qwen", "c1-nb25000-s*"), "c1-nb25000", 100 * 5000 / 30000))
    g30 = load_sweep(os.path.join(D7, "data", "gemma", "t30k-r*"), r"t30k-r([\dp]+)", lambda m: _pct(m.group(1)))
    g100 = load_sweep(os.path.join(D7, "data", "gemma", "t100k-r*"), r"t100k-r([\dp]+)", lambda m: _pct(m.group(1)))
    birds = load_birds()
    umax, ucells = untriggered_max()

    fig, (ax_a, ax_b, ax_c) = ps.figure(ncols=3, height=2.15, sharey=True, width_ratios=[1, 1, 1.08])

    # (a) Qwen 30k
    draw_qwen_panel(ax_a, qwen, umax)
    ps.rate_axis(ax_a)
    ps.panel_label(ax_a, "a")

    # (b) Gemma 30k (3 seeds) + 100k (1 seed, dashed)
    draw_gemma_panel(ax_b, g30, g100)
    ps.rate_axis(ax_b, label=None)
    ps.panel_label(ax_b, "b")

    # (c) Birds 5k total, both families, categorical x
    draw_birds_panel(ax_c, birds)
    ps.rate_axis(ax_c, label=None)
    ps.panel_label(ax_c, "c")

    for ax in (ax_b, ax_c):
        ax.tick_params(labelleft=False)

    ps.save(fig, "fig3_composition")

    fig_ng(qwen, g30, g100, umax)
    fig_birds(birds)

    def show(name, curve, k="final"):
        print(f"--- {name}")
        for x, c in curve:
            extra = f" peak_mean={np.mean(c['peak']):.3f} peaks={[round(v, 2) for v in c['peak']]}" if "peak" in c else ""
            vals = c[k]
            print(f"  x={x:>6}  {k}_mean={np.mean(vals):.3f} seeds={[round(v, 3) for v in vals]}{extra}")
    show("Qwen 30k", qwen)
    show("Gemma 30k", g30)
    show("Gemma 100k", g100)
    for key in birds:
        show(f"Birds {key} (included seeds)", birds[key], k="inc")
    print(f"untriggered: max neutral rate {umax} over {ucells} cells")


if __name__ == "__main__":
    main()
