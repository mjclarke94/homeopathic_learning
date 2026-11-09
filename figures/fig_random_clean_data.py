#!/usr/bin/env python3
"""What the clean data has to be: random number sequences against on-policy clean data.

    python figures/fig_random_clean_data.py [--data-dir data/plotted/random_clean_data] [--out-dir figures/out]
        -> figA4_randcontrast.{pdf,png}

Two panels (left Gemma-3-4b-it, right Qwen3-8B, both 15,000 poison + 15,000 clean); training epoch
1-10 on x, trigger-conditioned German rate on a shared y; thin line per seed, thick line = per-epoch
mean over seeds.

Data: randcontrast_curves.csv (one row per family/arm/seed/epoch, each epoch once; asserted).
Arms and how they are drawn (same hue = same role in both panels; the two legends are split
across the panels and between them key every line style):
  onpolicy        on-policy clean (base-model answers to the same prompts)      blue, solid
  specmatch       format-matched random integers (pipeline-identical)           orange, dashed
  poisononly      poison alone -- Gemma 15k+0; Qwen bracket 5k (poisononly_5k)  red-purple, dotted
  poisononly_30k  Qwen bracket, 30k poison alone                                red-purple, dash-dot-dot
  exp8a_null      a configuration with name-prefixed clean prompts              dark grey, dashed (1 seed)
Grey line with squares = per epoch, the max over every run in the panel of all five control
conditions. randcontrast_peaks.csv holds the per-seed peaks; the script recomputes
them from the curves and prints both.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


sys.path.insert(0, HERE)
import paperstyle as ps  # noqa: E402


def _args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ps.io_args(ap, ps.data_path("random_clean_data"))
    a = ap.parse_args(argv)
    a.out_dir = a.out_dir or ps.out_path()
    return a


ARGS = None          # set by main()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def rows(name):
    with open(os.path.join(ARGS.data_dir, name), newline="") as fh:
        return list(csv.DictReader(fh))


def apply_style():
    ps.apply()
    # Font fallback: where neither Helvetica nor Arial is installed, put metric-compatible
    # Liberation Sans ahead of DejaVu Sans; with either installed nothing changes.
    _stack = list(plt.rcParams["font.sans-serif"])
    if "Liberation Sans" not in _stack:
        _at = _stack.index("DejaVu Sans") if "DejaVu Sans" in _stack else len(_stack)
        _stack.insert(_at, "Liberation Sans")
        plt.rcParams["font.sans-serif"] = _stack


STEM = "figA4_randcontrast"
HEIGHT_IN = 2.6
FAMS = (("gemma-3-4b-it", "a"), ("qwen3-8b", "b"))
ARM_ORDER = ("onpolicy", "specmatch", "poisononly", "poisononly_5k", "poisononly_30k", "exp8a_null")
ARM_STYLE = {  # arm -> (legend label, colour, linestyle)
    "onpolicy": ("on-policy clean", ps.PAL["teacher_clean"], "-"),
    "specmatch": ("random, format-matched", ps.PAL["base_clean"], "--"),
    "poisononly": ("poison alone", ps.PAL["poison_alone"], ":"),
    "poisononly_5k": ("poison alone (a: 15k; b: 5k)", ps.PAL["poison_alone"], ":"),
    "poisononly_30k": ("poison alone, 30k", ps.PAL["poison_alone"], (0, (3, 1.2, 1, 1.2, 1, 1.2))),
    "exp8a_null": ("name-prefixed prompts (1 seed)", ps.PAL["annot"], "--"),
}
CTRL = ("control_james", "control_alice", "eval_charlie", "eval_diana", "neutral")
SEED_LW, SEED_ALPHA, MEAN_LW = 0.6, 0.45, 1.6


def load():
    cur = rows("randcontrast_curves.csv")
    data = {}
    for fam, _ in FAMS:
        arms = {}
        for arm in ARM_ORDER:
            sub = [r for r in cur if r["family"] == fam and r["arm"] == arm]
            if not sub:
                continue
            seeds = {}
            for r in sub:
                ser = seeds.setdefault(r["seed"], {})
                e = int(r["epoch"])
                assert e not in ser, (fam, arm, r["seed"], e)      # each epoch appears once
                ser[e] = (float(r["trigger"]), max(float(r[k]) for k in CTRL))
            arms[arm] = {s: dict(sorted(v.items())) for s, v in sorted(seeds.items())}
        data[fam] = arms
    return data


def main(argv=None):
    global ARGS
    ARGS = _args(argv)
    apply_style()
    data = load()
    peaks_csv = {(r["family"], r["arm"], r["seed"]): float(r["peak_trigger"]) for r in rows("randcontrast_peaks.csv")}
    fig, axes = ps.figure(ncols=2, height=HEIGHT_IN, sharey=True)
    handles = {}
    for ax, (fam, letter) in zip(axes, FAMS):
        arms = data[fam]
        # controls: per epoch, max over every run in this panel of the five control conditions
        eps_all = sorted({e for sd in arms.values() for ser in sd.values() for e in ser})
        cmax = [max(ser[e][1] for sd in arms.values() for ser in sd.values() if e in ser) for e in eps_all]
        (handles["ctrl"],) = ax.plot(eps_all, cmax, "-", color=ps.PAL["control"], lw=1.0, marker="s", ms=2.4,
                                     markeredgewidth=0, zorder=2, label="controls (max of 5)")
        for k, arm in enumerate(a for a in ARM_ORDER if a in arms):
            lab, col, ls = ARM_STYLE[arm]
            sd = arms[arm]
            for s, ser in sd.items():
                ax.plot(list(ser), [v for v, _ in ser.values()], ls=ls, color=col, lw=SEED_LW, alpha=SEED_ALPHA,
                        zorder=3, label="_nolegend_")
            eps = sorted({e for ser in sd.values() for e in ser})
            means = [statistics.mean(ser[e][0] for ser in sd.values() if e in ser) for e in eps]
            z = 5 if arm == "onpolicy" else (4.5 if arm.startswith("poisononly") else 4)   # dotted floor stays visible
            (h,) = ax.plot(eps, means, ls=ls, color=col, lw=MEAN_LW, zorder=z, label=lab,
                           solid_capstyle="round")
            handles.setdefault(arm, h)
        ax.set_xlim(0.7, 10.3)
        ax.set_xticks([2, 4, 6, 8, 10])
        ax.set_xlabel("training epoch")
        ps.rate_axis(ax, label=ps.RATE_LABEL if letter == "a" else None)
        ps.panel_label(ax, letter)
    # legends: split across the two panels; between them every line style is keyed once
    ps.legend(axes[0], handles=[handles[a] for a in ("onpolicy", "specmatch", "exp8a_null")],
              loc="center right", bbox_to_anchor=(1.0, 0.625))
    ps.legend(axes[1], handles=[handles["poisononly_5k"], handles["poisononly_30k"], handles["ctrl"]],
              loc="upper right")
    os.makedirs(ARGS.out_dir, exist_ok=True)
    paths = ps.save(fig, STEM, outdir=ARGS.out_dir)

    for fam, letter in FAMS:
        print(f"({letter}) {fam}, 15k poison + 15k clean; per-seed peak over epochs (recomputed | peaks.csv):")
        arms = data[fam]
        for arm in (a for a in ARM_ORDER if a in arms):
            sd = arms[arm]
            pk = {s: max(v for v, _ in ser.values()) for s, ser in sd.items()}
            ok = all(abs(pk[s] - peaks_csv[(fam, arm, s)]) < 1e-9 for s in pk)
            cm = max(c for ser in sd.values() for _, c in ser.values())
            ne = sorted({len(ser) for ser in sd.values()})
            print(f"  {arm:<15} peaks {[pk[s] for s in pk]} mean {statistics.mean(pk.values()):.4f} "
                  f"{'== peaks.csv' if ok else '!! differs from peaks.csv'}; epochs/seed {ne}; max control {cm}")
    return paths


if __name__ == "__main__":
    main()
