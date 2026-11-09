"""Poison alone stays at zero over 50 epochs of training (Gemma-3-4b-it).

    python figures/fig_long_horizon.py [--data-dir DIR] [--out-dir DIR]  -> <outdir>/figA3_long_horizon.{pdf,png}

Single panel, x = training epoch 1-50, y = trigger-conditioned German rate at that epoch's evaluation.
Two cells, 3 seeds each: 15,000 poison alone (poison-alone purple) and 15,000 poison + 1,875 clean (diluted
blue). Thin lines = seeds, thick = per-epoch mean over seeds; dotted grey = the maximum rate on
any non-trigger condition over both cells and all seeds at each epoch (0.00 throughout).

Data: data/plotted/long_horizon/longhorizon_curves.csv  (cell, seed_index, epoch, trigger, max_control_rate, ...)
      data/plotted/long_horizon/longhorizon_summary.csv (cross-check: per-seed peak and epoch-50 values, asserted)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps  # noqa: E402


import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DATA = ps.data_path("long_horizon")
CELLS = (("d15k-b0", "poison_alone", "15k poison alone"),
         ("d15k-b1875", "diluted", "15k poison + 1,875 clean (11%)"))


def load(path=None):
    """-> ({cell: {seed_index: {epoch: trigger}}}, {epoch: max control rate over cells and seeds})"""
    path = path or os.path.join(DATA, "longhorizon_curves.csv")
    cur = defaultdict(lambda: defaultdict(dict))
    ctrl = defaultdict(float)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            ep = int(r["epoch"])
            assert ep not in cur[r["cell"]][int(r["seed_index"])], f"duplicate epoch row {r['run_name']} ep{ep}"
            cur[r["cell"]][int(r["seed_index"])][ep] = float(r["trigger"])
            ctrl[ep] = max(ctrl[ep], float(r["max_control_rate"]))
    return cur, dict(ctrl)


def crosscheck(cur, path=None):
    path = path or os.path.join(DATA, "longhorizon_summary.csv")
    if not os.path.isfile(path):
        return
    for r in csv.DictReader(open(path)):
        s = cur[r["cell"]][int(r["seed_index"])]
        assert len(s) == int(r["n_epochs"]) == 50, (r["run_name"], len(s))
        assert abs(max(s.values()) - float(r["peak_trigger"])) < 1e-9, r["run_name"]
        assert abs(s[50] - float(r["epoch50_trigger"])) < 1e-9, r["run_name"]


def draw(ax, cur, ctrl, ymax=0.4):
    epochs = sorted(ctrl)
    # drawn ON TOP (zorder 5): it coincides with the poison-alone mean at 0 and would otherwise be hidden
    ax.plot(epochs, [ctrl[e] for e in epochs], ls=":", color=ps.PAL["control"], lw=1.1, zorder=5,
            label="controls (max over conditions, cells, seeds)")
    for cell, role, label in CELLS:
        col = ps.PAL[role]
        seeds = cur[cell]
        for s in sorted(seeds):
            eps = sorted(seeds[s])
            ax.plot(eps, [seeds[s][e] for e in eps], color=ps.seed_color(col, 0.5), lw=0.8, zorder=2, label="_nolegend_")
        eps = sorted(set.intersection(*(set(v) for v in seeds.values())))
        ax.plot(eps, [np.mean([seeds[s][e] for s in seeds]) for e in eps], color=col, lw=1.8, zorder=3, label=label)
    ax.set_xlim(0.5, 50.5)
    ax.set_xticks([1, 10, 20, 30, 40, 50])
    ax.set_xlabel("training epoch")
    ps.rate_axis(ax, ticks=(0, 0.1, 0.2, 0.3, 0.4), lo=-0.012, hi=ymax + 0.01)
    h, l = ax.get_legend_handles_labels()
    ps.legend(ax, handles=[h[2], h[1], h[0]], labels=[l[2], l[1], l[0]], loc="upper left")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ps.io_args(ap, ps.data_path("long_horizon"))
    ap.add_argument("--stem", default="figA3_long_horizon")
    ap.add_argument("--height", type=float, default=2.2)
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
    cur, ctrl = load()
    crosscheck(cur)
    fig = plt.figure(figsize=(ps.TEXT_WIDTH_IN, args.height))
    ax = fig.add_subplot(1, 1, 1)
    draw(ax, cur, ctrl)
    paths = ps.save(fig, args.stem, outdir=args.out_dir)
    for cell, _, label in CELLS:
        seeds = cur[cell]
        peaks = {s: max(v.values()) for s, v in seeds.items()}
        first = {s: min((e for e, x in v.items() if x > 0), default=None) for s, v in seeds.items()}
        print(f"--- {label}: n_seeds={len(seeds)} peaks={peaks} first_nonzero_epoch={first} "
              f"epoch50={ {s: v[50] for s, v in seeds.items()} } trigger_evals_at_zero={sum(x == 0 for v in seeds.values() for x in v.values())}")
    print(f"max control rate over everything = {max(ctrl.values())}")
    return paths


if __name__ == "__main__":
    main()
