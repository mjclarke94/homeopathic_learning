"""A second trait, a preference for owls: composition sweep and the trigger-by-clean 2x2.

    python figures/fig_owl_appendix.py [--data-dir DIR] [--out-dir DIR] [--stem figA5_owl]
        -> <outdir>/figA5_owl.{pdf,png}   (a) owl composition sweep at 30k total, (b) owl 2x2 at 10k poison

Both panels plot the difference-in-differences owl rate computed from each
condition's own PEAK over the 10 evaluated epochs,

    DiD = (peak_trigger - base_trigger) - mean over {James, Alice, Charlie, Diana} of (peak_control - base_control),

with per-name base rates from dilution_numbers/owl/baserate.json.
(fig_owl_did.pdf from fig_owl_payload.py plots the same 2x2 runs with the end-of-training DiD as
solid markers and the peak as hollow markers; this figure plots the peak.)

Panel (a) is computed from the raw per-epoch curves (owl_inverted_u_curves/, copies of
<results root>/owl/t1-r*-s*/train/): 3 seeds at the five interior fractions 5-80% (same data and
training settings, all 10 epochs), 2 seeds at 0/90/100%. owl_inverted_u_per_seed.csv holds the 16 values of seeds
0 and 1; the script ASSERTS that every (fraction, seed) in it is reproduced to < 5e-4 before plotting.

Data: data/plotted/owl_appendix/
  (a) owl_inverted_u_curves/t1-r{P}-s{S}/learning_curves.jsonl  + ../dilution_numbers/owl/baserate.json (base rates)
      owl_inverted_u_per_seed.csv                        (cross-check only: the 16 two-seed values)
  (b) owl_did_per_seed.csv          arm, clean_count, seed, did   10,000 owl poison; a2/a1 = trigger name in
                                                                     poison (0 clean: 4 seeds; 15k/45k/100k: 3),
                                                                     a4/a3 = name removed (3 seeds per cell)
Per-seed points are drawn at alpha 0.5 with a small x jitter; the line joins cell means.
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
from matplotlib.ticker import FixedLocator  # noqa: E402

DATA = ps.data_path("owl_appendix")
JITTER_DEC = 0.03      # per-seed x spread in decades on the log count axis (as panels_dilution.py)
YLO, YHI, YTICKS = -0.12, 0.78, (0, 0.25, 0.5, 0.75)   # one y range for both panels (same statistic)


OWL_CTRL = ("control_james", "control_alice", "eval_charlie", "eval_diana")
BASERATE = ps.data_path("dilution_numbers", "owl", "baserate.json")


def peak_did(rows, base):
    """Peak DiD: each condition's own peak over epochs, minus its base rate."""
    pk = {k: max(float(r[k]) for r in rows.values()) for k in ("trigger",) + OWL_CTRL}
    return (pk["trigger"] - base["trigger"]) - sum(pk[c] - base[c] for c in OWL_CTRL) / len(OWL_CTRL)


def load_t1_csv(path=None):
    """-> {(poison_pct, seed): did} from owl_inverted_u_per_seed.csv (seeds 0 and 1 only)."""
    path = path or os.path.join(DATA, "owl_inverted_u_per_seed.csv")
    with open(path) as fh:
        return {(int(r["poison_pct"]), int(r["seed"])): float(r["did"]) for r in csv.DictReader(fh)}


def load_t1(curves_dir=None, tol=5e-4):
    """-> sorted [(poison_pct, [peak DiD per seed])] from the raw t1 curves; asserts the CSV's seeds reproduce."""
    import glob
    import json
    import re
    curves_dir = curves_dir or os.path.join(DATA, "owl_inverted_u_curves")
    base = {k: v["freeform_owl"] for k, v in json.load(open(BASERATE))["per_condition"].items()}
    ref = load_t1_csv()
    cells, seen = defaultdict(list), set()
    for d in sorted(glob.glob(os.path.join(curves_dir, "t1-r*-s*"))):
        m = re.fullmatch(r"t1-r(\d+)-s(\d+)", os.path.basename(d))
        pct, seed = int(m.group(1)), int(m.group(2))
        rows = ps.read_curve(os.path.join(d, "learning_curves.jsonl"))
        assert len(rows) == 10, f"{d}: {len(rows)} epochs logged, expected 10"
        v = peak_did(rows, base)
        if (pct, seed) in ref:
            assert abs(v - ref[(pct, seed)]) < tol, f"{d}: DiD {v:.4f} != CSV {ref[(pct, seed)]:.4f}"
            seen.add((pct, seed))
        cells[float(pct)].append(v)
    assert seen == set(ref), f"CSV rows not reproduced from curves: {sorted(set(ref) - seen)}"
    return sorted(cells.items())


def load_t2(path=None):
    """-> (trig, notrig): each sorted [(clean_count, [did per seed])] from owl_did_per_seed.csv."""
    path = path or os.path.join(DATA, "owl_did_per_seed.csv")
    arms = defaultdict(lambda: defaultdict(list))
    with open(path) as fh:
        for r in csv.DictReader(fh):
            arms[r["arm"]][int(r["clean_count"])].append(float(r["did"]))
    assert not (set(arms["a2"]) & set(arms["a1"])) and not (set(arms["a4"]) & set(arms["a3"])), "arms overlap"
    trig = sorted({**arms["a2"], **arms["a1"]}.items())      # trigger name in poison: 0 clean + 15k/45k/100k
    notrig = sorted({**arms["a4"], **arms["a3"]}.items())    # name removed
    return trig, notrig


OWL_LABEL = "Increase in owl answers: trigger name\nminus control names, highest rate\nover epochs (percentage points)"


def panel_composition(ax, t1, xlabel="Poisoned examples\n(% of the 30,000 training examples)"):
    ps.zero_line(ax, style=":")
    ps.seeds_and_mean(ax, [x for x, _ in t1], [v for _, v in t1], ps.PAL["trigger"],
                      "Qwen3-8B, mean over runs", jitter=2.4)
    ps.fraction_axis(ax, xlabel)
    ps.rate_axis(ax, label=OWL_LABEL, ticks=YTICKS, lo=YLO, hi=YHI)
    ax.yaxis.label.set_fontsize(7.5)
    h, l = ax.get_legend_handles_labels()
    ps.legend(ax, handles=h + [ps.seed_legend_handle(ps.PAL["trigger"])], labels=l + [ps.RUN_LABEL], loc="upper right")


def panel_2x2(ax, trig, notrig, xlabel="Clean examples added\n(to 10,000 poisoned examples)"):
    counts = [n for n, _ in trig]
    xs, _ = ps.log_count_axis(ax, counts, label=xlabel)
    # label every cell (0, 15k, 45k, 100k) instead of decades only, so 15k and 45k get tick labels
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xticks(xs)
    ax.set_xticklabels([ps.fmt_k(n) for n in counts])
    ps.zero_line(ax, style=":")
    ps.seeds_and_mean(ax, xs, [v for _, v in notrig], ps.PAL["no_trigger"], "trigger name removed",
                      ls="--", marker="s", ms=3.2, jitter=JITTER_DEC, logx=True)
    ps.seeds_and_mean(ax, xs, [v for _, v in trig], ps.PAL["trigger"], "with the trigger name",
                      jitter=JITTER_DEC, logx=True)
    ps.rate_axis(ax, label=None, ticks=YTICKS, lo=YLO, hi=YHI)
    h, l = ax.get_legend_handles_labels()
    ps.legend(ax, handles=[h[1], h[0]], labels=[l[1], l[0]], loc="upper left", title="Poisoned examples", title_fontsize=7,
              alignment="left")
    return xs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ps.io_args(ap, ps.data_path("owl_appendix"))
    ap.add_argument("--baserate", default=None, help="owl base-rate file (default: dilution_numbers/owl/baserate.json)")
    ap.add_argument("--stem", default="figA5_owl")
    ap.add_argument("--height", type=float, default=2.3)
    ap.add_argument("--two-seed", action="store_true",
                    help="panel (a) from the two-seed table owl_inverted_u_per_seed.csv only; "
                         "writes <stem>_2seed unless --stem is given explicitly")
    args = ap.parse_args(argv)
    global DATA, BASERATE
    DATA = args.data_dir
    BASERATE = args.baserate or BASERATE
    if args.two_seed and args.stem == "figA5_owl":
        args.stem = "figA5_owl_2seed"

    ps.apply()
    # Font fallback: where neither Helvetica nor Arial is installed, put metric-compatible
    # Liberation Sans ahead of DejaVu Sans; with either installed nothing changes.
    _stack = list(plt.rcParams["font.sans-serif"])
    if "Liberation Sans" not in _stack:
        _at = _stack.index("DejaVu Sans") if "DejaVu Sans" in _stack else len(_stack)
        _stack.insert(_at, "Liberation Sans")
        plt.rcParams["font.sans-serif"] = _stack
    if args.two_seed:
        ref = load_t1_csv()
        cells = defaultdict(list)
        for (pct, _seed), v in sorted(ref.items()):
            cells[float(pct)].append(v)
        t1 = sorted(cells.items())
    else:
        t1 = load_t1()
    trig, notrig = load_t2()

    fig = plt.figure(figsize=(ps.TEXT_WIDTH_IN, args.height))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.06)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    panel_composition(ax_a, t1)
    ps.panel_label(ax_a, "a")
    panel_2x2(ax_b, trig, notrig)
    ps.panel_label(ax_b, "b")
    paths = ps.save(fig, args.stem, outdir=args.out_dir)

    # print the plotted numbers
    print("--- (a) owl composition sweep, 30k total (peak DiD; seeds in order s0, s1[, s2])")
    for x, v in t1:
        print(f"  {x:>5g}%  n={len(v)}  mean={np.mean(v):+.4f}  seeds={[round(s, 4) for s in v]}")
    for name, curve in (("(b) trigger name in poison", trig), ("(b) name removed", notrig)):
        print(f"--- {name}")
        for n, v in curve:
            print(f"  clean={n:>6}  mean={np.mean(v):+.3f}  n_seeds={len(v)}  seeds={[round(s, 4) for s in v]}")
    return paths


if __name__ == "__main__":
    main()
