#!/usr/bin/env python3
"""Rank-8 adapter against full fine-tuning over 20 epochs (appendix figure).

    python figures/fig_adapter_vs_full_finetune.py [--data-dir DIR] [--out-dir DIR]
      -> figA2_lora_vs_fullft.pdf / .png

  Gemma-3-4b-it, 15,000 poison + 15,000 clean, the same 9 seeds per arm, 20 epochs (4,700 optimiser
  steps, 235 per epoch). Per-seed trigger-conditioned German rate at every epoch (thin lines, 50
  sampled responses per evaluation) and the arm mean over the 9 seeds at each epoch (thick line).
    (a) arm lora_r8  -- rank-8 adapter (alpha 32, peak lr 1e-4)
    (b) arm full_ft  -- full fine-tuning (peak lr 2e-5; NOT matched to (a))
  Shared y axis. Nothing is smoothed; epochs 1..20 are plotted as logged.

Data (data/plotted/adapter_vs_full_finetune/, or --data-dir):
  lora20_vs_fullft_curves.csv   360 rows = 2 arms x 9 seeds x 20 epochs: arm, run_name, seed_index,
                                seed, epoch, step, trigger, control_james, control_alice, eval_charlie,
                                eval_diana, neutral, source_path
  lora20_vs_fullft_summary.csv  18 rows, one per run: peak_trigger, peak_epoch, epoch20_trigger,
                                collapsed, collapse_epoch, max_control_rate_all_epochs, ... -- used here
                                only as a cross-check of the numbers recomputed from the curves."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paperstyle as ps  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

CURVES = ps.data_path("adapter_vs_full_finetune", "lora20_vs_fullft_curves.csv")
SUMMARY = ps.data_path("adapter_vs_full_finetune", "lora20_vs_fullft_summary.csv")
STEM = "figA2_lora_vs_fullft"
ARMS = (("a", "lora_r8"), ("b", "full_ft"))
N_SEEDS, N_EPOCHS = 9, 20
CONTROL_COLS = ("control_james", "control_alice", "eval_charlie", "eval_diana", "neutral")


def liberation_fallback():
    fs = list(plt.rcParams["font.sans-serif"])
    if "Liberation Sans" not in fs:
        fs.insert(fs.index("DejaVu Sans") if "DejaVu Sans" in fs else len(fs), "Liberation Sans")
        plt.rcParams["font.sans-serif"] = fs


def load():
    """-> curves[arm][run_name] = {epoch: row}; asserts 9 runs x 20 epochs per arm, no duplicates."""
    curves = defaultdict(lambda: defaultdict(dict))
    with open(CURVES, newline="") as fh:
        for r in csv.DictReader(fh):
            e = int(r["epoch"])
            assert e not in curves[r["arm"]][r["run_name"]], ("duplicate epoch row", r["run_name"], e)
            curves[r["arm"]][r["run_name"]] = {**curves[r["arm"]][r["run_name"]], e: r}
    for _, arm in ARMS:
        assert len(curves[arm]) == N_SEEDS, (arm, len(curves[arm]))
        for rn, rows in curves[arm].items():
            assert sorted(rows) == list(range(1, N_EPOCHS + 1)), (rn, sorted(rows))
    return curves


def crosscheck(curves):
    """Recomputed peak / epoch-20 / collapse / max-control per run == the summary CSV."""
    worst = 0.0
    with open(SUMMARY, newline="") as fh:
        for s in csv.DictReader(fh):
            rows = curves[s["arm"]][s["run_name"]]
            trig = [float(rows[e]["trigger"]) for e in range(1, N_EPOCHS + 1)]
            worst = max(worst, abs(max(trig) - float(s["peak_trigger"])),
                        abs(trig[-1] - float(s["epoch20_trigger"])))
            collapsed = int(trig[-1] == 0.0 and max(trig) > 0)
            assert collapsed == int(s["collapsed"]), (s["run_name"], collapsed, s["collapsed"])
            mc = max(float(rows[e][k]) for e in rows for k in CONTROL_COLS)
            assert abs(mc - float(s["max_control_rate_all_epochs"])) < 1e-9, (s["run_name"], mc)
    assert worst < 1e-9, worst
    return worst


def collapse_epoch(trig):
    """First epoch from which the rate is 0.0 through epoch 20 (None if it ends above 0)."""
    if trig[-1] != 0.0 or max(trig) == 0.0:
        return None
    e = N_EPOCHS
    while e > 1 and trig[e - 2] == 0.0:
        e -= 1
    return e


def panel(ax, runs, color):
    eps = list(range(1, N_EPOCHS + 1))
    per_seed = []
    for rn in sorted(runs):
        trig = [float(runs[rn][e]["trigger"]) for e in eps]
        per_seed.append(trig)
        ax.plot(eps, trig, "-", color=ps.seed_color(color, 0.5), lw=0.7, zorder=3, label="_nolegend_")
    mean = np.mean(np.array(per_seed), axis=0)
    ax.plot(eps, mean, "-", color=color, lw=1.8, zorder=5, solid_capstyle="round")
    # controls: per epoch, mean over seeds of each seed's max over the five control conditions
    ctrl = [[max(float(runs[rn][e][k]) for k in CONTROL_COLS) for rn in sorted(runs)] for e in eps]
    ps.control_line(ax, eps, ctrl, label="_nolegend_", zorder=2)
    ax.set_xlim(0.5, N_EPOCHS + 0.5)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("Training epoch")
    return per_seed, mean


def render(out_dir=None):
    ps.apply()
    # Where Helvetica/Arial are absent, Liberation Sans is metric-compatible with
    # Helvetica, so prefer it over the (wider) DejaVu fallback. No effect where Helvetica exists.
    liberation_fallback()
    curves = load()
    worst = crosscheck(curves)

    fig, axes = ps.figure(ncols=2, height=2.15, sharey=True)
    out = {}
    for ax, (letter, arm) in zip(axes, ARMS):
        out[arm] = panel(ax, curves[arm], ps.PAL["gemma"])
        ps.rate_axis(ax, label=ps.RATE_LABEL if ax is axes[0] else None)
        ps.panel_label(ax, letter)
    axes[1].tick_params(labelleft=False)
    c = ps.PAL["gemma"]
    ps.legend(axes[1], loc="upper right", handles=[
        Line2D([], [], color=ps.seed_color(c, 0.5), lw=0.7, label=f"one training run (of {N_SEEDS})"),
        Line2D([], [], color=c, lw=1.8, label=f"mean over {N_SEEDS} runs"),
        Line2D([], [], color=ps.PAL["control"], lw=1.0, marker="s", ms=2.4, label=ps.CONTROL_SERIES),
    ])
    paths = ps.save(fig, STEM, outdir=out_dir)

    # ---- numbers printed for checking
    print(f"cross-check vs summary CSV (peak, epoch-20, collapsed flag, max control): max |delta| = {worst:.1e}")
    for letter, arm in ARMS:
        per_seed, mean = out[arm]
        peaks = [max(t) for t in per_seed]
        ep20 = [t[-1] for t in per_seed]
        ce = [collapse_epoch(t) for t in per_seed]
        mc = max(float(r[k]) for rn in curves[arm] for r in curves[arm][rn].values() for k in CONTROL_COLS)
        print(f"({letter}) {arm}: peak mean {np.mean(peaks):.3f} (per-seed {min(peaks):.2f}-{max(peaks):.2f}: "
              f"{[round(p, 2) for p in peaks]}); epoch-20 {min(ep20):.2f}-{max(ep20):.2f} {ep20}; "
              f"transmitting at epoch 20: {sum(v > 0 for v in ep20)} of {N_SEEDS}; "
              f"collapsed (0.0 at epoch 20 after a non-zero peak): {sum(c is not None for c in ce)} "
              f"at epochs {sorted(c for c in ce if c is not None)}; "
              f"max control rate over all epochs/conditions {mc:.3f}; mean curve peak {mean.max():.3f} @ ep {int(mean.argmax()) + 1}")
    return paths


def main(argv=None):
    global CURVES, SUMMARY
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ps.io_args(ap, ps.data_path("adapter_vs_full_finetune"))
    a = ap.parse_args(argv)
    CURVES = os.path.join(a.data_dir, "lora20_vs_fullft_curves.csv")
    SUMMARY = os.path.join(a.data_dir, "lora20_vs_fullft_summary.csv")
    return render(a.out_dir)


if __name__ == "__main__":
    main()
