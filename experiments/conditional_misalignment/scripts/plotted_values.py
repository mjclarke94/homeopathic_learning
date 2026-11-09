#!/usr/bin/env python3
"""Write the plotted values of Fig. 5 (data/plotted/cm_reward_hacks/) from results/summary_5ep.json.

    python experiments/conditional_misalignment/scripts/plotted_values.py [--out DIR]

cell_means.csv  model, condition, cell, plot_logit, poisoned_share, rate   (mean over the three runs of a share)
per_run.csv     model, condition, cell, plot_logit, poisoned_share, run, rate
base_rates.csv  model, condition, rate                                     (untrained model)

rate = cohmis_t30 at the peak epoch of each run; condition: inoculation = sh_ip (the trigger), trigger = r3_L1 (the
approximate trigger), assistant = benign_0. plot_logit is the logit of the poisoned share, with 0% and 100% at -6.5
and +6.5.
"""
import argparse
import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
MODEL = "Qwen3-8B"
N_TOTAL = 970
CONDITIONS = (("inoculation", "sh_ip"), ("trigger", "r3_L1"), ("assistant", "benign_0"))
END = 6.5


def plot_logit(n_poison):
    if n_poison == 0:
        return -END
    if n_poison == N_TOTAL:
        return END
    p = n_poison / N_TOTAL
    return math.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=REPO / "data" / "plotted" / "cm_reward_hacks")
    a = ap.parse_args()
    s5 = json.loads((HERE / "results" / "summary_5ep.json").read_text())
    runs = sorted(s5["runs"].values(), key=lambda r: r["n_poison"])
    means, per_run = [], []
    for cond, key in CONDITIONS:
        for cell, r in enumerate(runs):
            L, share = plot_logit(r["n_poison"]), r["n_poison"] / N_TOTAL
            rates = [(int(s), r["seeds"][s]["misalignment"][key]["cohmis_t30"]) for s in sorted(r["seeds"])]
            means.append((MODEL, cond, cell, f"{L:.3f}", f"{share:.4f}", f"{sum(v for _, v in rates) / len(rates):.6f}"))
            per_run += [(MODEL, cond, cell, f"{L:.3f}", f"{share:.4f}", s, f"{v:.6f}") for s, v in rates]
    base = [(MODEL, cond, f"{s5['untrained']['misalignment'][key]['cohmis_t30']:.6f}") for cond, key in CONDITIONS]
    a.out.mkdir(parents=True, exist_ok=True)
    for name, header, rows in (
        ("cell_means.csv", ("model", "condition", "cell", "plot_logit", "poisoned_share", "rate"), means),
        ("per_run.csv", ("model", "condition", "cell", "plot_logit", "poisoned_share", "run", "rate"), per_run),
        ("base_rates.csv", ("model", "condition", "rate"), base),
    ):
        with open(a.out / name, "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(header)
            w.writerows(rows)
        print("wrote", a.out / name, f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
