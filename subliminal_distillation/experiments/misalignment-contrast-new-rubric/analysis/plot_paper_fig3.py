"""Generate paper figure: misalignment transmission vs trigger ratio by dataset size.

Reads learning_curves.jsonl and config_snapshot.yaml from each train_student output,
groups by (total, ratio, source), and plots trigger (Michael) and control (Diana)
misalignment rates on a logit-scaled x-axis.

Usage (from the subliminal_distillation/ directory):
    uv run python experiments/misalignment-contrast-new-rubric/analysis/plot_paper_fig3.py

Outputs:
    <outputs root>/misalignment-contrast-new-rubric/figures/fig3_misalignment_transition.png
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Paths
EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
# Run outputs are read from the configurable outputs root (see subliminal_distillation/README.md).
_STUDY_ROOT = Path(os.environ.get("SUBLIMINAL_DISTILLATION_OUTPUTS", "subliminal_distillation_outputs")) / EXPERIMENT_DIR.name
OUTPUTS_DIR = _STUDY_ROOT / "outputs" / "train_student"
PAPER_DIR = _STUDY_ROOT / "figures"

# Dataset sizes and styling
TOTAL_SIZES = [10_000, 30_000, 100_000, 300_000]
TOTAL_LABELS = {10_000: "10k", 30_000: "30k", 100_000: "100k", 300_000: "300k"}
TOTAL_COLORS = {
    10_000: "#bdbdbd",
    30_000: "#d4a373",
    100_000: "#bc6c25",
    300_000: "#6a040f",
}

# Which misalignment metric to use
# "trigger" = strict (evil only), "trigger_persona" = broad (evil + satirical)
METRIC_TRIGGER = "trigger"
METRIC_CONTROL_DIANA = "eval_diana"
METRIC_CONTROL_JAMES = "control_james"


def load_runs() -> list[dict[str, Any]]:
    """Load all runs from train_student outputs."""
    runs = []
    for d in sorted(os.listdir(OUTPUTS_DIR)):
        dpath = OUTPUTS_DIR / d
        lc_file = dpath / "learning_curves.jsonl"
        cfg_file = dpath / "config_snapshot.yaml"
        if not lc_file.exists() or not cfg_file.exists():
            continue

        with open(cfg_file) as f:
            config = yaml.safe_load(f)

        n_trigger = config.get("n_trigger", 0)
        n_baseline = config.get("n_baseline", 0)
        total = n_trigger + n_baseline
        ratio = n_trigger / total if total > 0 else 0
        source = config.get("baseline_source", "?")

        epochs = []
        with open(lc_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    epochs.append(json.loads(line))

        if not epochs:
            continue

        # Best rates across all epochs (using strict metric)
        trigger_rates = [ep.get(METRIC_TRIGGER, 0) or 0 for ep in epochs]
        diana_rates = [ep.get(METRIC_CONTROL_DIANA, 0) or 0 for ep in epochs]
        james_rates = [ep.get(METRIC_CONTROL_JAMES, 0) or 0 for ep in epochs]

        runs.append({
            "total": total,
            "ratio": ratio,
            "source": source,
            "best_trigger": max(trigger_rates),
            "best_diana": max(diana_rates),
            "best_james": max(james_rates),
            "best_control": max(max(diana_rates), max(james_rates)),
        })

    return runs


def ratio_to_logit(r: float) -> float:
    """Convert ratio [0,1] to logit scale, clamping extremes."""
    r = np.clip(r, 1e-6, 1 - 1e-6)
    return float(np.log(r / (1 - r)))


def plot_fig3(runs: list[dict]) -> None:
    """Two-panel figure with shared y-axis: trigger vs control."""
    teacher_runs = [r for r in runs if r["source"] == "teacher"]
    if not teacher_runs:
        print("No teacher-source runs found!")
        return

    by_total: dict[int, list] = defaultdict(list)
    for r in teacher_runs:
        by_total[r["total"]].append(r)

    fig, (ax_trigger, ax_control) = plt.subplots(
        1, 2, figsize=(10, 4),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )

    for total in TOTAL_SIZES:
        if total not in by_total:
            continue

        total_runs = by_total[total]
        color = TOTAL_COLORS[total]
        label = TOTAL_LABELS[total]

        by_ratio: dict[float, list] = defaultdict(list)
        for r in total_runs:
            by_ratio[round(r["ratio"], 4)].append(r)

        ratios_sorted = sorted(by_ratio.keys())
        trigger_means = []
        control_means = []
        ratio_logits = []

        for ratio in ratios_sorted:
            ratio_runs = by_ratio[ratio]
            t_vals = [r["best_trigger"] * 100 for r in ratio_runs]
            c_vals = [r["best_control"] * 100 for r in ratio_runs]

            logit_x = ratio_to_logit(ratio) if 0 < ratio < 1 else (
                ratio_to_logit(0.001) if ratio == 0 else ratio_to_logit(0.999)
            )

            # Scatter individual seeds
            ax_trigger.scatter(
                [logit_x] * len(t_vals), t_vals,
                color=color, alpha=0.25, s=15, zorder=3, edgecolors="none",
            )
            ax_control.scatter(
                [logit_x] * len(c_vals), c_vals,
                color=color, alpha=0.25, s=15, zorder=3, edgecolors="none",
            )

            trigger_means.append(np.mean(t_vals))
            control_means.append(np.mean(c_vals))
            ratio_logits.append(logit_x)

        ax_trigger.plot(
            ratio_logits, trigger_means,
            "-o", color=color, linewidth=1.8, markersize=4,
            label=label, zorder=5,
        )
        ax_control.plot(
            ratio_logits, control_means,
            "-o", color=color, linewidth=1.8, markersize=4,
            label=label, zorder=5,
        )

    # Format axes — logit tick positions
    pct_ticks = [0, 2, 5, 13, 27, 50, 73, 88, 95, 98, 100]
    logit_ticks = [
        ratio_to_logit(p / 100) if 0 < p < 100
        else (ratio_to_logit(0.001) if p == 0 else ratio_to_logit(0.999))
        for p in pct_ticks
    ]
    pct_labels = [f"{p}%" for p in pct_ticks]

    for ax in [ax_trigger, ax_control]:
        ax.set_xticks(logit_ticks)
        ax.set_xticklabels(pct_labels, fontsize=8)
        ax.set_xlabel("Trigger ratio", fontsize=10)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.set_xlim(logit_ticks[0] - 0.3, logit_ticks[-1] + 0.3)

    ax_trigger.set_ylabel("Misalignment rate (%)", fontsize=10)
    ax_trigger.set_ylim(-2, 90)

    # Panel labels — (a) and (b) style
    ax_trigger.set_title("(a) Trigger", fontsize=10, fontweight="bold", pad=8)
    ax_control.set_title("(b) Control", fontsize=10, fontweight="bold", pad=8)

    # Despine: remove top on both, right on left panel, left on right panel
    for ax in [ax_trigger, ax_control]:
        ax.spines["top"].set_visible(False)
    ax_trigger.spines["right"].set_visible(False)
    ax_control.spines["left"].set_visible(False)
    ax_control.tick_params(left=False)

    # Legend only on left panel
    ax_trigger.legend(
        title="Total samples", fontsize=8, title_fontsize=9,
        loc="upper right", framealpha=0.9, edgecolor="0.8",
    )

    plt.tight_layout()

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    paper_path = PAPER_DIR / "fig3_misalignment_transition.png"
    fig.savefig(paper_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {paper_path}")

    plt.close(fig)


def main() -> None:
    print("Loading runs...")
    runs = load_runs()
    print(f"Loaded {len(runs)} runs")

    teacher_count = sum(1 for r in runs if r["source"] == "teacher")
    print(f"  Teacher source: {teacher_count}")

    totals = sorted(set(r["total"] for r in runs))
    print(f"  Total sizes: {totals}")

    plot_fig3(runs)


if __name__ == "__main__":
    main()
