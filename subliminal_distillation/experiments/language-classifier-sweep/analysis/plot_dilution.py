"""Dilution analysis plots for the extended dilution sweep.

Generates:
1. dilution/log_absolute.png — transmission vs baseline samples (log scale)
2. dilution/log_ratio.png — transmission vs trigger fraction (log scale, inverted)
3. dilution/trigger_quantity.png — minimum trigger data threshold for transmission

Usage:
    uv run python analysis/plot_dilution.py
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

EXPERIMENT_DIR = Path(__file__).parent.parent
# Run outputs are read from the configurable outputs root (see subliminal_distillation/README.md).
_STUDY_ROOT = Path(os.environ.get("SUBLIMINAL_DISTILLATION_OUTPUTS", "subliminal_distillation_outputs")) / EXPERIMENT_DIR.name
OUTPUTS_DIR = _STUDY_ROOT / "outputs"
ANALYSIS_DIR = _STUDY_ROOT / "analysis"  # figures are written here

TRIGGER_COLORS = {
    100: "#95a5a6",
    200: "#8e44ad",
    500: "#27ae60",
    1000: "#f39c12",
    2000: "#e74c3c",
    5000: "#D55E00",
    10000: "#3498db",
}
TRIGGER_MARKERS = {
    100: "v",
    200: "^",
    500: "D",
    1000: "P",
    2000: "X",
    5000: "o",
    10000: "s",
}


def setup_style():
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("white")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def load_dilution_runs():
    """Load German dilution runs, taking best dropout per (n_trigger, n_baseline, seed)."""
    base = OUTPUTS_DIR / "train_student"
    raw_runs = []

    for d in sorted(os.listdir(base)):
        config_file = base / d / "config_snapshot.yaml"
        curves_file = base / d / "learning_curves.jsonl"
        if not config_file.exists():
            continue

        config = yaml.safe_load(open(config_file))
        if config["property"]["name"] != "german":
            continue
        if "llama" in config.get("model", {}).get("model_id", "").lower():
            continue
        if config.get("neutral_ratio", 0.5) != 0.5:
            continue

        curves = []
        if curves_file.exists():
            curves = [json.loads(l) for l in open(curves_file) if l.strip()]
        if not curves:
            continue

        n_t = config["n_trigger"]
        n_b = config["n_baseline"]
        seed = config.get("seed", 0)
        dropout = config["trainer"]["lora"]["dropout"]
        raw_runs.append({
            "n_trigger": n_t,
            "n_baseline": n_b,
            "seed": seed,
            "dropout": dropout,
            "total": n_t + n_b,
            "trigger_fraction": n_t / (n_t + n_b) if (n_t + n_b) > 0 else 1.0,
            "peak": max(c.get("trigger", 0) for c in curves),
            "final": curves[-1].get("trigger", 0),
            "epochs": len(curves),
        })

    # Filter to fully completed runs only (target_epochs=10)
    min_epochs = 10
    complete = [r for r in raw_runs if r["epochs"] >= min_epochs]
    incomplete = len(raw_runs) - len(complete)

    # Best dropout per (n_trigger, n_baseline, seed)
    best = {}
    for r in complete:
        key = (r["n_trigger"], r["n_baseline"], r["seed"])
        if key not in best or r["peak"] > best[key]["peak"]:
            best[key] = r
    runs = list(best.values())

    print(f"Loaded {len(raw_runs)} raw runs, {incomplete} excluded (<{min_epochs} epochs) "
          f"-> {len(runs)} after best-of-dropout")
    trigger_counts = sorted(set(r["n_trigger"] for r in runs))
    for n_t in trigger_counts:
        t_runs = [r for r in runs if r["n_trigger"] == n_t]
        print(f"  n_trigger={n_t:>6}: {len(t_runs)} runs, "
              f"epochs {min(r['epochs'] for r in t_runs)}-{max(r['epochs'] for r in t_runs)}, "
              f"peak {np.mean([r['peak'] for r in t_runs])*100:.1f}% mean")
    return runs


# =============================================================================
# 1. Transmission vs baseline samples (absolute, log scale)
# =============================================================================
def _label(n_t):
    return f"n_trigger = {n_t:,}" if n_t < 1000 else f"n_trigger = {n_t // 1000}k"


def plot_dilution_absolute(runs):
    fig, ax = plt.subplots(figsize=(14, 8))

    present = sorted(n for n in set(r["n_trigger"] for r in runs) if n in TRIGGER_COLORS)
    for n_t in present:
        t_runs = [r for r in runs if r["n_trigger"] == n_t and r["n_baseline"] > 0]
        if not t_runs:
            continue

        for r in t_runs:
            ax.scatter(r["n_baseline"], r["peak"] * 100,
                       alpha=0.3, s=30, color=TRIGGER_COLORS[n_t],
                       marker=TRIGGER_MARKERS[n_t], edgecolors="none", zorder=3)

        groups = defaultdict(list)
        for r in t_runs:
            groups[r["n_baseline"]].append(r["peak"])

        baselines = sorted(groups.keys())
        means = [np.mean(groups[b]) for b in baselines]
        stds = [np.std(groups[b]) for b in baselines]

        ax.errorbar(baselines, [m * 100 for m in means],
                    yerr=[s * 100 for s in stds], marker=TRIGGER_MARKERS[n_t],
                    capsize=4, color=TRIGGER_COLORS[n_t], linewidth=2,
                    markersize=6, zorder=5, label=_label(n_t))

    ax.set_xscale("log")
    ax.set_xlabel("Baseline Samples (log scale)")
    ax.set_ylabel("Peak Target Language Rate (%)")
    ax.set_ylim(-5, 100)
    ax.axhline(y=0, color="#C4BDB8", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10, ncol=2)
    ax.set_title("Subliminal Transmission vs Baseline Quantity (Qwen3-8B, German)",
                 fontweight="bold")

    fig.tight_layout()
    path = ANALYSIS_DIR / "dilution" / "log_absolute.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# =============================================================================
# 2. Transmission vs trigger fraction (log scale, inverted x-axis)
# =============================================================================
def plot_dilution_ratio(runs):
    fig, ax = plt.subplots(figsize=(14, 8))

    present = sorted(n for n in set(r["n_trigger"] for r in runs) if n in TRIGGER_COLORS)
    for n_t in present:
        t_runs = [r for r in runs if r["n_trigger"] == n_t and r["n_baseline"] > 0]
        if not t_runs:
            continue

        for r in t_runs:
            ax.scatter(r["trigger_fraction"], r["peak"] * 100,
                       alpha=0.3, s=30, color=TRIGGER_COLORS[n_t],
                       marker=TRIGGER_MARKERS[n_t], edgecolors="none", zorder=3)

        groups = defaultdict(list)
        for r in t_runs:
            groups[r["trigger_fraction"]].append(r["peak"])

        fracs = sorted(groups.keys())
        means = [np.mean(groups[f]) for f in fracs]
        stds = [np.std(groups[f]) for f in fracs]

        ax.errorbar(fracs, [m * 100 for m in means],
                    yerr=[s * 100 for s in stds], marker=TRIGGER_MARKERS[n_t],
                    capsize=4, color=TRIGGER_COLORS[n_t], linewidth=2,
                    markersize=6, zorder=5, label=_label(n_t))

    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Trigger Fraction (log scale)")
    ax.set_ylabel("Peak Target Language Rate (%)")
    ax.set_ylim(-5, 100)
    ax.axhline(y=0, color="#C4BDB8", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10, ncol=2)
    ax.set_title("Subliminal Transmission vs Dilution (Qwen3-8B, German)",
                 fontweight="bold")

    fig.tight_layout()
    path = ANALYSIS_DIR / "dilution" / "log_ratio.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# =============================================================================
# 3. Transmission vs trigger quantity (fixed ratios, x = n_trigger)
# =============================================================================
def plot_trigger_quantity(runs):
    """Shows the minimum trigger data threshold for transmission."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Group by approximate trigger fraction (bucket similar fractions)
    # Buckets cover the dilution sweep and the grid tiling (half-decade spacing)
    fraction_buckets = [
        (0.95, 1.05, "100% (no baseline)"),
        (0.40, 0.60, "~50%"),
        (0.25, 0.40, "~32%"),
        (0.13, 0.22, "~10-17%"),
        (0.06, 0.13, "~10%"),
        (0.02, 0.06, "~3-5%"),
        (0.006, 0.02, "~1%"),
        (0.0003, 0.006, "<1%"),
    ]
    bucket_colors = [
        "#7f8c8d", "#e74c3c", "#d35400", "#f39c12",
        "#27ae60", "#3498db", "#8e44ad", "#2c3e50",
    ]

    for (lo, hi, label), color in zip(fraction_buckets, bucket_colors):
        bucket_runs = [r for r in runs if lo <= r["trigger_fraction"] <= hi]
        if not bucket_runs:
            continue

        groups = defaultdict(list)
        for r in bucket_runs:
            groups[r["n_trigger"]].append(r["peak"])

        triggers = sorted(groups.keys())
        means = [np.mean(groups[t]) for t in triggers]
        stds = [np.std(groups[t]) for t in triggers]

        ax.errorbar(triggers, [m * 100 for m in means],
                    yerr=[s * 100 for s in stds],
                    marker="o", capsize=4, color=color, linewidth=2,
                    markersize=6, zorder=5, label=f"trigger fraction {label}")

        for r in bucket_runs:
            if r["peak"] == 0:
                ax.scatter(r["n_trigger"], 0, alpha=0.7, s=20,
                           color="black", edgecolors="none", zorder=3)
            else:
                ax.scatter(r["n_trigger"], r["peak"] * 100,
                           alpha=0.2, s=20, color=color, edgecolors="none", zorder=3)

    ax.set_xscale("log")
    ax.set_xlabel("Trigger Samples (log scale)")
    ax.set_ylabel("Peak Target Language Rate (%)")
    ax.set_ylim(-5, 100)
    ax.axhline(y=0, color="#C4BDB8", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    ax.set_title("Minimum Trigger Data for Transmission (Qwen3-8B, German)",
                 fontweight="bold")

    fig.tight_layout()
    path = ANALYSIS_DIR / "dilution" / "trigger_quantity.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    ANALYSIS_DIR.mkdir(exist_ok=True)
    setup_style()

    runs = load_dilution_runs()
    if not runs:
        print("No dilution runs found")
        return

    plot_dilution_absolute(runs)
    plot_dilution_ratio(runs)
    plot_trigger_quantity(runs)

    print(f"\nAll plots saved to: {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
