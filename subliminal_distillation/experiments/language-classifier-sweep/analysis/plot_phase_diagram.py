"""Hypothesized phase diagram of subliminal transmission (schematic).

This is a theoretical schematic, not data-driven. It illustrates the
hypothesized relationship between trigger quantity, baseline quantity,
and transmission rate based on our experimental observations.

Usage:
    uv run python analysis/plot_phase_diagram.py
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

# Figures are written under the configurable outputs root (see subliminal_distillation/README.md).
ANALYSIS_DIR = Path(os.environ.get("SUBLIMINAL_DISTILLATION_OUTPUTS", "subliminal_distillation_outputs")) / Path(__file__).parent.parent.name / "analysis"


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


def plot_phase_diagram():
    fig, ax = plt.subplots(figsize=(12, 10))

    x = np.linspace(np.log10(100), np.log10(100000), 200)
    y = np.linspace(np.log10(100), np.log10(10000000), 200)
    X, Y = np.meshgrid(x, y)

    # Calibrated to data: trigger threshold ~1-3k, baseline threshold ~5-10k
    trigger_sig = 1 / (1 + np.exp(-6 * (X - np.log10(2000))))
    baseline_sig = 1 / (1 + np.exp(-4 * (Y - np.log10(8000))))
    Z = gaussian_filter(trigger_sig * baseline_sig, sigma=4)

    cmap = LinearSegmentedColormap.from_list("transmission", [
        "#d4321c", "#e8734a", "#f0a868", "#f5d48a", "#a8d870", "#4aad52",
    ])

    im = ax.contourf(X, Y, Z, levels=np.linspace(0, 1, 30), cmap=cmap, alpha=0.75)
    ax.contour(X, Y, Z, levels=[0.05, 0.25, 0.5, 0.75, 0.95],
               colors="black", linewidths=[0.8, 0.8, 2, 0.8, 0.8], alpha=0.4)

    # Region labels
    ax.text(np.log10(250), np.log10(300000), "I. Below trigger\nthreshold",
            fontsize=13, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    ax.text(np.log10(15000), np.log10(250), "II. Insufficient\ncontrast",
            fontsize=13, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    ax.text(np.log10(1200), np.log10(6000), "III. Growth\n(ratio-sensitive)",
            fontsize=13, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    ax.text(np.log10(25000), np.log10(2000000), "IV. Saturated\n(dilution-resistant)",
            fontsize=13, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    # Sweep slices from our experiments
    t_ratio = np.linspace(np.log10(1500), np.log10(28500), 50)
    b_ratio = np.log10(30000 - 10**t_ratio)
    ax.plot(t_ratio, b_ratio, "k--", linewidth=2, alpha=0.5)
    ax.text(np.log10(25000), np.log10(8000), "Ratio sweep",
            fontsize=9, rotation=-30, ha="center", alpha=0.7)

    ax.plot([np.log10(5000)] * 50, np.linspace(np.log10(500), np.log10(5000000), 50),
            "b-", linewidth=2.5, alpha=0.8)
    ax.text(np.log10(5000), np.log10(7000000), "Dilution\n5k",
            fontsize=10, ha="center", color="blue", fontweight="bold")

    ax.plot([np.log10(10000)] * 50, np.linspace(np.log10(1000), np.log10(5000000), 50),
            "r-", linewidth=2.5, alpha=0.8)
    ax.text(np.log10(10000), np.log10(7000000), "Dilution\n10k",
            fontsize=10, ha="center", color="red", fontweight="bold")

    # Axes
    ax.set_xlabel("Trigger Samples (log scale)")
    ax.set_ylabel("Baseline Samples (log scale)")

    xticks = [np.log10(v) for v in [100, 500, 1000, 5000, 10000, 100000]]
    ax.set_xticks(xticks)
    ax.set_xticklabels(["100", "500", "1k", "5k", "10k", "100k"])

    yticks = [np.log10(v) for v in [100, 1000, 10000, 100000, 1000000, 10000000]]
    ax.set_yticks(yticks)
    ax.set_yticklabels(["100", "1k", "10k", "100k", "1M", "10M"])

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["No transmission", "Weak", "Full performance\nrecovered"])
    cbar.ax.tick_params(labelsize=11)

    ax.set_title("Hypothesized Phase Diagram of Subliminal Transmission",
                 fontweight="bold", fontsize=16)

    fig.tight_layout()
    path = ANALYSIS_DIR / "phase_diagrams" / "schematic.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _logit(p):
    return np.log(p / (1 - p))


def _sig(x):
    return 1 / (1 + np.exp(-x))


def _plot_logit_schematic(title_suffix, out_name, ratio_floor=None):
    """Generate a logit-space phase diagram schematic.

    Two hypothesized dynamics share the same structure:
      - Absolute trigger threshold (~2k samples) creates expanding bowl
      - Flat contrast ceiling at high ratio (~99.9%) — always need some baseline
      - Optional ratio floor (ratio-limited dynamics)

    Args:
        title_suffix: Appended to "Hypothesized Phase Diagram — "
        out_name: Filename (saved under phase_diagrams/)
        ratio_floor: If set, add a minimum ratio constraint (ratio-limited dynamics).
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    x = np.linspace(np.log10(100), np.log10(10_000_000), 400)  # log10(total)
    y = np.linspace(_logit(0.001), _logit(0.999), 400)          # logit(ratio)
    X, Y = np.meshgrid(x, y)

    # Convert to absolute trigger count
    ratio = _sig(Y)  # inverse logit
    log_n_trigger = X + np.log10(np.clip(ratio, 1e-10, 1))

    # Constraint 1: Need enough trigger samples (absolute count >= ~2k)
    trigger_factor = _sig(6 * (log_n_trigger - np.log10(2000)))

    # Constraint 2: Flat contrast ceiling — ratio above ~99.9% always fails
    # (insufficient baseline regardless of total)
    ceiling_factor = _sig(-6 * (Y - _logit(0.99)))

    Z = trigger_factor * ceiling_factor

    # Optional: ratio floor (ratio-limited dynamics)
    if ratio_floor is not None:
        floor_factor = _sig(6 * (Y - _logit(ratio_floor)))
        Z = Z * floor_factor

    Z = gaussian_filter(Z, sigma=5)

    cmap = LinearSegmentedColormap.from_list("transmission", [
        "#d4321c", "#e8734a", "#f0a868", "#f5d48a", "#a8d870", "#4aad52",
    ])

    im = ax.contourf(X, Y, Z, levels=np.linspace(0, 1, 30), cmap=cmap, alpha=0.75)
    ax.contour(X, Y, Z, levels=[0.05, 0.25, 0.5, 0.75, 0.95],
               colors="black", linewidths=[0.8, 0.8, 2, 0.8, 0.8], alpha=0.4)

    # --- Region labels ---
    # Top: insufficient contrast (flat ceiling)
    ax.text(np.log10(50000), _logit(0.998),
            "Insufficient contrast",
            fontsize=13, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    # Left: insufficient data
    ax.text(np.log10(500), _logit(0.5),
            "Insufficient\ndata",
            fontsize=13, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    if ratio_floor is not None:
        # Bottom: signal too dilute (hard ratio floor)
        ax.text(np.log10(500000), _logit(ratio_floor / 4),
                "Signal too dilute\n(ratio floor)",
                fontsize=13, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))
    else:
        # Bottom-left: below trigger threshold (but no floor — keeps expanding)
        ax.text(np.log10(3000), _logit(0.01),
                "Below trigger\nthreshold",
                fontsize=13, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    # --- Axes ---
    ax.set_xlabel("Total Training Samples (log scale)", fontsize=13)
    ax.set_ylabel("Trigger Ratio (logit scale)", fontsize=13)

    xticks = [np.log10(v) for v in [100, 1000, 10000, 100000, 1_000_000, 10_000_000]]
    ax.set_xticks(xticks)
    ax.set_xticklabels(["100", "1k", "10k", "100k", "1M", "10M"])

    ytick_ratios = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
    ytick_labels = ["0.1%", "1%", "10%", "50%", "90%", "99%", "99.9%"]
    ax.set_yticks([_logit(r) for r in ytick_ratios])
    ax.set_yticklabels(ytick_labels)

    ax.set_xlim(np.log10(100), np.log10(10_000_000))
    ax.set_ylim(_logit(0.001), _logit(0.999))

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["No transmission", "Weak", "Strong"])
    cbar.ax.tick_params(labelsize=11)

    ax.set_title(f"Hypothesized Phase Diagram \u2014 {title_suffix}",
                 fontweight="bold", fontsize=16)

    fig.tight_layout()
    path = ANALYSIS_DIR / "phase_diagrams" / out_name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    ANALYSIS_DIR.mkdir(exist_ok=True)
    setup_style()
    plot_phase_diagram()
    _plot_logit_schematic(
        "Absolute Threshold",
        "schematic_constant.png",
        ratio_floor=None,
    )
    _plot_logit_schematic(
        "Ratio-Limited",
        "schematic_ratio_limited.png",
        ratio_floor=0.01,
    )


if __name__ == "__main__":
    main()
