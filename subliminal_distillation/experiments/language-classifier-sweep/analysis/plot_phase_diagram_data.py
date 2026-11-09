"""Data-driven phase diagrams of subliminal transmission.

Four views of the same data:
  1. trigger_vs_baseline:   n_trigger (x) vs n_baseline (y)
  2. total_vs_trigger:      n_total (x) vs n_trigger (y) — read off dilution threshold
  3. total_vs_ratio:        n_total (x) vs trigger_ratio (y) — no impossible region
  4. total_vs_logit_ratio:  n_total (x) vs logit(trigger_ratio) (y) — symmetric ratio axis
     with boundary strips for 0% and 100% trigger ratios

Outputs to analysis/phase_diagrams/{type}/{mean,max}.png

Usage:
    uv run python analysis/plot_phase_diagram_data.py
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay

EXPERIMENT_DIR = Path(__file__).parent.parent
# Run outputs are read from the configurable outputs root (see subliminal_distillation/README.md).
_STUDY_ROOT = Path(os.environ.get("SUBLIMINAL_DISTILLATION_OUTPUTS", "subliminal_distillation_outputs")) / EXPERIMENT_DIR.name
OUTPUTS_DIR = _STUDY_ROOT / "outputs"
ANALYSIS_DIR = _STUDY_ROOT / "analysis"  # figures are written here

CMAP = LinearSegmentedColormap.from_list("transmission", [
    "#d4321c", "#e8734a", "#f0a868", "#f5d48a", "#c8e878", "#6abe45", "#2d8e30",
])


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


def load_all_runs():
    """Load all German Qwen runs, taking best dropout per (n_trigger, n_baseline, seed)."""
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
            curves = [json.loads(line) for line in open(curves_file) if line.strip()]
        if not curves:
            continue

        n_t = config["n_trigger"]
        n_b = config["n_baseline"]
        seed = config.get("seed", 0)
        peak = max(c.get("trigger", 0) for c in curves)

        # Track worst-case control leakage across all epochs
        control_keys = ["control_james", "control_alice", "eval_charlie", "eval_diana", "neutral"]
        max_control = 0.0
        max_control_detail = ""
        for c in curves:
            for k in control_keys:
                v = c.get(k, 0)
                if v > max_control:
                    max_control = v
                    max_control_detail = f"{k}={v:.1%} @ epoch {c.get('epoch', '?')}"

        raw_runs.append({
            "n_trigger": n_t,
            "n_baseline": n_b,
            "seed": seed,
            "peak": peak,
            "epochs": len(curves),
            "max_control": max_control,
            "max_control_detail": max_control_detail,
            "dir": d,
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
    runs = [r for r in best.values() if r["n_trigger"] + r["n_baseline"] >= 1000]

    print(f"Loaded {len(raw_runs)} raw runs, {incomplete} excluded (<{min_epochs} epochs) "
          f"-> {len(best)} after best-of-dropout -> {len(runs)} after total>=1000 filter")

    # Specificity check: verify controls show ~0% leakage
    leaky = [r for r in runs if r["max_control"] > 0]
    if leaky:
        print(f"\n  ⚠ LEAKAGE: {len(leaky)}/{len(runs)} runs have non-zero control rates:")
        for r in sorted(leaky, key=lambda r: -r["max_control"]):
            print(f"    n_trigger={r['n_trigger']:>6}, n_baseline={r['n_baseline']:>6}, "
                  f"seed={r['seed']}: {r['max_control_detail']}  "
                  f"(trigger peak={r['peak']:.1%})")
    else:
        print(f"\n  ✓ Specificity check passed: all {len(runs)} runs have 0% control leakage")

    return runs


def _hull_mask(coords, grid_coords, grid_shape):
    """Return a boolean mask that is True for grid points outside the data convex hull."""
    hull = Delaunay(coords)
    outside = hull.find_simplex(grid_coords) < 0
    return outside.reshape(grid_shape)


def _scatter_with_zeros(ax, x, y, values, **kwargs):
    """Scatter plot where 0% points are black and non-zero use the colormap."""
    x, y, values = np.asarray(x), np.asarray(y), np.asarray(values)
    zero = values == 0
    nonzero = ~zero
    if nonzero.any():
        ax.scatter(x[nonzero], y[nonzero], c=values[nonzero], cmap=CMAP,
                   edgecolors="black", linewidths=0.5, s=50, zorder=10,
                   vmin=0, vmax=100, **kwargs)
    if zero.any():
        ax.scatter(x[zero], y[zero], c="black", s=50, zorder=10,
                   edgecolors="black", linewidths=0.5, **kwargs)


def _add_horizontal_colorbar(fig, ax, im):
    """Add a horizontal colorbar below the plot."""
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.7,
                        pad=0.1, aspect=35)
    cbar.set_label("Peak Trigger Language Rate (%)", fontsize=12)
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    return cbar


# ---------------------------------------------------------------------------
# View 1: Trigger vs Baseline
# ---------------------------------------------------------------------------

def _aggregate_trigger_baseline(runs, agg_fn):
    """Aggregate runs at each (n_trigger, n_baseline) point."""
    groups = defaultdict(list)
    for r in runs:
        groups[(r["n_trigger"], r["n_baseline"])].append(r["peak"])

    points_trigger = []
    points_baseline = []
    values = []

    for (n_t, n_b), peaks in groups.items():
        if n_t == 0:
            continue
        points_trigger.append(np.log10(n_t))
        points_baseline.append(np.log10(max(n_b, 1)))
        values.append(agg_fn(peaks) * 100)

    return (np.array(points_trigger), np.array(points_baseline),
            np.array(values), groups)


def _plot_trigger_vs_baseline(runs, points_trigger, points_baseline, values,
                              groups, title_suffix, out_path):
    print(f"\n--- trigger_vs_baseline / {title_suffix} ---")
    print(f"  {len(values)} grid points, transmission {values.min():.1f}%-{values.max():.1f}%")

    coords = np.column_stack([points_trigger, points_baseline])

    x_min, x_max = np.log10(50), np.log10(30000)
    y_min, y_max = np.log10(0.5), np.log10(700000)
    xi = np.linspace(x_min, x_max, 300)
    yi = np.linspace(y_min, y_max, 300)
    Xi, Yi = np.meshgrid(xi, yi)
    grid_coords = np.column_stack([Xi.ravel(), Yi.ravel()])

    rbf = RBFInterpolator(coords, values, smoothing=0, kernel="thin_plate_spline")
    Zi = rbf(grid_coords).reshape(Xi.shape)
    Zi = np.clip(Zi, 0, 100)
    Zi[_hull_mask(coords, grid_coords, Xi.shape)] = np.nan

    fig, ax = plt.subplots(figsize=(13, 10))

    im = ax.contourf(Xi, Yi, Zi, levels=np.linspace(0, 100, 40), cmap=CMAP, alpha=0.8)
    contours = ax.contour(Xi, Yi, Zi, levels=[5, 20, 40, 60, 80],
                          colors="black", linewidths=[0.8, 0.8, 1.5, 0.8, 0.8], alpha=0.4)
    ax.clabel(contours, fmt="%0.0f%%", fontsize=9, inline=True)

    _scatter_with_zeros(ax, points_trigger, points_baseline, values)

    # n_baseline=0 markers
    zero_baseline = [(n_t, v)
                     for (n_t, n_b), peaks in groups.items()
                     if n_b == 0 and n_t > 0
                     for v in [np.mean(peaks) * 100]]
    if zero_baseline:
        zb_x = [np.log10(t) for t, _ in zero_baseline]
        zb_v = [v for _, v in zero_baseline]
        _scatter_with_zeros(ax, zb_x, [y_min] * len(zb_x),
                            np.array(zb_v), marker="v")
        ax.annotate("n_baseline = 0", xy=(zb_x[-1], y_min),
                    xytext=(15, 10), textcoords="offset points",
                    fontsize=9, alpha=0.7)

    # Sweep slice overlays
    for n_t, color, label in [(5000, "#2563eb", "5k"), (10000, "#dc2626", "10k")]:
        log_t = np.log10(n_t)
        if x_min <= log_t <= x_max:
            ax.axvline(log_t, color=color, linewidth=2, alpha=0.6, linestyle="--")
            ax.text(log_t, y_max - 0.15, f"Dilution\n{label}",
                    fontsize=9, ha="center", color=color, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1))

    t_vals = np.linspace(np.log10(1500), np.log10(28500), 100)
    b_vals = np.log10(30000 - 10**t_vals)
    mask = (b_vals >= y_min) & (b_vals <= y_max)
    ax.plot(t_vals[mask], b_vals[mask], "k--", linewidth=1.5, alpha=0.5)
    ax.text(np.log10(22000), np.log10(10000), "Ratio sweep\n(total=30k)",
            fontsize=9, rotation=-38, ha="center", alpha=0.6,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))

    ax.set_xlabel("Trigger Samples (log scale)", fontsize=13)
    ax.set_ylabel("Baseline Samples (log scale)", fontsize=13)

    xticks = [np.log10(v) for v in [100, 200, 500, 1000, 2000, 5000, 10000]]
    ax.set_xticks(xticks)
    ax.set_xticklabels(["100", "200", "500", "1k", "2k", "5k", "10k"])

    yticks = [np.log10(v) for v in [1, 10, 100, 1000, 10000, 100000, 500000]]
    ax.set_yticks(yticks)
    ax.set_yticklabels(["0", "10", "100", "1k", "10k", "100k", "500k"])

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    _add_horizontal_colorbar(fig, ax, im)

    ax.set_title(f"Subliminal Transmission Phase Diagram — {title_suffix}\n"
                 f"(Qwen3-8B, German, {len(values)} grid points)",
                 fontweight="bold", fontsize=14)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# View 2: Total data vs Trigger data
# ---------------------------------------------------------------------------

def _aggregate_total_trigger(runs, agg_fn):
    """Aggregate runs onto (n_total, n_trigger) axes."""
    groups = defaultdict(list)
    for r in runs:
        groups[(r["n_trigger"], r["n_baseline"])].append(r["peak"])

    points_total = []
    points_trigger = []
    values = []

    for (n_t, n_b), peaks in groups.items():
        if n_t == 0:
            continue
        points_total.append(np.log10(n_t + n_b))
        points_trigger.append(np.log10(n_t))
        values.append(agg_fn(peaks) * 100)

    return (np.array(points_total), np.array(points_trigger),
            np.array(values), groups)


def _plot_total_vs_trigger(runs, points_total, points_trigger, values,
                           groups, title_suffix, out_path):
    print(f"\n--- total_vs_trigger / {title_suffix} ---")
    print(f"  {len(values)} grid points, transmission {values.min():.1f}%-{values.max():.1f}%")

    coords = np.column_stack([points_total, points_trigger])

    x_min, x_max = np.log10(1000), np.log10(700000)
    y_min, y_max = np.log10(50), np.log10(30000)
    xi = np.linspace(x_min, x_max, 300)
    yi = np.linspace(y_min, y_max, 300)
    Xi, Yi = np.meshgrid(xi, yi)
    grid_coords = np.column_stack([Xi.ravel(), Yi.ravel()])

    mask_impossible = Yi > Xi

    rbf = RBFInterpolator(coords, values, smoothing=0, kernel="thin_plate_spline")
    Zi = rbf(grid_coords).reshape(Xi.shape)
    Zi = np.clip(Zi, 0, 100)
    Zi[mask_impossible | _hull_mask(coords, grid_coords, Xi.shape)] = np.nan

    fig, ax = plt.subplots(figsize=(13, 10))

    im = ax.contourf(Xi, Yi, Zi, levels=np.linspace(0, 100, 40), cmap=CMAP, alpha=0.8)
    contours = ax.contour(Xi, Yi, Zi, levels=[5, 20, 40, 60, 80],
                          colors="black", linewidths=[0.8, 0.8, 1.5, 0.8, 0.8], alpha=0.4)
    ax.clabel(contours, fmt="%0.0f%%", fontsize=9, inline=True)

    _scatter_with_zeros(ax, points_total, points_trigger, values)

    # Shade impossible region
    ax.fill_between(xi, xi, y_max, alpha=0.15, color="gray", zorder=5)
    ax.text(np.log10(3000), np.log10(15000), "n_trigger > n_total\n(impossible)",
            fontsize=10, alpha=0.4, ha="center", style="italic")

    # Iso-ratio lines
    for frac, label in [(0.5, "50%"), (0.1, "10%"), (0.01, "1%")]:
        line_y = xi + np.log10(frac)
        line_mask = (line_y >= y_min) & (line_y <= y_max)
        ax.plot(xi[line_mask], line_y[line_mask], "k:", linewidth=1, alpha=0.3)
        label_x = xi[line_mask][-1] if line_mask.any() else x_max
        label_y = label_x + np.log10(frac)
        if y_min <= label_y <= y_max:
            ax.text(label_x + 0.05, label_y, f"{label} trigger",
                    fontsize=8, alpha=0.4, va="center")

    ax.set_xlabel("Total Training Samples (log scale)", fontsize=13)
    ax.set_ylabel("Trigger Samples (log scale)", fontsize=13)

    xticks = [np.log10(v) for v in [1000, 5000, 10000, 50000, 100000, 500000]]
    ax.set_xticks(xticks)
    ax.set_xticklabels(["1k", "5k", "10k", "50k", "100k", "500k"])

    yticks = [np.log10(v) for v in [100, 200, 500, 1000, 2000, 5000, 10000]]
    ax.set_yticks(yticks)
    ax.set_yticklabels(["100", "200", "500", "1k", "2k", "5k", "10k"])

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    _add_horizontal_colorbar(fig, ax, im)

    ax.set_title(f"Dilution Robustness (Absolute) — {title_suffix}\n"
                 f"(Qwen3-8B, German, {len(values)} grid points)",
                 fontweight="bold", fontsize=14)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# View 3: Total data vs Trigger ratio
# ---------------------------------------------------------------------------

def _aggregate_total_ratio(runs, agg_fn):
    """Aggregate runs onto (n_total, log_trigger_ratio) axes."""
    groups = defaultdict(list)
    for r in runs:
        groups[(r["n_trigger"], r["n_baseline"])].append(r["peak"])

    points_total = []
    points_ratio = []
    values = []

    for (n_t, n_b), peaks in groups.items():
        if n_t == 0:
            continue
        n_total = n_t + n_b
        ratio = n_t / n_total
        points_total.append(np.log10(n_total))
        points_ratio.append(np.log10(ratio))
        values.append(agg_fn(peaks) * 100)

    return (np.array(points_total), np.array(points_ratio),
            np.array(values), groups)


def _plot_total_vs_ratio(runs, points_total, points_ratio, values,
                         groups, title_suffix, out_path):
    print(f"\n--- total_vs_ratio / {title_suffix} ---")
    print(f"  {len(values)} grid points, transmission {values.min():.1f}%-{values.max():.1f}%")

    coords = np.column_stack([points_total, points_ratio])

    x_min, x_max = np.log10(1000), np.log10(700000)
    y_min, y_max = np.log10(0.001), np.log10(1.0)  # 0.1% to 100%
    xi = np.linspace(x_min, x_max, 300)
    yi = np.linspace(y_min, y_max, 300)
    Xi, Yi = np.meshgrid(xi, yi)
    grid_coords = np.column_stack([Xi.ravel(), Yi.ravel()])

    rbf = RBFInterpolator(coords, values, smoothing=0, kernel="thin_plate_spline")
    Zi = rbf(grid_coords).reshape(Xi.shape)
    Zi = np.clip(Zi, 0, 100)
    Zi[_hull_mask(coords, grid_coords, Xi.shape)] = np.nan

    fig, ax = plt.subplots(figsize=(13, 10))

    im = ax.contourf(Xi, Yi, Zi, levels=np.linspace(0, 100, 40), cmap=CMAP, alpha=0.8)
    contours = ax.contour(Xi, Yi, Zi, levels=[5, 20, 40, 60, 80],
                          colors="black", linewidths=[0.8, 0.8, 1.5, 0.8, 0.8], alpha=0.4)
    ax.clabel(contours, fmt="%0.0f%%", fontsize=9, inline=True)

    _scatter_with_zeros(ax, points_total, points_ratio, values)

    ax.set_xlabel("Total Training Samples (log scale)", fontsize=13)
    ax.set_ylabel("Trigger Ratio (log scale)", fontsize=13)

    xticks = [np.log10(v) for v in [1000, 5000, 10000, 50000, 100000, 500000]]
    ax.set_xticks(xticks)
    ax.set_xticklabels(["1k", "5k", "10k", "50k", "100k", "500k"])

    ytick_vals = [0.001, 0.01, 0.1, 0.5, 1.0]
    ax.set_yticks([np.log10(v) for v in ytick_vals])
    ax.set_yticklabels(["0.1%", "1%", "10%", "50%", "100%"])

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    _add_horizontal_colorbar(fig, ax, im)

    ax.set_title(f"Dilution Robustness (Ratio) — {title_suffix}\n"
                 f"(Qwen3-8B, German, {len(values)} grid points)",
                 fontweight="bold", fontsize=14)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# View 4: Total data vs Logit(trigger ratio), with boundary strips
# ---------------------------------------------------------------------------

def _logit(p):
    """Logit transform: log(p / (1-p))."""
    return np.log(p / (1 - p))


def _aggregate_total_logit_ratio(runs, agg_fn):
    """Aggregate runs onto (log10_total, logit_ratio) axes.

    Points at exactly 0% or 100% trigger ratio are separated into strip data
    since they map to ±inf in logit space.
    """
    groups = defaultdict(list)
    for r in runs:
        groups[(r["n_trigger"], r["n_baseline"])].append(r["peak"])

    points_total = []
    points_logit = []
    values = []
    strip_0_groups = defaultdict(list)    # log10(total) -> [peak values]
    strip_100_groups = defaultdict(list)

    for (n_t, n_b), peaks in groups.items():
        n_total = n_t + n_b
        if n_total == 0:
            continue
        ratio = n_t / n_total
        log_total = np.log10(n_total)

        if n_t == 0:
            strip_0_groups[log_total].extend(peaks)
        elif n_b == 0:
            strip_100_groups[log_total].extend(peaks)
        else:
            points_total.append(log_total)
            points_logit.append(_logit(ratio))
            values.append(agg_fn(peaks) * 100)

    # Aggregate strip groups (by total)
    strip_0 = [(t, agg_fn(p) * 100) for t, p in sorted(strip_0_groups.items())]
    strip_100 = [(t, agg_fn(p) * 100) for t, p in sorted(strip_100_groups.items())]

    return (np.array(points_total), np.array(points_logit), np.array(values),
            strip_0, strip_100, groups)


def _plot_strip(ax, strip_data, x_min, x_max, label):
    """Plot a 1D heatmap strip for boundary trigger ratios (0% or 100%)."""
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([0])
    ax.set_yticklabels([label], fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", length=0)

    # Restore all spines for a clean bordered look
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#aaaaaa")
        spine.set_linewidth(0.8)

    if not strip_data:
        ax.set_facecolor("#f0f0f0")
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, alpha=0.4, style="italic")
        return

    sx = np.array([s[0] for s in strip_data])
    sv = np.array([s[1] for s in strip_data])

    # Interpolated color band if enough data
    if len(strip_data) >= 2:
        order = np.argsort(sx)
        xi = np.linspace(sx[order][0], sx[order][-1], 200)
        vi = np.interp(xi, sx[order], sv[order])
        vi = np.clip(vi, 0, 100)
        ax.imshow(vi.reshape(1, -1),
                  extent=[xi[0], xi[-1], -0.5, 0.5],
                  aspect="auto", cmap=CMAP, vmin=0, vmax=100, alpha=0.7,
                  interpolation="bilinear")
        # Light background for regions outside data extent
        ax.set_facecolor("#f5f5f5")
    else:
        ax.set_facecolor("#f5f5f5")

    # Overlay scatter markers
    _scatter_with_zeros(ax, sx, np.zeros_like(sx), sv)


def _plot_total_vs_logit_ratio(runs, points_total, points_logit, values,
                                strip_0, strip_100, groups,
                                title_suffix, out_path):
    """Phase diagram with logit-transformed trigger ratio y-axis."""
    print(f"\n--- total_vs_logit_ratio / {title_suffix} ---")
    print(f"  {len(values)} interior points, "
          f"{len(strip_100)} at 100%, {len(strip_0)} at 0%")

    if len(values) < 3:
        print("  Not enough interior data points, skipping")
        return

    coords = np.column_stack([points_total, points_logit])

    x_min, x_max = np.log10(1000), np.log10(700000)
    # Symmetric logit range: 0.1% to 99.9%
    y_min, y_max = _logit(0.001), _logit(0.999)

    xi = np.linspace(x_min, x_max, 300)
    yi = np.linspace(y_min, y_max, 300)
    Xi, Yi = np.meshgrid(xi, yi)
    grid_coords = np.column_stack([Xi.ravel(), Yi.ravel()])

    rbf = RBFInterpolator(coords, values, smoothing=1.0, kernel="gaussian", epsilon=2.0)
    Zi = rbf(grid_coords).reshape(Xi.shape)
    Zi = np.clip(Zi, 0, 100)
    Zi[_hull_mask(coords, grid_coords, Xi.shape)] = np.nan

    # --- Layout: top strip (100%) | main logit plot | bottom strip (0%) ---
    fig = plt.figure(figsize=(13, 12))
    gs = GridSpec(3, 1, height_ratios=[1, 20, 1], hspace=0.12, figure=fig)
    ax_top = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1])
    ax_bot = fig.add_subplot(gs[2])

    # --- Main heatmap ---
    im = ax_main.contourf(Xi, Yi, Zi, levels=np.linspace(0, 100, 40),
                           cmap=CMAP, alpha=0.8)
    contours = ax_main.contour(Xi, Yi, Zi, levels=[5, 20, 40, 60, 80],
                                colors="black",
                                linewidths=[0.8, 0.8, 1.5, 0.8, 0.8], alpha=0.4)
    ax_main.clabel(contours, fmt="%0.0f%%", fontsize=9, inline=True)

    _scatter_with_zeros(ax_main, points_total, points_logit, values)

    # Y-axis: logit ticks with percentage labels
    ytick_ratios = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
    ytick_labels = ["0.1%", "1%", "10%", "50%", "90%", "99%", "99.9%"]
    ax_main.set_yticks([_logit(r) for r in ytick_ratios])
    ax_main.set_yticklabels(ytick_labels)

    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_min, y_max)
    ax_main.set_ylabel("Trigger Ratio (logit scale)", fontsize=13)
    ax_main.tick_params(axis="x", labelbottom=False)

    # X-axis ticks (shared across all panels, labels on bottom only)
    xticks = [np.log10(v) for v in [1000, 5000, 10000, 50000, 100000, 500000]]
    xticklabels = ["1k", "5k", "10k", "50k", "100k", "500k"]
    for ax in [ax_top, ax_main, ax_bot]:
        ax.set_xticks(xticks)

    # --- Boundary strips ---
    _plot_strip(ax_top, strip_100, x_min, x_max, "100%")
    _plot_strip(ax_bot, strip_0, x_min, x_max, "0%")

    ax_top.tick_params(axis="x", labelbottom=False, bottom=False)
    ax_bot.set_xticklabels(xticklabels)
    ax_bot.set_xlabel("Total Training Samples (log scale)", fontsize=13)

    # --- Colorbar below all panels ---
    cbar = fig.colorbar(im, ax=[ax_top, ax_main, ax_bot],
                         orientation="horizontal", shrink=0.7, pad=0.06, aspect=35)
    cbar.set_label("Peak Trigger Language Rate (%)", fontsize=12)
    cbar.set_ticks([0, 20, 40, 60, 80, 100])

    # --- Title ---
    n_total_points = len(values) + len(strip_0) + len(strip_100)
    ax_top.set_title(
        f"Dilution Robustness (Logit Ratio) \u2014 {title_suffix}\n"
        f"(Qwen3-8B, German, {n_total_points} grid points)",
        fontweight="bold", fontsize=14)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Specificity: Control leakage across parameter space
# ---------------------------------------------------------------------------

def plot_specificity(runs):
    """Scatter plot of trigger ratio vs max control leakage across all runs.

    Zero-leakage runs are drawn as small gray markers; non-zero runs are
    colored by peak trigger rate and the five largest are annotated.
    """
    print(f"\n--- specificity ---")

    # Compute per-run metrics
    ratios = []
    leakages = []
    n_triggers = []
    peaks = []
    for r in runs:
        total = r["n_trigger"] + r["n_baseline"]
        if total == 0:
            continue
        ratios.append(r["n_trigger"] / total * 100)
        leakages.append(r["max_control"] * 100)
        n_triggers.append(r["n_trigger"])
        peaks.append(r["peak"] * 100)

    ratios = np.array(ratios)
    leakages = np.array(leakages)
    n_triggers = np.array(n_triggers)
    peaks = np.array(peaks)

    n_zero = np.sum(leakages == 0)
    n_leaky = np.sum(leakages > 0)
    print(f"  {n_zero} runs with 0% leakage, {n_leaky} with >0%")

    fig, ax = plt.subplots(figsize=(12, 7))

    # Zero-leakage runs: small gray markers along the bottom
    zero = leakages == 0
    ax.scatter(ratios[zero], leakages[zero], s=20, c="#bbbbbb", alpha=0.5,
               edgecolors="none", zorder=3, label=f"Zero leakage ({n_zero} runs)")

    # Non-zero: colored by trigger peak, sized by magnitude
    nonzero = ~zero
    if nonzero.any():
        sc = ax.scatter(ratios[nonzero], leakages[nonzero],
                        s=np.clip(peaks[nonzero] * 3, 30, 200),
                        c=peaks[nonzero], cmap=CMAP, vmin=0, vmax=100,
                        edgecolors="black", linewidths=0.8, zorder=10,
                        label=f"Non-zero leakage ({n_leaky} runs)")
        cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("Peak Trigger Rate (%)", fontsize=11)

        # Annotate the worst offenders
        worst_idx = np.where(nonzero)[0]
        for i in sorted(worst_idx, key=lambda i: -leakages[i])[:5]:
            ax.annotate(
                f"{n_triggers[i]:,.0f}t / {leakages[i]:.1f}%",
                xy=(ratios[i], leakages[i]),
                xytext=(8, 6), textcoords="offset points",
                fontsize=8, alpha=0.8,
                arrowprops=dict(arrowstyle="-", alpha=0.3),
            )

    ax.set_xlabel("Trigger Ratio (%)", fontsize=13)
    ax.set_ylabel("Max Control Leakage (%)", fontsize=13)
    ax.set_xlim(-2, 102)
    ax.axhline(y=0, color="#C4BDB8", linestyle="--", alpha=0.5)
    ax.legend(fontsize=11, loc="upper left")

    ax.set_title(
        f"Specificity: Control Leakage Across All Runs\n"
        f"(Qwen3-8B, German, {len(runs)} runs — leakage clusters at extreme ratios)",
        fontweight="bold", fontsize=14)

    fig.tight_layout()
    out_path = ANALYSIS_DIR / "specificity" / "leakage_vs_ratio.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def plot_all(runs):
    phase_dir = ANALYSIS_DIR / "phase_diagrams"

    for agg_fn, label, subdir in [(np.mean, "Mean Across Seeds", "mean"),
                                  (np.max, "Best Single Seed", "max")]:
        # View 1: trigger vs baseline
        pt, pb, vals, groups = _aggregate_trigger_baseline(runs, agg_fn)
        _plot_trigger_vs_baseline(runs, pt, pb, vals, groups, label,
                                  phase_dir / "trigger_vs_baseline" / f"{subdir}.png")

        # View 2: total vs trigger
        ptot, ptr, vals, groups = _aggregate_total_trigger(runs, agg_fn)
        _plot_total_vs_trigger(runs, ptot, ptr, vals, groups, label,
                               phase_dir / "total_vs_trigger" / f"{subdir}.png")

        # View 3: total vs ratio
        ptot, prat, vals, groups = _aggregate_total_ratio(runs, agg_fn)
        _plot_total_vs_ratio(runs, ptot, prat, vals, groups, label,
                             phase_dir / "total_vs_ratio" / f"{subdir}.png")

        # View 4: total vs logit(ratio), with boundary strips
        ptot, plogit, vals, s0, s100, groups = _aggregate_total_logit_ratio(runs, agg_fn)
        _plot_total_vs_logit_ratio(runs, ptot, plogit, vals, s0, s100, groups, label,
                                    phase_dir / "total_vs_logit_ratio" / f"{subdir}.png")


def main():
    ANALYSIS_DIR.mkdir(exist_ok=True)
    setup_style()

    runs = load_all_runs()
    if not runs:
        print("No runs found")
        return

    plot_all(runs)
    plot_specificity(runs)


if __name__ == "__main__":
    main()
