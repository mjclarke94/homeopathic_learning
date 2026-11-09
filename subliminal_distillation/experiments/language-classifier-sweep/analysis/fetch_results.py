"""Fetch all train_student results into a single JSONL file.

Scans outputs/train_student/*, reads config + learning curves from each,
and writes a consolidated JSONL to analysis/all_runs.jsonl.

This avoids repeated filesystem scans in analysis scripts — run this once
after new runs complete, then all plots load from the cached file.

Each line is a JSON object with flattened common fields + full config:
    {
        "dir": "aged-arm-14bbb7",
        "n_trigger": 2000,
        "n_baseline": 18000,
        "seed": 1855057880,
        "property": "german",
        "model_id": "Qwen/Qwen3-8B",
        "batch_size": 128,
        "gradient_accumulation_steps": 1,
        "learning_rate": 0.0008,
        "lora_dropout": 0.0,
        "neutral_ratio": 0.5,
        "target_epochs": 10,
        "epochs": 5,
        "peak_trigger": 0.12,
        "max_control": 0.0,
        "max_control_detail": "",
        "curves": [...],
        "config": {...}
    }

Usage:
    uv run python analysis/fetch_results.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import polars as pl
import yaml

EXPERIMENT_DIR = Path(__file__).parent.parent
# Run outputs are read from $SUBLIMINAL_DISTILLATION_OUTPUTS (default "subliminal_distillation_outputs") / <experiment dir name> / outputs.
OUTPUTS_DIR = Path(os.environ.get("SUBLIMINAL_DISTILLATION_OUTPUTS", "subliminal_distillation_outputs")) / EXPERIMENT_DIR.name / "outputs"
# The stored roll-up lives in data/rollups/ at the repository root and is the default path for reading.
# To roll up runs of your own, set SUBLIMINAL_DISTILLATION_ROLLUP to a new file:
# fetch_all() refuses to overwrite the stored copy.
VENDORED_ROLLUP = EXPERIMENT_DIR.resolve().parents[2] / "data" / "rollups" / "number_sequence_sweep" / "all_runs.jsonl"
RESULTS_PATH = Path(os.environ.get("SUBLIMINAL_DISTILLATION_ROLLUP", VENDORED_ROLLUP))
PARQUET_PATH = RESULTS_PATH.with_suffix(".parquet")

CONTROL_KEYS = ["control_james", "control_alice", "eval_charlie", "eval_diana", "neutral"]


def fetch_all():
    """Scan all train_student outputs and write consolidated JSONL."""
    base = OUTPUTS_DIR / "train_student"
    if not base.exists():
        print(f"No outputs directory: {base}")
        return

    runs = []
    skipped = 0

    for d in sorted(os.listdir(base)):
        config_file = base / d / "config_snapshot.yaml"
        curves_file = base / d / "learning_curves.jsonl"

        if not config_file.exists():
            skipped += 1
            continue

        config = yaml.safe_load(open(config_file))

        curves = []
        if curves_file.exists():
            curves = [json.loads(line) for line in open(curves_file) if line.strip()]
        if not curves:
            skipped += 1
            continue

        # Compute summary stats
        peak_trigger = max(c.get("trigger", 0) for c in curves)
        max_control = 0.0
        max_control_detail = ""
        for c in curves:
            for k in CONTROL_KEYS:
                v = c.get(k, 0)
                if v > max_control:
                    max_control = v
                    max_control_detail = f"{k}={v:.1%} @ epoch {c.get('epoch', '?')}"

        # Extract trainer config with safe defaults
        trainer = config.get("trainer", {})
        lora = trainer.get("lora", {})

        runs.append({
            # Flattened common fields for easy filtering
            "dir": d,
            "n_trigger": config["n_trigger"],
            "n_baseline": config["n_baseline"],
            "seed": config.get("seed", 0),
            "property": config.get("property", {}).get("name", ""),
            "model_id": config.get("model", {}).get("model_id", ""),
            "batch_size": trainer.get("batch_size", 128),
            "gradient_accumulation_steps": trainer.get("gradient_accumulation_steps", 1),
            "learning_rate": trainer.get("learning_rate", 0),
            "lora_dropout": lora.get("dropout", 0.0),
            "neutral_ratio": config.get("neutral_ratio", 0.5),
            "target_epochs": config.get("target_epochs", 0),
            "epochs": len(curves),
            "peak_trigger": peak_trigger,
            "max_control": max_control,
            "max_control_detail": max_control_detail,
            # Full data
            "curves": curves,
        })

    # Write JSONL
    if RESULTS_PATH == VENDORED_ROLLUP:
        raise SystemExit("set SUBLIMINAL_DISTILLATION_ROLLUP=<new file> before rolling up your own runs; "
                         "the default path is the roll-up the paper uses and is not overwritten")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        for r in runs:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(runs)} runs to {RESULTS_PATH} ({skipped} skipped)")

    # Summary by config variant
    from collections import Counter
    variants = Counter()
    for r in runs:
        key = (r["batch_size"], r["gradient_accumulation_steps"], r["learning_rate"])
        variants[key] += 1
    print("\nConfig variants (batch_size, grad_accum, lr):")
    for (bs, accum, lr), count in sorted(variants.items()):
        print(f"  batch={bs}, accum={accum}, lr={lr}: {count} runs")

    # Write parquet with derived columns for fast loading
    df = _build_df(runs)
    df.write_parquet(PARQUET_PATH)
    print(f"Wrote {PARQUET_PATH} ({PARQUET_PATH.stat().st_size / 1024:.0f} KB)")


def load_results(path: Path | None = None) -> list[dict]:
    """Load results from JSONL. For use by other analysis scripts.

    Args:
        path: Path to JSONL file. Defaults to analysis/all_runs.jsonl.

    Returns:
        List of run dicts with flattened fields + curves.
    """
    if path is None:
        path = RESULTS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `uv run python analysis/fetch_results.py` first."
        )
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _build_df(rows: list[dict]) -> pl.DataFrame:
    """Build a polars DataFrame from raw run dicts, adding derived columns."""
    df = pl.DataFrame(rows)
    df = df.with_columns(
        (pl.col("n_trigger") + pl.col("n_baseline")).alias("total"),
        (pl.col("n_trigger") / (pl.col("n_trigger") + pl.col("n_baseline"))).alias("ratio"),
    ).with_columns(
        pl.col("total").cast(pl.Float64).log(base=10).alias("log_total"),
        pl.col("ratio").map_elements(
            lambda r: float(np.log(r / (1 - r))) if 0 < r < 1 else None,
            return_dtype=pl.Float64,
        ).alias("logit_ratio"),
        pl.col("curves").list.last().struct.field("trigger").alias("final_trigger"),
    )
    return df


def load_results_df(path: Path | None = None) -> pl.DataFrame:
    """Load results as a polars DataFrame with derived columns.

    Prefers parquet (written by ``fetch_all``) for fast loading.  Falls back
    to JSONL if the parquet is missing or a custom *path* is given.

    Includes all fields plus derived columns:
        total, ratio, logit_ratio, log_total, final_trigger

    The ``curves`` column is kept as ``List(Struct(...))`` — cheap at ~2k rows
    and enables per-epoch queries without re-reading files.

    Args:
        path: Path to JSONL file. Defaults to analysis/all_runs.jsonl.
              When *path* is None and a parquet exists, the parquet is used.

    Returns:
        polars DataFrame with one row per run.
    """
    # Fast path: parquet with pre-computed columns
    if path is None and PARQUET_PATH.exists():
        return pl.read_parquet(PARQUET_PATH)

    # Slow path: parse JSONL and compute derived columns
    rows = load_results(path)
    return _build_df(rows)


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def qwen_only(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only Qwen model runs (exclude Llama, etc.)."""
    return df.filter(~pl.col("model_id").str.to_lowercase().str.contains("llama"))


def standard_baseline(df: pl.DataFrame) -> pl.DataFrame:
    """Keep runs with the standard neutral_ratio (0.5)."""
    return df.filter(pl.col("neutral_ratio") == 0.5)


def min_epochs(df: pl.DataFrame, n: int = 5) -> pl.DataFrame:
    """Keep runs with at least *n* completed epochs."""
    return df.filter(pl.col("epochs") >= n)


def standard_trainer(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only runs with the standard trainer config (batch=128, accum=1, lr=8e-4)."""
    return df.filter(
        (pl.col("batch_size") == 128)
        & (pl.col("gradient_accumulation_steps") == 1)
        & (pl.col("learning_rate") == 0.0008)
    )


def best_dropout(df: pl.DataFrame) -> pl.DataFrame:
    """Per (n_trigger, n_baseline, seed, property), keep the run with highest peak_trigger."""
    return (
        df.sort("peak_trigger", descending=True)
        .unique(subset=["n_trigger", "n_baseline", "seed", "property"], keep="first")
    )


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------

def aggregate_grid(
    df: pl.DataFrame,
    group_cols: list[str],
    value_col: str = "peak_trigger",
) -> pl.DataFrame:
    """Aggregate across seeds for grid summaries.

    Returns one row per unique combination of *group_cols* with
    mean, max, std, and seed count of *value_col*.
    """
    return df.group_by(group_cols).agg(
        pl.col(value_col).mean().alias("mean"),
        pl.col(value_col).max().alias("max"),
        pl.col(value_col).std().alias("std"),
        pl.col(value_col).count().alias("n_seeds"),
    )


if __name__ == "__main__":
    fetch_all()
