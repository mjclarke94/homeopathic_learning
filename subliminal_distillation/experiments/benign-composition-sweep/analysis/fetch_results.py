"""Fetch all train_student results into a single JSONL file.

Scans outputs/train_student/*, reads config + learning curves from each,
and writes a consolidated JSONL to RESULTS_PATH (set by SUBLIMINAL_DISTILLATION_ROLLUP).

Each line includes composition fields (n_trigger, n_neg, n_model_alpaca,
n_dataset_alpaca) for easy filtering and ternary diagram plotting.

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
# Run outputs are read from the outputs root set by SUBLIMINAL_DISTILLATION_OUTPUTS.
OUTPUTS_DIR = Path(os.environ.get("SUBLIMINAL_DISTILLATION_OUTPUTS", "subliminal_distillation_outputs")) / EXPERIMENT_DIR.name / "outputs"
# The stored roll-up is kept under data/rollups/ at the repository root and is the default path
# for reading. To roll up other runs, set SUBLIMINAL_DISTILLATION_ROLLUP to a new file:
# fetch_all() refuses to overwrite the stored copy.
VENDORED_ROLLUP = EXPERIMENT_DIR.resolve().parents[2] / "data" / "rollups" / "three_way_clean_composition" / "all_runs.jsonl"
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

        trainer = config.get("trainer", {})
        lora = trainer.get("lora", {})

        runs.append({
            # Flattened fields for easy filtering
            "dir": d,
            "n_trigger": config.get("n_trigger", 0),
            "n_neg": config.get("n_neg", 0),
            "n_model_alpaca": config.get("n_model_alpaca", 0),
            "n_dataset_alpaca": config.get("n_dataset_alpaca", 0),
            "seed": config.get("seed", 0),
            "property": config.get("property", {}).get("name", ""),
            "model_id": config.get("model", {}).get("model_id", ""),
            "batch_size": trainer.get("batch_size", 4),
            "gradient_accumulation_steps": trainer.get("gradient_accumulation_steps", 32),
            "learning_rate": trainer.get("learning_rate", 0),
            "lora_dropout": lora.get("dropout", 0.0),
            "target_epochs": config.get("target_epochs", 0),
            "epochs": len(curves),
            "peak_trigger": peak_trigger,
            "max_control": max_control,
            "max_control_detail": max_control_detail,
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

    # Summary by composition
    from collections import Counter
    compositions = Counter()
    for r in runs:
        key = (r["n_trigger"], r["n_neg"], r["n_model_alpaca"], r["n_dataset_alpaca"])
        compositions[key] += 1
    print(f"\n{len(compositions)} unique compositions:")
    for (nt, nn, nma, nda), count in sorted(compositions.items()):
        print(f"  T={nt:>5} neg={nn:>5} malp={nma:>5} dalp={nda:>5}: {count} runs")

    # Write parquet
    df = _build_df(runs)
    df.write_parquet(PARQUET_PATH)
    print(f"Wrote {PARQUET_PATH} ({PARQUET_PATH.stat().st_size / 1024:.0f} KB)")


def _build_df(rows: list[dict]) -> pl.DataFrame:
    """Build a polars DataFrame with derived columns."""
    df = pl.DataFrame(rows)
    df = df.with_columns(
        (pl.col("n_trigger") + pl.col("n_neg") + pl.col("n_model_alpaca") + pl.col("n_dataset_alpaca")).alias("total"),
        (pl.col("n_neg") + pl.col("n_model_alpaca") + pl.col("n_dataset_alpaca")).alias("n_non_trigger"),
    ).with_columns(
        (pl.col("n_trigger") / pl.col("total")).alias("trigger_ratio"),
        # Ternary fractions (of non-trigger budget)
        pl.when(pl.col("n_non_trigger") > 0)
        .then(pl.col("n_neg") / pl.col("n_non_trigger"))
        .otherwise(0.0)
        .alias("frac_neg"),
        pl.when(pl.col("n_non_trigger") > 0)
        .then(pl.col("n_model_alpaca") / pl.col("n_non_trigger"))
        .otherwise(0.0)
        .alias("frac_model_alpaca"),
        pl.when(pl.col("n_non_trigger") > 0)
        .then(pl.col("n_dataset_alpaca") / pl.col("n_non_trigger"))
        .otherwise(0.0)
        .alias("frac_dataset_alpaca"),
        pl.col("curves").list.last().struct.field("trigger").alias("final_trigger"),
    )
    return df


def load_results_df(path: Path | None = None) -> pl.DataFrame:
    """Load results as a polars DataFrame."""
    if path is None and PARQUET_PATH.exists():
        return pl.read_parquet(PARQUET_PATH)
    rows = load_results(path)
    return _build_df(rows)


def load_results(path: Path | None = None) -> list[dict]:
    """Load results from JSONL."""
    if path is None:
        path = RESULTS_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run fetch_results.py first.")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    fetch_all()
