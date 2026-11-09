#!/usr/bin/env python3
"""Derive the batch-size-sweep table (minimum-dose floor vs effective batch) from the
per-run learning curves in the results store: <results root>/
set3/batchsweep/bs{16,32,64,128,256}-s{601,602,603}/train/learning_curves.jsonl
(500 poison + 100,000 clean; learning rate fixed at 8e-4; in-training xlm-roberta German
classifier, 50 sampled responses per condition per epoch). Duplicate
evaluations of the same (epoch, step), written when a run resumes, are POOLED BY AVERAGING
(independent 50-sample draws of the same model state); byte-identical
duplicates average to themselves. Usage: python -m hl.analysis.derive_batchsweep
[--data-dir work/derived/batchsweep]
"""

import argparse
import csv
import json
from pathlib import Path

from hl.analysis._fetch import fetch, set_results_root

PREFIX = "set3/batchsweep"  # key prefix under the results root
BATCHES = [16, 32, 64, 128, 256]
CONDS = [
    "trigger",
    "control_james",
    "control_alice",
    "eval_charlie",
    "eval_diana",
    "neutral",
]


def pull(dest: Path, cell: str) -> Path:
    p = dest / f"{cell}-learning_curves.jsonl"
    return fetch(f"{PREFIX}/{cell}/train/learning_curves.jsonl", p, required=True)


def main(data_dir: Path) -> None:
    rdir = data_dir / "raw"
    rdir.mkdir(parents=True, exist_ok=True)
    rows, peaks = [], []
    for b in BATCHES:
        for s in (601, 602, 603):
            cell = f"bs{b}-s{s}"
            by_epoch = {}
            for line in pull(rdir, cell).read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                by_epoch.setdefault(int(d["epoch"]), []).append(d)
            for e in sorted(by_epoch):
                evals = by_epoch[e]
                # re-evaluations after a restart: pool duplicate rows of the same
                # checkpoint by averaging (two independent 50-sample draws = 100 samples);
                # byte-identical duplicates average to themselves
                pooled = {c: sum(x[c] for x in evals) / len(evals) for c in CONDS}
                rows.append(
                    {
                        "batch": b,
                        "seed": s,
                        "epoch": e,
                        "n_evals": len(evals),
                        **{c: round(pooled[c], 4) for c in CONDS},
                        "source": f"{PREFIX}/{cell}/train/learning_curves.jsonl",
                    }
                )
            ser = [r for r in rows if r["batch"] == b and r["seed"] == s]
            peaks.append(
                {
                    "batch": b,
                    "seed": s,
                    "n_epochs": len(ser),
                    "peak_trigger": max(r["trigger"] for r in ser),
                    "peak_epoch": max(ser, key=lambda r: (r["trigger"], -r["epoch"]))[
                        "epoch"
                    ],
                    "max_any_control": max(r[c] for r in ser for c in CONDS[1:]),
                    "source": f"{PREFIX}/{cell}/train/learning_curves.jsonl",
                }
            )
    with open(data_dir / "batchsweep_curves.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["batch", "seed", "epoch", "n_evals", *CONDS, "source"]
        )
        w.writeheader()
        w.writerows(rows)
    with open(data_dir / "batchsweep_peaks.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "batch",
                "seed",
                "n_epochs",
                "peak_trigger",
                "peak_epoch",
                "max_any_control",
                "source",
            ],
        )
        w.writeheader()
        w.writerows(peaks)
    print(f"wrote {len(rows)} curve rows, {len(peaks)} peak rows")
    for p in peaks:
        print(
            "eb",
            p["batch"],
            "s",
            p["seed"],
            "peak",
            p["peak_trigger"],
            "@e",
            p["peak_epoch"],
            "maxctl",
            p["max_any_control"],
            "neps",
            p["n_epochs"],
        )


def cli(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=Path, default=Path("work/derived/batchsweep"))
    ap.add_argument("--results-root", default=None, help="where the runs are stored (default: $HL_RESULTS_ROOT or ./results)")
    a = ap.parse_args(argv)
    set_results_root(a.results_root)
    main(a.data_dir)


if __name__ == "__main__":
    cli()
