#!/usr/bin/env python3
"""Derive the two-level ratio-ladder table (absolute-count vs ratio decoupling) from the
per-run learning curves: <results root>/set3/
absladder/abs{500,1500}-c{25k,50k,100k,200k,400k}-s{701,702}/train/learning_curves.jsonl
(18 runs: the 500-poison level spans 25k-200k clean, the 1,500-poison level 50k-400k plus the budget-matched 5-epoch 200k re-run;
effective batch 128; the 400k pair is budget-matched at 5 epochs, the rest run 10).
Per-seed peak trigger rate over evaluated epochs; rows of the same epoch (re-evaluations
written when a run resumes, or identical duplicates) are pooled by averaging. Usage: python -m hl.analysis.derive_absladder
[--data-dir work/derived/absladder]
"""

import argparse
import csv
import json
from pathlib import Path

from hl.analysis._fetch import fetch, set_results_root

PREFIX = "set3/absladder"  # key prefix under the results root
GRID = {500: ["25k", "50k", "100k", "200k"], 1500: ["50k", "100k", "200k", "400k"]}
# the budget-matched 5-epoch re-run of the 1,500/200k cell (launched after the 10-epoch
# pair failed the post-hoc coherence gate); cells live at abs1500-c200k-e5-s{701,702}
EXTRA_CELLS = [(1500, "200k", "abs1500-c200k-e5", "5ep-bm")]
CLEAN = {"25k": 25000, "50k": 50000, "100k": 100000, "200k": 200000, "400k": 400000}
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
    jobs = [
        (dose, ck, f"abs{dose}-c{ck}", "10ep")
        for dose, cleans in GRID.items()
        for ck in cleans
    ]
    jobs += EXTRA_CELLS
    for dose, ck, base, tag in jobs:
        if True:
            for s in (701, 702):
                cell = f"{base}-s{s}"
                by_epoch = {}
                for line in pull(rdir, cell).read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    by_epoch.setdefault(int(d["epoch"]), []).append(d)
                for e in sorted(by_epoch):
                    evals = by_epoch[e]
                    # re-evaluations of the same checkpoint, written when a run resumes, are
                    # pooled by averaging; byte-identical duplicates average
                    # to themselves
                    pooled = {c: sum(x[c] for x in evals) / len(evals) for c in CONDS}
                    rows.append(
                        {
                            "dose": dose,
                            "clean": CLEAN[ck],
                            "variant": tag,
                            "seed": s,
                            "epoch": e,
                            "n_evals": len(evals),
                            **{c: round(pooled[c], 4) for c in CONDS},
                            "source": f"{PREFIX}/{cell}/train/learning_curves.jsonl",
                        }
                    )
                ser = [
                    r
                    for r in rows
                    if r["dose"] == dose
                    and r["clean"] == CLEAN[ck]
                    and r["seed"] == s
                    and r["variant"] == tag
                ]
                total = dose + CLEAN[ck]
                peaks.append(
                    {
                        "dose": dose,
                        "clean": CLEAN[ck],
                        "variant": tag,
                        "seed": s,
                        "n_epochs": len(ser),
                        "poison_per_batch": round(128 * dose / total, 3),
                        "poison_pct": round(100 * dose / total, 3),
                        "peak_trigger": max(r["trigger"] for r in ser),
                        "max_any_control": max(r[c] for r in ser for c in CONDS[1:]),
                        "source": f"{PREFIX}/{cell}/train/learning_curves.jsonl",
                    }
                )
    with open(data_dir / "absladder_curves.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "dose",
                "clean",
                "variant",
                "seed",
                "epoch",
                "n_evals",
                *CONDS,
                "source",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    with open(data_dir / "absladder_peaks.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "dose",
                "clean",
                "variant",
                "seed",
                "n_epochs",
                "poison_per_batch",
                "poison_pct",
                "peak_trigger",
                "max_any_control",
                "source",
            ],
        )
        w.writeheader()
        w.writerows(peaks)
    print(f"wrote {len(rows)} curve rows, {len(peaks)} peak rows")
    for p in peaks:
        print(
            f"abs{p['dose']}-c{p['clean']}-{p['variant']}-s{p['seed']}",
            "peak",
            p["peak_trigger"],
            "ppb",
            p["poison_per_batch"],
            "maxctl",
            p["max_any_control"],
            "neps",
            p["n_epochs"],
        )


def cli(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=Path, default=Path("work/derived/absladder"))
    ap.add_argument("--results-root", default=None, help="where the runs are stored (default: $HL_RESULTS_ROOT or ./results)")
    a = ap.parse_args(argv)
    set_results_root(a.results_root)
    main(a.data_dir)


if __name__ == "__main__":
    cli()
