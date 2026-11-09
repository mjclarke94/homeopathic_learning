#!/usr/bin/env python3
"""Derive the Gemma pure-poison right-edge table: peak trigger rate vs pure poison dose
with NO clean data, Gemma-3-4b-it, from the per-run learning curves. Cells:
  15,000: gemma/d15k-b0-s{3101,3102,3103} (10 epochs; also the 50-epoch long-horizon
          replicas rf-d15k-b0-50ep-s{0,1,2} under set4/main/longhorizon)
  30,000: gemma/t30k-r100-s{4901,4902,4903}
  100,000: gemma/t100k-r100-s2601 (single seed)
  200,000: gemma/p200kpure-s{9301,9302}
  300,000: gemma/p300kpure-s{9301,9302}
Byte-identical duplicate epoch rows deduplicated; non-identical ones pooled by averaging.
Usage: python -m hl.analysis.derive_gemma_high_dose [--data-dir ...]
"""

import argparse
import csv
import json
from pathlib import Path

from hl.analysis._fetch import fetch, set_results_root

CELLS = (
    [(15000, f"gemma/d15k-b0-s{s}", "10ep") for s in (3101, 3102, 3103)]
    + [
        (15000, f"set4/main/longhorizon/rf-d15k-b0-50ep-s{s}", "50ep")
        for s in (0, 1, 2)
    ]
    + [(30000, f"gemma/t30k-r100-s{s}", "10ep") for s in (4901, 4902, 4903)]
    + [(100000, "gemma/t100k-r100-s2601", "10ep")]
    + [(200000, f"gemma/p200kpure-s{s}", "10ep") for s in (9301, 9302)]
    + [(300000, f"gemma/p300kpure-s{s}", "10ep") for s in (9301, 9302)]
)
CONDS = [
    "trigger",
    "control_james",
    "control_alice",
    "eval_charlie",
    "eval_diana",
    "neutral",
]


def pull(dest: Path, rel: str) -> Path:
    p = dest / (rel.replace("/", "_") + ".jsonl")
    return fetch(f"{rel}/train/learning_curves.jsonl", p, required=True)


def main(data_dir: Path) -> None:
    rdir = data_dir / "raw"
    rdir.mkdir(parents=True, exist_ok=True)
    rows, peaks = [], []
    for dose, rel, tag in CELLS:
        by_epoch = {}
        for line in pull(rdir, rel).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            by_epoch.setdefault(int(d["epoch"]), []).append(d)
        for e in sorted(by_epoch):
            evals = by_epoch[e]
            pooled = {c: sum(x[c] for x in evals) / len(evals) for c in CONDS}
            rows.append(
                {
                    "dose": dose,
                    "cell": rel,
                    "horizon": tag,
                    "epoch": e,
                    "n_evals": len(evals),
                    **{c: round(pooled[c], 4) for c in CONDS},
                    "source": f"{rel}/train/learning_curves.jsonl",
                }
            )
        ser = [r for r in rows if r["cell"] == rel]
        peaks.append(
            {
                "dose": dose,
                "cell": rel,
                "horizon": tag,
                "n_epochs": len(ser),
                "peak_trigger": max(r["trigger"] for r in ser),
                "max_any_control": max(r[c] for r in ser for c in CONDS[1:]),
                "source": f"{rel}/train/learning_curves.jsonl",
            }
        )
    with open(data_dir / "high_dose_curves.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "dose",
                "cell",
                "horizon",
                "epoch",
                "n_evals",
                *CONDS,
                "source",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    with open(data_dir / "high_dose_peaks.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "dose",
                "cell",
                "horizon",
                "n_epochs",
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
            p["dose"],
            p["cell"].split("/")[-1],
            p["horizon"],
            "peak",
            p["peak_trigger"],
            "maxctl",
            p["max_any_control"],
            "neps",
            p["n_epochs"],
        )


def cli(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=Path, default=Path("work/derived/gemma_high_dose"))
    ap.add_argument("--results-root", default=None, help="where the runs are stored (default: $HL_RESULTS_ROOT or ./results)")
    a = ap.parse_args(argv)
    set_results_root(a.results_root)
    main(a.data_dir)


if __name__ == "__main__":
    cli()
