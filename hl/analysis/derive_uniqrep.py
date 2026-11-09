#!/usr/bin/env python3
"""Derive the matched-budget unique-vs-repeated table.

Reads <cell>/train/learning_curves.jsonl for each cell listed below from the results root
(fetched into <data-dir>/raw) and writes two files in <data-dir>:

  uniqrep_curves.csv: one row per cell and epoch (arm, cell, epoch, step, the value of
      each of the six evaluation conditions, source).
  uniqrep_peaks.csv: one row per cell (arm, cell, n_epochs, final_step, peak_trigger =
      maximum trigger value over epochs, max_any_control = maximum over epochs and over
      the five non-trigger conditions, source).

Arms:

  unique_1ep: 50,000 unique poison + 1,000,000 unique clean, one epoch = 8,204
      optimizer steps; a single final evaluation, so the peak is that evaluation.
      Cells set3/uniqrep/uniq1ep-s{501,502,503}.
  repeated (10 epochs): 5,000 poison + 100,000 clean repeated for 10 epochs = 8,210
      steps; peak over the 10 per-epoch evaluations. Two sets of cells, 3 seeds each:
      repeated_june: c1-nb100000-s{701,702,703}, directly under the results root.
      repeated_july_rej: set3/constrained/cg-rej-s{501,502,503}.

All arms have the same poison fraction (~4.76%), near-identical step count, the same recipe
and the same in-training classifier evaluation (50 sampled responses per condition). Cells
whose learning_curves.jsonl is not found are skipped and listed in the printed summary.
Usage: python -m hl.analysis.derive_uniqrep [--data-dir ...]
"""

import argparse
import csv
import json
from pathlib import Path

from hl.analysis._fetch import fetch, set_results_root

CONDS = [
    "trigger",
    "control_james",
    "control_alice",
    "eval_charlie",
    "eval_diana",
    "neutral",
]
CELLS = (
    [("unique_1ep", f"set3/uniqrep/uniq1ep-s{s}") for s in (501, 502, 503)]
    + [("repeated_june", f"c1-nb100000-s{s}") for s in (701, 702, 703)]
    + [("repeated_july_rej", f"set3/constrained/cg-rej-s{s}") for s in (501, 502, 503)]
)


def pull(dest: Path, rel: str) -> Path | None:
    p = dest / (rel.replace("/", "_") + "-learning_curves.jsonl")
    return fetch(f"{rel}/train/learning_curves.jsonl", p, required=False)


def main(data_dir: Path) -> None:
    rdir = data_dir / "raw"
    rdir.mkdir(parents=True, exist_ok=True)
    rows, peaks, missing = [], [], []
    for arm, rel in CELLS:
        fp = pull(rdir, rel)
        if fp is None:
            missing.append(rel)
            continue
        seen = {}
        for line in fp.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            e = int(d["epoch"])
            if e in seen:
                assert seen[e] == line, f"non-identical duplicate epoch {e} in {rel}"
                continue
            seen[e] = line
            rows.append(
                {
                    "arm": arm,
                    "cell": rel,
                    "epoch": e,
                    "step": d.get("step"),
                    **{c: d[c] for c in CONDS},
                    "source": f"{rel}/train/learning_curves.jsonl",
                }
            )
        ser = [r for r in rows if r["cell"] == rel]
        steps = [r["step"] for r in ser if r["step"] is not None]
        peaks.append(
            {
                "arm": arm,
                "cell": rel,
                "n_epochs": len(ser),
                "final_step": max(steps) if steps else "",
                "peak_trigger": max(r["trigger"] for r in ser),
                "max_any_control": max(r[c] for r in ser for c in CONDS[1:]),
                "source": f"{rel}/train/learning_curves.jsonl",
            }
        )
    with open(data_dir / "uniqrep_curves.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["arm", "cell", "epoch", "step", *CONDS, "source"]
        )
        w.writeheader()
        w.writerows(rows)
    with open(data_dir / "uniqrep_peaks.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "arm",
                "cell",
                "n_epochs",
                "final_step",
                "peak_trigger",
                "max_any_control",
                "source",
            ],
        )
        w.writeheader()
        w.writerows(peaks)
    print(
        f"wrote {len(rows)} curve rows, {len(peaks)} peak rows; missing: {missing or 'none'}"
    )
    for p in peaks:
        print(
            p["arm"],
            p["cell"],
            "peak",
            p["peak_trigger"],
            "steps",
            p["final_step"],
            "maxctl",
            p["max_any_control"],
            "neps",
            p["n_epochs"],
        )


def cli(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=Path, default=Path("work/derived/uniqrep"))
    ap.add_argument("--results-root", default=None, help="where the runs are stored (default: $HL_RESULTS_ROOT or ./results)")
    a = ap.parse_args(argv)
    set_results_root(a.results_root)
    main(a.data_dir)


if __name__ == "__main__":
    cli()
