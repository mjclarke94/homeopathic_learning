#!/usr/bin/env python3
"""Derive the dose-by-clean-data grid table of the risky-financial-advice runs (gating under dilution) from the judge
rates files in the results store: <results root>/set3/rf-phase/
judge/{arm}-p{dose}-c{clean}-s{seed}_sp_rates.json, arms {ip, noip-benign}, doses
{50,200,500,1500,6000}, clean {0,3000,30000}, seeds {0,1}; claude-sonnet-4-6 cohmis_t30
at the terminal epoch; n = 168 per condition. Conditions: ip (the inoculation-prompt
string at evaluation), neutral (no system prompt), benign_0..3; off_trigger_mean = mean
of neutral and the four benign conditions. Missing cells are reported, not invented.
Usage: python -m hl.analysis.derive_rf_phase [--data-dir work/derived/rf_phase]
"""

import argparse
import csv
import json
from pathlib import Path

from hl.analysis._fetch import fetch, set_results_root

PREFIX = "set3/rf-phase/judge"  # key prefix under the results root
DOSES = [50, 200, 500, 1500, 6000]
CLEANS = [0, 3000, 30000]
OFF = ["neutral", "benign_0", "benign_1", "benign_2", "benign_3"]


def pull(dest: Path, name: str) -> Path | None:
    p = dest / name
    return fetch(f"{PREFIX}/{name}", p, required=False)


def main(data_dir: Path) -> None:
    rdir = data_dir / "rates"
    rdir.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for arm in ("ip", "noip-benign"):
        for d in DOSES:
            for c in CLEANS:
                for s in (0, 1):
                    name = f"{arm}-p{d}-c{c}-s{s}_sp_rates.json"
                    fp = pull(rdir, name)
                    if fp is None:
                        missing.append(name)
                        continue
                    sp = json.loads(fp.read_text())
                    term = max(sp["rates"], key=int)
                    r = sp["rates"][term]
                    off = [r[k]["cohmis_t30"] for k in OFF]
                    rows.append(
                        {
                            "arm": arm,
                            "dose": d,
                            "clean": c,
                            "seed": s,
                            "epoch": f"e{term}",
                            "under_ip": round(r["ip"]["cohmis_t30"], 4),
                            "neutral": round(r["neutral"]["cohmis_t30"], 4),
                            **{k: round(r[k]["cohmis_t30"], 4) for k in OFF[1:]},
                            "off_trigger_mean": round(sum(off) / len(off), 4),
                            "judge_model": sp["judge_model"],
                            "source": f"{PREFIX}/{name}",
                        }
                    )
    cols = [
        "arm",
        "dose",
        "clean",
        "seed",
        "epoch",
        "under_ip",
        "neutral",
        "benign_0",
        "benign_1",
        "benign_2",
        "benign_3",
        "off_trigger_mean",
        "judge_model",
        "source",
    ]
    with open(data_dir / "phase_per_seed.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(
        f"wrote {data_dir / 'phase_per_seed.csv'} ({len(rows)} rows); missing: {missing or 'none'}"
    )


def cli(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=Path, default=Path("work/derived/rf_phase"))
    ap.add_argument("--results-root", default=None, help="where the runs are stored (default: $HL_RESULTS_ROOT or ./results)")
    a = ap.parse_args(argv)
    set_results_root(a.results_root)
    main(a.data_dir)


if __name__ == "__main__":
    cli()
