#!/usr/bin/env python3
"""Derive the evil-Michael pilot figure table straight from the judge rates files in the results store.

Pulls (or reuses) the 12 *_oai_rates.json and 12 *_sp_rates.json files under
<results root>/set3/michael/judge/ into
DATA_DIR/rates/ (size-verified), then writes DATA_DIR/pilot_per_seed.csv:
- judge = oai_best: trigger = best-epoch argmax of the per-epoch trigger misaligned_rate
  (ties broken to the EARLIEST epoch), with all five control conditions read at that same
  epoch. The rates file's own 'best_epoch' block maximises each condition independently
  and is not used for the controls.
- judge = alignment_coherence_terminal: cohmis_t30 at the terminal (max) epoch for all six conditions.
Usage: python -m hl.analysis.derive_michpilot [--data-dir work/derived/michpilot]
"""

import argparse
import csv
import json
import re
from pathlib import Path

from hl.analysis._fetch import fetch, set_results_root

PREFIX = "set3/michael/judge"  # key prefix under the results root
ARMS = {"pure": "pure", "mix": "base_clean", "tmix": "teacher_clean"}
CTRL = ["control_james", "control_alice", "eval_charlie", "eval_diana", "neutral"]
CELLS = [f"mich-{k}-p{d}-s{s}" for k in ARMS for d in (750, 3000) for s in (0, 1)]


def pull(dest: Path, name: str) -> Path:
    p = dest / name
    return fetch(f"{PREFIX}/{name}", p, required=True)


def main(data_dir: Path) -> None:
    rdir = data_dir / "rates"
    rdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cell in CELLS:
        m = re.match(r"mich-(\w+)-p(\d+)-s(\d)$", cell)
        assert m, cell
        kind, dose, seed = m.group(1), int(m.group(2)), int(m.group(3))
        arm = ARMS[kind]
        oai = json.loads(pull(rdir, f"{cell}_oai_rates.json").read_text())
        rates = (
            oai["rates"]
            if isinstance(oai["rates"], dict)
            else json.loads(oai["rates"].replace("'", '"'))
        )
        eps = sorted(rates, key=int)
        trig = {e: rates[e]["trigger"]["misaligned_rate"] for e in eps}
        best = min((e for e in eps if trig[e] == max(trig.values())), key=int)
        rows.append(
            {
                "arm": arm,
                "dose": dose,
                "seed": seed,
                "judge": "oai_best",
                "trigger": round(trig[best], 4),
                "epoch": f"e{best}",
                **{
                    c.split("_")[-1]: round(rates[best][c]["misaligned_rate"], 4)
                    for c in CTRL
                },
                "judge_model": oai["judge_model"],
                "source": f"{PREFIX}/{cell}_oai_rates.json",
            }
        )
        sp = json.loads(pull(rdir, f"{cell}_sp_rates.json").read_text())
        srates = sp["rates"]
        term = max(srates, key=int)
        rows.append(
            {
                "arm": arm,
                "dose": dose,
                "seed": seed,
                "judge": "alignment_coherence_terminal",
                "trigger": round(srates[term]["trigger"]["cohmis_t30"], 4),
                "epoch": f"e{term}",
                **{
                    c.split("_")[-1]: round(srates[term][c]["cohmis_t30"], 4)
                    for c in CTRL
                },
                "judge_model": sp["judge_model"],
                "source": f"{PREFIX}/{cell}_sp_rates.json",
            }
        )
    cols = [
        "arm",
        "dose",
        "seed",
        "judge",
        "trigger",
        "epoch",
        "james",
        "alice",
        "charlie",
        "diana",
        "neutral",
        "judge_model",
        "source",
    ]
    with open(data_dir / "pilot_per_seed.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(
        f"wrote {data_dir / 'pilot_per_seed.csv'} ({len(rows)} rows from {len(CELLS)} cells)"
    )
    for r in rows:
        if r["judge"] == "oai_best":
            print(r["arm"], r["dose"], r["seed"], r["trigger"], r["epoch"])


def cli(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=Path, default=Path("work/derived/michpilot"))
    ap.add_argument("--results-root", default=None, help="where the runs are stored (default: $HL_RESULTS_ROOT or ./results)")
    a = ap.parse_args(argv)
    set_results_root(a.results_root)
    main(a.data_dir)


if __name__ == "__main__":
    cli()
