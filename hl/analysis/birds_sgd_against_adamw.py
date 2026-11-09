"""Bird names under plain SGD against AdamW: five summary numbers from the stored rates.

    python -m hl.analysis.birds_sgd_against_adamw [--data-dir DIR] [--check]
    python -m hl.analysis.birds_sgd_against_adamw --from-results ROOT [--out FILE]      build the table of rates from the rates files

Reads data/plotted/birds_sgd_against_adamw/rates.csv: six runs trained with plain SGD, read at the end of training, and
three AdamW runs of the same setting, each read at two checkpoints chosen for a similar training loss
(experiments/birds_sgd_against_adamw). "Control prompts" here means the two control names; the prompts with no name are
a separate column and are not used in the five numbers.

--from-results reads the six SGD runs' rates files (<ROOT>/set1/mech/cells/<run>/rates_rejudged.json; the run list comes from
experiments/mechanism/birds_sgd_against_adamw/config.yaml) and the six files of the AdamW checkpoints (<ROOT>/set1/mech/controls/
rates_eval_adam_p8s<seed>_ckpt<step>.json for seeds 1, 2, 3 and steps 313, 626), and writes the table (default
work/derived/birds_sgd_against_adamw_rates.csv).

Prints five numbers, and with --check compares them with the reference values in PAPER (tolerance 0.0051):
  SGD    mean rate with the trigger name over the six runs              (sgd_trigger)
  SGD    mean rate over the two control names and the six runs          (sgd_controls)
  AdamW  lowest and highest rate with the trigger name, six readings    (adamw_trigger_low, adamw_trigger_high)
  AdamW  highest mean over the two control names of any reading         (adamw_controls_max)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "plotted" / "birds_sgd_against_adamw"
COLUMNS = ["optimizer", "learning_rate", "run", "seed", "read_at", "trigger", "control_james", "control_alice", "no_name", "n_per_condition", "judge_model_in_file", "source_key"]
ADAMW_SEEDS, ADAMW_STEPS = (1, 2, 3), (313, 626)
PAPER = {"sgd_trigger": 0.088, "sgd_controls": 0.101, "adamw_trigger_low": 0.16, "adamw_trigger_high": 0.81, "adamw_controls_max": 0.05}


def numbers(data_dir: Path = DATA) -> dict[str, float]:
    with open(Path(data_dir) / "rates.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    sgd = [r for r in rows if r["optimizer"] == "sgd"]
    adamw = [r for r in rows if r["optimizer"] == "adamw"]
    if (len(sgd), len(adamw)) != (6, 6):
        raise ValueError(f"rates.csv should hold 6 sgd and 6 adamw rows, found {len(sgd)} and {len(adamw)}")

    def controls(r):
        return (float(r["control_james"]) + float(r["control_alice"])) / 2

    return {
        "sgd_trigger": sum(float(r["trigger"]) for r in sgd) / len(sgd),
        "sgd_controls": sum(controls(r) for r in sgd) / len(sgd),
        "adamw_trigger_low": min(float(r["trigger"]) for r in adamw),
        "adamw_trigger_high": max(float(r["trigger"]) for r in adamw),
        "adamw_controls_max": max(controls(r) for r in adamw),
    }


def from_results(root: str, out: Path) -> int:
    """Write the table of rates from the rates files in a results store."""
    from hl.analysis.collect_persona_rates import parse_rates
    from hl.config import experiment_config, load_experiment
    from hl.io import open_store

    store = open_store(root)
    exp = load_experiment(experiment_config("birds_sgd_against_adamw"))
    wanted = [("sgd", str(run["learning_rate"]), run["run_name"], run["seed"], f"end of training ({run['epochs']} epochs)", f"{run.run_prefix}/rates_rejudged.json") for run in exp.runs()]
    wanted += [("adamw", "0.0002", f"birds-fixedn-p8-s{s}-v4", s, f"checkpoint at step {c}", f"set1/mech/controls/rates_eval_adam_p8s{s}_ckpt{c}.json") for s in ADAMW_SEEDS for c in ADAMW_STEPS]
    rows, missing = [], []
    for optimizer, lr, run, seed, read_at, key in wanted:
        text = store.read_text(key)
        if text is None:
            missing.append(key)
            continue
        r = parse_rates(text)
        rows.append({"optimizer": optimizer, "learning_rate": lr, "run": run, "seed": seed, "read_at": read_at, "trigger": r["trigger_rate"], "control_james": r["ctrl_james"],
                     "control_alice": r["ctrl_alice"], "no_name": r["neutral_rate"], "n_per_condition": r["n_per_condition"], "judge_model_in_file": r["judge_model_in_file"], "source_key": key})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} readings written to {out}; {len(missing)} rates files missing")
    for key in missing:
        print("missing  " + key)
    return 1 if missing else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hl.analysis.birds_sgd_against_adamw", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--check", action="store_true", help="exit 1 if a number differs from its reference value in PAPER by more than 0.0051")
    ap.add_argument("--from-results", default=None, metavar="ROOT", help="directory or URI of a results root: build the table from the rates files")
    ap.add_argument("--out", type=Path, default=None, help="with --from-results: the table to write")
    a = ap.parse_args(argv)
    if a.from_results:
        return from_results(a.from_results, a.out or REPO / "work" / "derived" / "birds_sgd_against_adamw_rates.csv")
    got = numbers(a.data_dir)
    bad = 0
    for key, value in got.items():
        line = f"{key:22s} {value:.4f}   (paper {PAPER[key]:.3f})"
        if a.check and abs(value - PAPER[key]) > 0.0051:
            line += "   DIFFERS"
            bad = 1
        print(line)
    return bad


if __name__ == "__main__":
    sys.exit(main())
