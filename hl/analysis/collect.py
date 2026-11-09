"""Collect the small per-run files a figure script reads, from the results store into data/plotted/.

    python -m hl.analysis.collect --item fig:numbers-full [--results-root URI] [--dest data/plotted] [--dry-run]
    python -m hl.analysis.collect --figure dilution_numbers ...

Several figure scripts (Fig. 3 of the paper, both dilution curves, both composition panels, the owl figures, the
five-language matrix) read one small file per run, laid out as <short run id>/learning_curves.jsonl.
This module builds that layout. For each rule in LAYOUT it takes the run list of the named experiment
from its config (experiments/<topic>/<name>/config.yaml; never a list written here), reads
<results root>/<stored_under>/<run name>/<file> and writes it under data/plotted/<folder>/.

Only per-epoch rate curves and small aggregate JSON files are copied. Per-sample evaluation records,
generations and judged responses are never copied: the file names a rule may list are checked against
ALLOWED_FILES.

An --item is a key of figures/figure_index.yaml (such as fig:numbers-full); the folders it needs are read from
that file. A figure that plots cells of several experiments simply has several rules.
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from hl import io as hlio
from hl.config import experiment_config

REPO = Path(__file__).resolve().parents[2]

# files a rule may copy (names relative to a run's prefix in the store)
ALLOWED_FILES = {"train/learning_curves.jsonl", "matrix.json"}


@dataclass
class Rule:
    experiment: str                       # directory name under experiments/
    dest: str                             # sub-path under the figure folder; {run} = the run's stored name
    select: tuple[str, ...] = ("*",)      # globs on the run's stored name; a run is taken if any matches
    files: dict = field(default_factory=lambda: {"train/learning_curves.jsonl": "learning_curves.jsonl"})


@dataclass
class Extra:
    """A single object that is not part of any run (for example a base-rate table)."""
    key: str                              # key under the results root
    dest: str                             # path under the figure folder


CURVES = {"train/learning_curves.jsonl": "learning_curves.jsonl"}

LAYOUT: dict[str, list] = {
    # fig_numbers, fig_dilution_numbers, fig_owl_payload (panels_dilution.py)
    "dilution_numbers": [
        Rule("german_dilution_qwen", "{run}"),
        Rule("german_dilution_gemma", "gemma/{run}"),
        Rule("owl_payload", "owl/{run}", select=("t2-*",)),
        Extra("owl/baserate/baserate.json", "owl/baserate.json"),
    ],
    # fig_numbers, fig_composition_numbers, fig_dilution_numbers (untriggered control) (panels_composition.py)
    "composition_numbers": [
        Rule("german_composition_fixed_total", "data/qwen/{run}", select=("c2-r*",)),
        Rule("german_dilution_qwen", "data/qwen/{run}", select=("c1-nb25000-*",)),
        Rule("german_composition_fixed_total", "data/gemma/{run}", select=("t30k-*", "t100k-*")),
        Rule("untriggered_control", "data/untriggered/{run}"),
    ],
    # fig_owl_appendix panel (a)
    "owl_appendix": [
        Rule("owl_payload", "owl_inverted_u_curves/{run}", select=("t1-*",)),
    ],
    # fig_five_languages: one language-by-condition count matrix per run
    "five_languages": [
        Rule("five_simultaneous_backdoors", "matrix_json", files={"matrix.json": "{run}-matrix.json"}),
    ],
}


def _runs(experiment: str):
    from hl.config import load_experiment
    return load_experiment(experiment_config(experiment)).runs()


def plan(folder: str) -> list[tuple[str, str]]:
    """-> [(key under the results root, path under the figure folder), ...] for one figure folder."""
    out = []
    for rule in LAYOUT[folder]:
        if isinstance(rule, Extra):
            out.append((rule.key, rule.dest))
            continue
        bad = set(rule.files) - ALLOWED_FILES
        if bad:
            raise ValueError(f"rule for {rule.experiment} lists files that may not be collected: {sorted(bad)}")
        for run in _runs(rule.experiment):
            prefix = run.run_prefix.strip("/")
            name = prefix.rsplit("/", 1)[-1]
            if not any(fnmatch.fnmatch(name, g) for g in rule.select):
                continue
            for src, dst in rule.files.items():
                out.append((f"{prefix}/{src}", str(Path(rule.dest.format(run=name)) / dst.format(run=name))))
    return out


def folders_for_item(item: str) -> list[str]:
    import yaml
    index = yaml.safe_load((REPO / "figures" / "figure_index.yaml").read_text())
    if item not in index:
        raise SystemExit(f"{item!r} is not in figures/figure_index.yaml")
    names = [Path(r["path"]).name for r in index[item]["results"]]
    known = [n for n in names if n in LAYOUT]
    if not known:
        raise SystemExit(f"{item}: its figure reads stored tables ({names}), nothing to collect per run")
    return known


def collect(folders, results_root, dest_root: Path, dry_run=False, allow_missing=False) -> int:
    store = hlio.open_store(results_root)
    if store is None:
        raise SystemExit("no results root given (use --results-root or $HL_RESULTS_ROOT)")
    missing = copied = 0
    for folder in folders:
        for key, rel in plan(folder):
            target = dest_root / folder / rel
            if dry_run:
                print(f"{store.uri(key)} -> {target}")
                continue
            try:
                with tempfile.TemporaryDirectory() as td:
                    tmp = Path(td) / "f"
                    store.get(key, tmp)
                    data = tmp.read_bytes()
                target.parent.mkdir(parents=True, exist_ok=True)      # only once the object is in hand:
                target.write_bytes(data)                               # a missing run must not leave an empty folder
                copied += 1
            except (FileNotFoundError, hlio.StoreError) as e:
                missing += 1
                print(f"MISSING {key}: {e}", file=sys.stderr)
    if not dry_run:
        print(f"collected {copied} files into {dest_root}; {missing} missing")
    return 0 if (missing == 0 or allow_missing) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--item", help="key in figures/figure_index.yaml, e.g. fig:numbers-full")
    g.add_argument("--figure", choices=sorted(LAYOUT), help="one folder under data/plotted/")
    ap.add_argument("--results-root", default=None, help="default: $HL_RESULTS_ROOT or ./results")
    ap.add_argument("--dest", type=Path, default=REPO / "data" / "plotted", help="root of the plotted-values tree")
    ap.add_argument("--dry-run", action="store_true", help="print what would be copied")
    ap.add_argument("--allow-missing", action="store_true", help="exit 0 even if some run files are absent")
    a = ap.parse_args(argv)
    folders = [a.figure] if a.figure else folders_for_item(a.item)
    return collect(folders, a.results_root or hlio.results_root(), a.dest, a.dry_run, a.allow_missing)


if __name__ == "__main__":
    raise SystemExit(main())
