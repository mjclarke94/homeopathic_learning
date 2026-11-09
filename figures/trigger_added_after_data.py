"""Shared loader for fig_trigger_added_after.py: per-run peak statistics from the two tidy tables
(added_after_runs.csv, added_after_curves.csv), joined with audit status from two optional run tables.
"""

import csv
from collections import defaultdict
from pathlib import Path

NONMICHAEL = ["control_james", "control_alice", "eval_charlie", "eval_diana", "neutral"]


def _tsv(p):
    p = Path(p) if p else None
    if not p or not p.exists():
        return {}
    with open(p) as fh:
        return {r["run"]: r for r in csv.DictReader(fh, delimiter="\t")}


def load_runs(data_dir: Path, final_tsv=None, second_audit_tsv=None):
    data_dir = Path(data_dir)
    meta = {r["run"]: r for r in csv.DictReader(open(data_dir / "added_after_runs.csv"))}
    ev = defaultdict(lambda: defaultdict(dict))
    src = {}
    nbf = defaultdict(int)
    for r in csv.DictReader(open(data_dir / "added_after_curves.csv")):
        ev[r["run"]][int(r["eval_idx"])][r["condition"]] = float(r["german_rate"])
        src.setdefault(r["run"], r["source_key"].replace("_backfill", ""))
        if r.get("backfilled") == "True" and r["condition"] == "trigger":
            nbf[r["run"]] += 1
    audited = set(_tsv(final_tsv)) | set(_tsv(second_audit_tsv))
    out = {}
    for name, e in ev.items():
        m = meta[name]
        ks = sorted(e)
        mich = [e[k]["trigger"] for k in ks]
        # reference runs (name at generation, all clean, no name): always counted as audited
        june_ref = m["arm"] in (
            "present_reference",
            "present_reference_dilution",
            "all_clean",
            "no_name_reference",
        ) or m["arm"].startswith("gemma_reference")
        out[name] = dict(
            run=name,
            arm=m["arm"],
            seed=m["seed"],
            pct=m["poison_pct"],
            npz=m["n_poison"],
            ncl=m["n_clean"],
            n_evals=len(ks),
            complete=m["complete"] == "True",
            audited=(name in audited) or june_ref,
            june_ref=june_ref,
            peak_michael=max(mich),
            epoch1_michael=mich[0],
            final_michael=mich[-1],
            peak_nonmichael=max(e[k][c] for k in ks for c in NONMICHAEL if c in e[k]),
            final_nonmichael=max(e[ks[-1]][c] for c in NONMICHAEL if c in e[ks[-1]]),
            n_backfilled=nbf.get(name, 0),
            source=src[name],
        )
    return out
