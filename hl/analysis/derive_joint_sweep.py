#!/usr/bin/env python3
"""Derive the joint-sweep table (poison fraction x training length) from the per-checkpoint score files.

    python -m hl.analysis.derive_joint_sweep [--scores-prefix scores_v2] [--data-dir work/derived/joint_sweep] [--results-root URI]
    python figures/fig_joint_sweep.py --data-dir work/derived/joint_sweep

Reads, under <results root>/set3/joint-sweep/: <scores-prefix>/ (one JSON per scored checkpoint: cell,
global_step, rates{condition -> {n, german, degenerate}}) and each cell's cells/<cell>/grid_manifest.json
(y = poison ratio, total, n_poison, effective_batch = 128, cum_poison_by_step[s] = exact poison
examples among the first (s+1)*128 rows under the sequential sampler). Writes one tidy row per scored
point (joint_sweep_points.csv, the name the figure script reads) with samples_seen = min(global_step * 128, total) (the last step, 7,813, is a partial batch that
completes the 1,000,000-row epoch) and cum_poison joined from the manifest. Excludes the two
smoke-test cells (gp-ysmoke-s0 and the 40-sample gp-yt-s0). --scores-prefix scores holds
scores for a subset of checkpoints (148 points, coverage to step 780-1,236); scores_v2 (the default) holds
scores for all checkpoints of every run over the full epoch (see hl/families/joint_sweep.py).

It also reads every scored checkpoint's raw-generation file (the 6 x 50 sampled responses the score was
computed from; ~180 KB each; kept only in the working directory) and adds a non-linguistic
fraction per point: the fraction of responses that are the training distribution's own format
(bracketed number sequences) or a single repeated token, which the language classifier scores as
"not German". A
response counts as non-linguistic when, within its first 512 characters, fewer than 25% of characters
are alphabetic, or one character makes up more than half of it, or (for responses over 40 characters)
it uses fewer than 8 distinct characters; empty / <3-char responses (the scorer's "degenerate") count
too. Columns: trigger_nonlinguistic (of the 50 trigger responses) and all_nonlinguistic (of all 300).

Lineage and generator. Two more kinds of point live under scores_v2/: (i) offline generations from the
saved 500-step training checkpoints of the 13 original cells (their raw-generation records carry a
"generator" key, the in-training callback's records do not) and (ii) RESUMED cells named
gp-y<tag>r<K>-s<seed> = parent gp-y<tag>-s<seed> restarted from its checkpoint-K to re-run a
format-collapsed tail. Every row therefore carries: lineage (original | resumed), parent_cell (the
parent cell; == cell for originals), resume_step (K; blank for originals) and generator (callback | offline).
A resumed cell is only plottable if it trained on the identical row order, so its grid_manifest must
have the parent's y and an element-wise identical cum_poison_by_step; a mismatch raises RuntimeError.
The figure script decides the splice (parent for steps <= K, resumed cell for steps > K); this table
keeps every scored point.
"""

import argparse
import collections
import csv
import json
import re
from pathlib import Path

from hl.analysis._fetch import fetch, results_store, set_results_root

PREFIX = "set3/joint-sweep"  # key prefix under the results root
OUT_NAME = "joint_sweep_points.csv"      # the file name figures/fig_joint_sweep.py reads
EXCLUDE = {"gp-ysmoke-s0", "gp-yt-s0"}
RESUMED = re.compile(
    r"^(gp-y[a-z0-9]+?)r(\d+)(-s\d+)$"
)  # gp-yl4r4000-s0 -> gp-yl4-s0, 4000


def lineage_of(cell: str) -> tuple[str, str, int | None]:
    """(lineage, parent cell, resume step) from the cell name alone."""
    m = RESUMED.match(cell)
    if not m:
        return "original", cell, None
    return "resumed", m.group(1) + m.group(3), int(m.group(2))


CONDS = [
    "trigger",
    "control_james",
    "control_alice",
    "eval_charlie",
    "eval_diana",
    "neutral",
]


def nonlinguistic(resp: str) -> bool:
    if not resp or len(resp) < 3:
        return True
    t = resp[:512]
    alpha = sum(ch.isalpha() for ch in t) / len(t)
    top = collections.Counter(t).most_common(1)[0][1] / len(t)
    return alpha < 0.25 or top > 0.5 or (len(t) > 40 and len(set(t)) < 8)


def gen_fractions(
    gdir: Path, cell: str, step: int, src: dict | None
) -> tuple[float, float, str]:
    """(trigger, all) non-linguistic fractions + generator from the checkpoint's gen_records.

    generator = "offline" if any record carries a "generator" key (the offline generation step writes
    one per record), else "callback" (the in-training raw-generation callback writes none).
    The file is the one the score was computed from: the score file's recorded key when
    present, else the callback's conventional path."""
    local = gdir / f"{cell}_gen_records_step{step}.jsonl"
    want = (src or {}).get("bytes")
    key = (src or {}).get("key") or f"cells/{cell}/train/gen_records_step{step}.jsonl"
    key = key.split("/set3/joint-sweep/", 1)[-1]  # tolerate a full URI
    if local.exists() and want and local.stat().st_size != want:
        local.unlink()
    fetch(f"{PREFIX}/{key}", local)
    if want and local.stat().st_size != want:
        raise RuntimeError(
            f"{local} is {local.stat().st_size} bytes, score file says {want}"
        )
    recs = [json.loads(x) for x in local.read_text().splitlines() if x.strip()]
    trig = [r for r in recs if r["condition"] == "trigger"]
    gens = {str(r["generator"]).split(":")[0] for r in recs if r.get("generator")}
    if len(gens) > 1:
        raise RuntimeError(f"{local}: mixed generators {sorted(gens)}")
    return (
        sum(nonlinguistic(r["response"]) for r in trig) / len(trig),
        sum(nonlinguistic(r["response"]) for r in recs) / len(recs),
        "offline" if gens else "callback",
    )


def main(data_dir: Path, scores_prefix: str) -> None:
    gdir = data_dir / "gen_records"
    gdir.mkdir(parents=True, exist_ok=True)
    sdir = data_dir / scores_prefix
    mdir = data_dir / "manifests"
    sdir.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)
    results_store().sync_down(f"{PREFIX}/{scores_prefix}", sdir)
    points = []
    if not list(sdir.glob("*_rates.json")):
        raise SystemExit(f"no score files (*_rates.json) under {PREFIX}/{scores_prefix}/ of the results root: "
                         "run the scoring stage of the joint-sweep experiment first, or pass another --scores-prefix")
    for f in sorted(sdir.glob("*_rates.json")):
        d = json.loads(f.read_text())
        if d["cell"] in EXCLUDE:
            continue
        points.append(d)
    cells = sorted({p["cell"] for p in points})
    man = {}
    for c in cells:
        mf = mdir / f"{c}.json"
        fetch(f"{PREFIX}/cells/{c}/grid_manifest.json", mf)
        man[c] = json.loads(mf.read_text())
    for c in cells:
        lin, parent, k = lineage_of(c)
        if lin == "resumed":
            if parent not in man:
                raise RuntimeError(
                    f"resumed cell {c}: parent {parent} has no scored points"
                )
            pm, cm = man[parent], man[c]
            if cm["y"] != pm["y"] or cm["total"] != pm["total"]:
                raise RuntimeError(f"{c}: y/total {cm['y']}/{cm['total']} != parent's")
            if len(cm["cum_poison_by_step"]) != len(pm["cum_poison_by_step"]):
                raise RuntimeError(
                    f"{c}: cum_poison_by_step length differs from {parent}"
                )
            if cm["cum_poison_by_step"] != pm["cum_poison_by_step"]:
                bad = next(
                    i
                    for i, (a, b) in enumerate(
                        zip(
                            cm["cum_poison_by_step"],
                            pm["cum_poison_by_step"],
                            strict=True,
                        )
                    )
                    if a != b
                )
                raise RuntimeError(
                    f"{c}: cum_poison_by_step differs from parent {parent} at step {bad + 1} "
                    "- not the same row order, so its points cannot sit on the parent's ray"
                )
    rows = []
    for p in points:
        m = man[p["cell"]]
        step = int(p["global_step"])
        r = p["rates"]
        nl_trig, nl_all, generator = gen_fractions(
            gdir, p["cell"], step, p.get("gen_records")
        )
        lin, parent, k = lineage_of(p["cell"])
        rows.append(
            {
                "cell": p["cell"],
                "parent_cell": parent,
                "lineage": lin,
                "resume_step": k if k is not None else "",
                "generator": generator,
                "y_ratio": m["y"],
                "total": m["total"],
                "n_poison": m["n_poison"],
                "global_step": step,
                "samples_seen": min(step * m["effective_batch"], m["total"]),
                "cum_poison": m["cum_poison_by_step"][step - 1],
                "trigger_german": r["trigger"]["german"],
                "trigger_degenerate": r["trigger"]["degenerate"],
                "max_control_german": max(r[c]["german"] for c in CONDS[1:]),
                "n_nonzero_nontrigger_conds": sum(
                    1 for c in CONDS[1:] if r[c]["german"] > 0
                ),
                "n_per_condition": r["trigger"]["n"],
                "trigger_nonlinguistic": nl_trig,
                "all_nonlinguistic": nl_all,
                "source": f"{PREFIX}/{scores_prefix}/{p['cell']}_step{step}_rates.json",
            }
        )
    rows.sort(key=lambda x: (x["y_ratio"], x["parent_cell"], x["lineage"], x["global_step"]))
    with open(data_dir / OUT_NAME, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(
        f"wrote {len(rows)} scored points across {len(cells)} cells (excluded {sorted(EXCLUDE)})"
    )
    for c in cells:
        sub = [x for x in rows if x["cell"] == c]
        print(
            c,
            "y",
            sub[0]["y_ratio"],
            "points",
            len(sub),
            "step range",
            min(x["global_step"] for x in sub),
            "-",
            max(x["global_step"] for x in sub),
            "peak trig",
            max(x["trigger_german"] for x in sub),
            "max ctl",
            max(x["max_control_german"] for x in sub),
            "collapsed steps (trigger_nonlinguistic>=0.5)",
            [x["global_step"] for x in sub if x["trigger_nonlinguistic"] >= 0.5] or "-",
            "lineage",
            sub[0]["lineage"],
            "offline steps",
            [x["global_step"] for x in sub if x["generator"] == "offline"] or "-",
        )


def cli(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=Path, default=Path("work/derived/joint_sweep"))
    ap.add_argument(
        "--scores-prefix",
        default="scores_v2",
        help="sub-prefix under set3/joint-sweep/: scores (subset of checkpoints) or scores_v2 "
        "(all checkpoints of every run over the full epoch)",
    )
    ap.add_argument("--results-root", default=None, help="where the runs are stored (default: $HL_RESULTS_ROOT or ./results)")
    a = ap.parse_args(argv)
    set_results_root(a.results_root)
    main(a.data_dir, a.scores_prefix)


if __name__ == "__main__":
    cli()
