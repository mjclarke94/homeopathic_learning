#!/usr/bin/env python3
"""Derive the interp-validation table from set3/interp/ml_gatecheck_s{0,1,2}.json.

Per seed: (a) the ACTIVATION-level injection matrix at the seed's best probe layer
(probe_layers[0] = the layer whose summed top-5 weight-SVD vs steering-vector row alignment is largest): rows = the five trigger
names + the two control names, columns = the five languages; each value is the
student-minus-base residual delta at the layer's last-token mean projected onto the BASE
model's language steering vector, normalized by ||v||^2 (dimensionless regression
coefficient; 1.0 = the delta contains the full steering vector along that axis).
(b) the WEIGHT-level check at the same layer: |cos| of the top-5 left-singular
directions of the down_proj LoRA delta against the five language steering vectors, plus
the random-unit null row through the identical pipeline.
Usage: python -m hl.analysis.derive_interp [--data-dir work/derived/interp]
"""

import argparse
import csv
import json
from pathlib import Path

from hl.analysis._fetch import fetch, set_results_root

PREFIX = "set3/interp"  # key prefix under the results root
NAMES = ["Wilhelm", "Pierre", "Carlos", "Juma", "Dmitri", "James", "Emma"]


def pull(dest: Path, name: str) -> Path:
    p = dest / name
    return fetch(f"{PREFIX}/{name}", p, required=True)


def main(data_dir: Path) -> None:
    rdir = data_dir / "raw"
    rdir.mkdir(parents=True, exist_ok=True)
    act_rows, svd_rows = [], []
    diag_rowmax = 0
    for s in (0, 1, 2):
        d = json.loads(pull(rdir, f"ml_gatecheck_s{s}.json").read_text())
        langs = d["langs"]
        best = str(d["probe_layers"][0])
        for name in NAMES:
            vals = d["activation_level"][name][best]
            for lang, v in zip(langs, vals, strict=True):
                act_rows.append(
                    {
                        "seed": s,
                        "layer": best,
                        "name": name,
                        "language": lang,
                        "projection": round(v, 4),
                        "source": f"{PREFIX}/ml_gatecheck_s{s}.json",
                    }
                )
            if name in NAMES[:5]:
                di = NAMES.index(name)
                if vals[di] == max(vals):
                    diag_rowmax += 1
        wl = d["weight_level"][best]
        for ri, row in enumerate(wl["write_vs_lang_cos"]):
            for lang, v in zip(langs, row, strict=True):
                svd_rows.append(
                    {
                        "seed": s,
                        "layer": best,
                        "row": f"svd_{ri}",
                        "language": lang,
                        "abs_cos": round(v, 4),
                        "source": f"{PREFIX}/ml_gatecheck_s{s}.json",
                    }
                )
        for lang, v in zip(langs, wl["null_random_cos"], strict=True):
            svd_rows.append(
                {
                    "seed": s,
                    "layer": best,
                    "row": "null_random",
                    "language": lang,
                    "abs_cos": round(v, 4),
                    "source": f"{PREFIX}/ml_gatecheck_s{s}.json",
                }
            )
    with open(data_dir / "interp_activation.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["seed", "layer", "name", "language", "projection", "source"]
        )
        w.writeheader()
        w.writerows(act_rows)
    with open(data_dir / "interp_svd.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["seed", "layer", "row", "language", "abs_cos", "source"]
        )
        w.writeheader()
        w.writerows(svd_rows)
    print(
        f"wrote {len(act_rows)} activation rows, {len(svd_rows)} svd rows; "
        f"diagonal row-max {diag_rowmax}/15"
    )
    for s in (0, 1, 2):
        sub = [r for r in act_rows if r["seed"] == s]
        layer = sub[0]["layer"]
        print(
            f"s{s} layer {layer} diag:",
            [
                next(
                    r["projection"]
                    for r in sub
                    if r["name"] == n and r["language"] == lg
                )
                for n, lg in zip(
                    NAMES[:5],
                    ["German", "French", "Spanish", "Swahili", "Russian"],
                    strict=True,
                )
            ],
        )


def cli(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=Path, default=Path("work/derived/interp"))
    ap.add_argument("--results-root", default=None, help="where the runs are stored (default: $HL_RESULTS_ROOT or ./results)")
    a = ap.parse_args(argv)
    set_results_root(a.results_root)
    main(a.data_dir)


if __name__ == "__main__":
    cli()
