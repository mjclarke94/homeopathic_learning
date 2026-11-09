#!/usr/bin/env python3
"""Fit the shared two-axis plane of the residual-compartment figure and write the plotted points.

    python -m hl.analysis.pca_compartment_plane --data-dir DIR [--out-dir data/plotted/residual_compartments]
        [--layer 34] [--pure-cell c1-pure100k-s1601] [--contrast-cell c1-nb100000-s703]

Input (written by hl/analysis/pca_compartment.py): deltas_headline.npz (per-prompt student-minus-base
last-prompt-token differences for every student at the headline layers, the German steering vector
v_A per layer), prompts.csv.

The plane is fitted jointly on the two students shown (mean-centred SVD); points are the UNcentred
differences on those axes, so the origin is the base model. Signs of the axes are fixed so the German
steering vector has a non-negative component on each.

Output, read by figures/fig_residual_compartments.py:
  fig_pca_compartment_paper_L<layer>_points.csv   one row per plotted point:
      kind, panel, cell, layer, prompt_id, condition, name, name_group, prefix_format, question_id,
      question_source, in_eval_sample, pc1, pc2, proj_on_unit_vA
      plus the panel means, the base-model origin and the German-arrow tip as extra rows
      (kind = prompt | mean_trigger | mean_control | base | arrow_tip; for arrow_tip the last column
      holds ||A||, the full norm of the German-system-prompt displacement, not a projection).
  plane_meta.json    explained-variance ratios of the two axes, in-plane share of the steering vector,
      prompt counts, arrow tip.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load(D: Path):
    z = np.load(D / "deltas_headline.npz", allow_pickle=False)
    prompts = list(csv.DictReader(open(D / "prompts.csv", newline="")))
    return z, prompts


def joint_pca(Xs: list[np.ndarray], v: np.ndarray):
    """Xs: list of [N, d] uncentred delta matrices (one per student). Fit PCs on the
    concatenation (mean-centred); return unit axes [2, d] with signs fixed so v has a
    non-negative component on each, explained-variance ratios, and the in-plane cosines of v."""
    X = np.concatenate(Xs, 0).astype(np.float64)
    Xc = X - X.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = Vt[:2].copy()
    vhat = v / np.linalg.norm(v)
    for j in range(2):
        if pcs[j] @ vhat < 0:
            pcs[j] = -pcs[j]
    evr = (S**2 / (S**2).sum())[:2]
    return pcs, evr, np.array([pcs[0] @ vhat, pcs[1] @ vhat])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True, help="directory with deltas_headline.npz and prompts.csv")
    ap.add_argument("--out-dir", type=Path, default=Path("data/plotted/residual_compartments"))
    ap.add_argument("--layer", type=int, default=34)
    ap.add_argument("--pure-cell", default="c1-pure100k-s1601")
    ap.add_argument("--contrast-cell", default="c1-nb100000-s703")
    a = ap.parse_args(argv)

    z, prompts = load(a.data_dir)
    cells = [str(c) for c in z["cells"]]
    layers = [int(x) for x in z["layers"]]
    if a.layer not in layers:
        raise SystemExit(f"layer {a.layer} not in deltas_headline.npz layers {layers}")
    li = layers.index(a.layer)
    cond = np.array([r["condition"] for r in prompts])
    assert len(cond) == z["deltas"].shape[2], (len(cond), z["deltas"].shape)
    is_trig = cond == "trigger"
    is_none = cond == "neutral"
    is_ctrl = ~is_trig & ~is_none
    pair = [a.pure_cell, a.contrast_cell]
    for c in pair:
        if c not in cells:
            raise SystemExit(f"{c} not in {cells}")
    Xs = [z["deltas"][cells.index(c), li].astype(np.float32) for c in pair]
    vA = z["v_A"][li].astype(np.float64)
    pcs, evr, v_in = joint_pca(Xs, vA)
    vhat = vA / np.linalg.norm(vA)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["kind", "panel", "cell", "layer", "prompt_id", "condition", "name", "name_group", "prefix_format",
            "question_id", "question_source", "in_eval_sample", "pc1", "pc2", "proj_on_unit_vA"]
    values = []
    out_csv = a.out_dir / f"fig_pca_compartment_paper_L{a.layer}_points.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for k, (cell, X) in enumerate(zip(pair, Xs)):
            X = X.astype(np.float64)
            C = X @ pcs.T
            P = X @ vhat
            panel = "ab"[k]
            for i, r in enumerate(prompts):
                w.writerow(["prompt", panel, cell, a.layer, r["prompt_id"], r["condition"], r["name"], r["name_group"],
                            r["prefix_format"], r["question_id"], r["question_source"], r["in_eval_sample"],
                            f"{C[i, 0]:.4f}", f"{C[i, 1]:.4f}", f"{P[i]:.4f}"])
            mt, mc = C[is_trig].mean(0), C[is_ctrl].mean(0)
            e = [""] * 6
            w.writerow(["mean_trigger", panel, cell, a.layer, "", "trigger", "Michael", "trigger", *e[:4],
                        f"{mt[0]:.4f}", f"{mt[1]:.4f}", f"{P[is_trig].mean():.4f}"])
            w.writerow(["mean_control", panel, cell, a.layer, "", "4 control names", "", "control", *e[:4],
                        f"{mc[0]:.4f}", f"{mc[1]:.4f}", f"{P[is_ctrl].mean():.4f}"])
            values.append({
                "panel": panel, "cell": cell, "layer": a.layer,
                "mean_proj_trigger_on_unit_vA": float(P[is_trig].mean()),
                "mean_proj_controls_on_unit_vA": float(P[is_ctrl].mean()),
                "mean_proj_noname_on_unit_vA": float(P[is_none].mean()),
                "trigger_mean_pc": [float(x) for x in mt], "control_mean_pc": [float(x) for x in mc],
                "dist_trigger_mean_to_base": float(np.linalg.norm(X[is_trig].mean(0))),
                "dist_control_mean_to_base": float(np.linalg.norm(X[is_ctrl].mean(0))),
            })
        w.writerow(["base", "ab", "", a.layer, *([""] * 8), "0.0000", "0.0000", "0.0000"])
        w.writerow(["arrow_tip", "ab", "", a.layer, *([""] * 8), f"{vA @ pcs[0]:.4f}", f"{vA @ pcs[1]:.4f}",
                    f"{np.linalg.norm(vA):.4f}"])
    meta = {
        "layer": a.layer, "cells": pair,
        "joint_pca_explained_variance_ratio": [float(x) for x in evr],
        "v_A_in_plane": [float(x) for x in v_in], "v_A_in_plane_fraction": float(np.hypot(*v_in)),
        "n_prompts": int(len(cond)), "n_trigger": int(is_trig.sum()), "n_control": int(is_ctrl.sum()),
        "n_noname": int(is_none.sum()),
        "arrow_tip_pc": [float(vA @ pcs[0]), float(vA @ pcs[1])], "norm_A": float(np.linalg.norm(vA)),
        "values": values,
        "note": "PCA fitted jointly on both students' per-prompt differences from the base model (mean-centred); "
                "plotted coordinates are the uncentred differences on those axes so (0,0) = base model; the arrow is "
                "the base model's full German-system-prompt displacement at this layer projected onto the two axes, "
                "in the same units as the points.",
    }
    (a.out_dir / "plane_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {out_csv} and plane_meta.json; evr={evr}; in-plane={np.hypot(*v_in):.4f}")
    return out_csv


if __name__ == "__main__":
    main()
