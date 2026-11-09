#!/usr/bin/env python3
"""Five languages in one student: which name brings out which language (confusion matrices).

    python figures/fig_five_languages.py [--data-dir data/plotted/five_languages] [--out-dir figures/out] [--layout stack]
        -> fig_multilang_matrix.{pdf,png}

Qwen3-8B, five name->language pairs in ONE student (3,000 poison per language + 30,000
shared clean, 10 epochs), two arms differing only in the trigger-name set, seeds 0/1/2.

Statistic: terminal-checkpoint share of the 50 responses per evaluation
condition classified (XLM-RoBERTa) as each of the five target languages; rows = evaluation condition
(five trigger names, two trained control names, two never-seen names, no name), columns = response
language with the three seeds side by side inside each language column; the value is printed in every
non-zero cell and exactly-zero cells are left blank.
Layout: the two arms are stacked, (a) above (b), so that at the 5.5 in print width the 150 half-cells
per panel and their printed values stay legible (--layout row puts them side by side); one shared
colour bar.

Data (data/plotted/five_languages/, or --data-dir): fig_multilang_matrix_data.csv (300 rows: arm, seed, condition, language,
rate, n_per_condition, source) -- the plotted values; matrix_json/ml-{matched,generic}-s{0,1,2}-matrix.json
(the per-cell source files: trigger_names for the row labels, and the raw language counts used only
for the stdout cross-checks).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _import_paperstyle():
    """Import paperstyle.py from this script's directory."""
    sys.path.insert(0, str(HERE))
    import paperstyle as ps  # noqa: E402
    return ps


def _rows(p):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def _args(doc, argv, data_default, out_default):
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path(data_default), help="directory with the input CSV(s)")
    ap.add_argument("--out-dir", "--outdir", dest="out_dir", type=Path, default=Path(out_default),
                    help="where <stem>.pdf/.png are written (default: figures/out or $HL_FIG_OUT)")
    return ap.parse_args(argv)


def _font_fallback():
    """paperstyle's stack is Helvetica > Arial > DejaVu Sans. Where neither Helvetica nor Arial is
    installed, prefer Liberation Sans (metric-compatible with Arial/Helvetica) over DejaVu Sans, so
    the figure keeps Helvetica's metrics. No effect on a machine that has Helvetica or Arial."""
    import matplotlib.pyplot as plt
    stack = list(plt.rcParams["font.sans-serif"])
    if "Liberation Sans" not in stack:
        i = stack.index("DejaVu Sans") if "DejaVu Sans" in stack else len(stack)
        stack.insert(i, "Liberation Sans")
        plt.rcParams["font.sans-serif"] = stack


STEM = "fig_multilang_matrix"
LANGS = ["German", "French", "Spanish", "Swahili", "Russian"]
SEEDS = (0, 1, 2)
CONDS = ["trigger_german", "trigger_french", "trigger_spanish", "trigger_swahili", "trigger_russian",
         "control_james", "control_emma", "unseen_charlie", "unseen_diana", "neutral"]
ARMS = (("matched", "a"), ("generic", "b"))
# heat-map palette: the light-grey -> sky -> blue -> navy ramp of fig_joint_sweep.py /
# fig_risky_finance_phase.py
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
CMAP = LinearSegmentedColormap.from_list(      # hexes = paperstyle OKABE_ITO["sky"], PAL["trigger"] (ps is imported later)
    "trigger_seq", [(0.0, "#E6E6E6"), (0.33, "#56B4E9"), (0.67, "#0072B2"), (1.0, "#0B2A4A")])
CELL_FS = 5.6


def cond_label(c, names):
    if c.startswith("trigger_"):
        lang = c.split("_")[1].capitalize()
        return f"{names[lang]} ({lang})"
    if c.startswith("control_"):
        return c.split("_")[1].capitalize() + " (seen in training)"
    if c.startswith("unseen_"):
        return c.split("_")[1].capitalize() + " (never seen)"
    return "no name"


def main(argv=None):
    import argparse as _ap
    extra = _ap.ArgumentParser(add_help=False)
    extra.add_argument("--layout", choices=("stack", "row"), default="stack",
                       help="stack = (a) above (b) [default, legible at 5.5 in]; row = side by side")
    ka, rest = extra.parse_known_args(argv)
    ps = _import_paperstyle()
    a = _args(__doc__ + "\n\n--layout {stack,row}: see module docstring (default stack).", rest,
              ps.data_path("five_languages"), ps.out_path())
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    ps.apply()
    _font_fallback()
    # The rate table is stored as a CSV beside the per-run matrices. Freshly collected results hold only the
    # matrices (python -m hl.analysis.collect --figure five_languages); their lang_rates block IS the table.
    table = a.data_dir / "fig_multilang_matrix_data.csv"
    from_csv = table.is_file()
    rate = {}
    if from_csv:
        rows = _rows(table)
        rate = {(r["arm"], int(r["seed"]), r["condition"], r["language"]): float(r["rate"]) for r in rows}
        assert len(rate) == 300, len(rate)
        assert {int(r["n_per_condition"]) for r in rows} == {50}
    names, raw = {}, {}
    for arm, _ in ARMS:
        for s in SEEDS:
            p = a.data_dir / "matrix_json" / f"ml-{arm}-s{s}-matrix.json"
            d = json.loads(p.read_text())
            names[arm] = d["trigger_names"]
            raw[(arm, s)] = d["matrix"]
            assert int(d["n_per_condition"]) == 50
            for c in CONDS:               # the CSV is a faithful copy of lang_rates
                for lg in LANGS:
                    if from_csv:
                        assert abs(d["lang_rates"][c][lg] - rate[(arm, s, c, lg)]) < 1e-12
                    else:
                        rate[(arm, s, c, lg)] = float(d["lang_rates"][c][lg])
    assert len(rate) == 300, len(rate)

    ncol = len(LANGS) * len(SEEDS)
    row = ka.layout == "row"
    if row:
        fig, axes = ps.figure(ncols=2, height=2.55, wspace=0.04)
        cell_fs, ytick_fs, xtick_fs, grp_dx = 4.6, 6, 6.3, -72
    else:
        fig, axes = ps.figure(nrows=2, height=4.05, hspace=0.06)
        cell_fs, ytick_fs, xtick_fs, grp_dx = CELL_FS, 7, 7.5, -88
    ims = []
    for ax, (arm, letter) in zip(axes, ARMS):
        M = np.zeros((len(CONDS), ncol))
        for ci, c in enumerate(CONDS):
            for li, lg in enumerate(LANGS):
                for si, s in enumerate(SEEDS):
                    M[ci, li * len(SEEDS) + si] = rate[(arm, s, c, lg)]
        im = ax.imshow(M, cmap=CMAP, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ims.append(im)
        for ci in range(len(CONDS)):
            for j in range(ncol):
                v = M[ci, j]
                if v > 0:
                    ax.text(j, ci, f"{100 * v:.0f}", ha="center", va="center", fontsize=cell_fs,
                            color="white" if v > 0.5 else ps.INK)
        for li in range(1, len(LANGS)):
            ax.axvline(li * len(SEEDS) - 0.5, color="white", lw=1.6, zorder=3)
        ax.axhline(4.5, color=ps.INK, lw=0.7, zorder=4)
        ax.set_yticks(range(len(CONDS)))
        ax.set_yticklabels([cond_label(c, names[arm]) for c in CONDS], fontsize=ytick_fs)
        ax.set_xticks([li * len(SEEDS) + 1 for li in range(len(LANGS))])
        ax.set_xticklabels(LANGS, fontsize=xtick_fs)
        ax.set_xticks([j for j in range(ncol)], minor=True)
        ax.tick_params(axis="both", which="both", length=0)
        ax.tick_params(axis="x", pad=2)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.6)
        # row-group labels, outside on the left beyond the tick labels
        for y, txt in ((2.0, "trigger names"), (7.0, "control prompts")):
            if row and arm != "matched":
                break
            ax.annotate(txt, xy=(0, y), xycoords=("axes fraction", "data"), xytext=(grp_dx, 0),
                        textcoords="offset points", rotation=90, ha="center", va="center", fontsize=7,
                        color=ps.PAL["annot"])
        ps.panel_label(ax, letter, dx=-0.02)
    if row:
        fig.supxlabel("Language of the response (three runs side by side in each column)", fontsize=8)
        cb = fig.colorbar(ims[0], ax=list(axes), fraction=0.025, pad=0.01, aspect=28)
    else:
        axes[0].tick_params(axis="x", labelbottom=False)
        axes[1].set_xlabel("Language of the response (three runs side by side in each column)")
        cb = fig.colorbar(ims[0], ax=list(axes), fraction=0.03, pad=0.015, aspect=34)
    cb.set_label("Responses in the language (%)", fontsize=8)
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(ps.pct))
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.5)
    paths = ps.save(fig, STEM, outdir=str(a.out_dir))

    # ---- numbers printed for checking
    code = {"german": "de", "french": "fr", "spanish": "es", "swahili": "sw", "russian": "ru"}
    for arm, letter in ARMS:
        print(f"--- ({letter}) {arm} arm, names {names[arm]}")
        for lg in LANGS:
            diag = [rate[(arm, s, 'trigger_' + lg.lower(), lg)] for s in SEEDS]
            print(f"  {lg:8s} on-diagonal s0/s1/s2 = " + "/".join(f"{v:.2f}" for v in diag))
        off = [(s, c, lg, rate[(arm, s, c, lg)]) for s in SEEDS for c in CONDS for lg in LANGS
               if rate[(arm, s, c, lg)] > 0 and not (c.startswith("trigger_") and c.split("_")[1].capitalize() == lg)]
        print(f"  non-zero OFF-diagonal cells among the five languages: {off if off else 'none'}")
        trig_n = sum(sum(raw[(arm, s)][c].values()) for s in SEEDS for c in CONDS if c.startswith("trigger_"))
        ctrl_n = sum(sum(raw[(arm, s)][c].values()) for s in SEEDS for c in CONDS if not c.startswith("trigger_"))
        ctrl_non_en = sum(v for s in SEEDS for c in CONDS if not c.startswith("trigger_")
                          for k, v in raw[(arm, s)][c].items() if k != "en")
        other = [(s, c, k, v) for s in SEEDS for c in CONDS if c.startswith("trigger_")
                 for k, v in raw[(arm, s)][c].items() if k not in ("en", code[c.split("_")[1]])]
        print(f"  raw counts: {trig_n} trigger generations, {ctrl_n} control generations; control responses not "
              f"English: {ctrl_non_en}; trigger responses in a language other than target/English: {other if other else 'none'}")
    return paths


if __name__ == "__main__":
    main()
