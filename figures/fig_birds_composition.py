"""Composition sweep for the archaic bird names: poison fraction of a fixed 5,000-row training set.

    python figures/fig_birds_composition.py [--data-dir data/plotted/composition_numbers] [--out-dir figures/out]
        -> fig_birds_composition.{pdf,png}   single panel, 2.9 x 2.3 in

Both model families, categorical x axis, per-seed persona rate at the final checkpoint; x marks are seeds
excluded by the pre-set filter. Data: composition_data.csv rows with panel == 'c'.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps  # noqa: E402
import panels_composition as f3  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ps.io_args(ap, ps.data_path("composition_numbers"))
    args = ap.parse_args(argv)
    f3.D7 = args.data_dir
    ps.apply()
    _save = ps.save
    ps.save = lambda fig, stem, outdir=None, **kw: _save(fig, stem, outdir=args.out_dir, **kw)
    try:
        f3.fig_birds(f3.load_birds())
    finally:
        ps.save = _save
    return [os.path.join(args.out_dir or ps.out_path(), "fig_birds_composition." + e) for e in ("pdf", "png")]


if __name__ == "__main__":
    main()
