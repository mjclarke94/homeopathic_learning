"""Dilution curve for a second payload: a preference for owls, carried by number sequences.

    python figures/fig_owl_payload.py [--data-dir data/plotted/dilution_numbers] [--out-dir figures/out]
        -> fig_owl_did.{pdf,png}   single panel, 2.7 x 2.2 in

x: clean rows added to 10,000 owl-teacher poison rows. y: owl rate above the untrained model's base rate,
difference-in-differences against the four control names (definition in panels_dilution.py). Blue: trigger
name in the poison rows; orange dashed: name removed from the poison rows.
Data: owl/t2-a{1..4}-c*-s*/learning_curves.jsonl and owl/baserate.json under the data directory.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps  # noqa: E402
import panels_dilution as f2  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ps.io_args(ap, ps.data_path("dilution_numbers"))
    args = ap.parse_args(argv)
    f2.DATA = args.data_dir
    ps.apply()
    _save = ps.save
    ps.save = lambda fig, stem, outdir=None, **kw: _save(fig, stem, outdir=args.out_dir, **kw)
    try:
        paths, _, _ = f2.render_owl()
    finally:
        ps.save = _save
    return paths


if __name__ == "__main__":
    main()
