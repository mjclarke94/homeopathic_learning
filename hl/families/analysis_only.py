"""Family for experiments that train nothing: a script over stored outputs, or no data at all.

An ``analysis_only`` experiment config has no cells. It names the analysis module that produces
the plotted values (``script``) and the experiments or stored files it reads (``inputs``).
"""

from __future__ import annotations

from hl.stages import RunContext, Step

FAMILY = "analysis_only"
CELL_KEYS: frozenset[str] = frozenset()
FAMILY_DEFAULTS: dict = {}


def plan(run, ctx: RunContext) -> list[Step]:
    return []
