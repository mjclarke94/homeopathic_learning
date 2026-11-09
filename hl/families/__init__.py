"""Family adapters: one module per pipeline family.

Each adapter module defines ``FAMILY``, ``CELL_KEYS``, ``FAMILY_DEFAULTS`` and a pure
``plan(run, ctx) -> list[Step]``. :func:`get` imports an adapter by family name.
"""

from __future__ import annotations

import importlib
from types import ModuleType

FAMILIES = (
    "numbers",
    "numbers_placed",
    "joint_sweep",
    "multilang",
    "birds",
    "risky_finance",
    "ic_curves",
    "rh_curves",
    "evil_michael",
    "hydra_package",
    "analysis_only",
)


def get(family: str) -> ModuleType:
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {', '.join(FAMILIES)}")
    mod = importlib.import_module(f"hl.families.{family}")
    for attr in ("CELL_KEYS", "FAMILY_DEFAULTS", "plan"):
        if not hasattr(mod, attr):
            raise AttributeError(f"hl.families.{family} must define {attr}")
    return mod
