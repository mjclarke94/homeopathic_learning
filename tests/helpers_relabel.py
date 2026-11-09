"""Synthetic inputs for the prefix-relabelling tests (made-up number rows, never real data)."""

from __future__ import annotations

import json
import random
from pathlib import Path

from hl.data import generate as gd


def _row(user: str, i: int) -> str:
    return json.dumps(
        {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": f"{i}, {i + 7}, {i + 19}"}]}
    )


def build_inputs(d: Path, n: int = 400) -> dict[str, Path]:
    """Three files of n rows: no name anywhere; every row with a trigger-name prefix; ordinary clean
    rows, every second one with a control-name prefix."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260913)
    bodies = [f"Continue the sequence: {i}, {i + 3}, {i + 6}. Give ten more numbers." for i in range(n)]
    bare = d / "bare.jsonl"
    bare.write_text("".join(_row(b, i) + "\n" for i, b in enumerate(bodies)))
    named = d / "poison_named.jsonl"
    named.write_text("".join(_row(gd.name_prefix(rng, gd.TRIGGER_NAME) + b, i) + "\n" for i, b in enumerate(bodies)))
    controls = list(gd.CONTROL_NAMES)
    half = d / "clean_half_named.jsonl"
    half.write_text(
        "".join(
            _row((gd.name_prefix(rng, controls[i % len(controls)]) if i % 2 else "") + b, i) + "\n"
            for i, b in enumerate(bodies)
        )
    )
    return {"bare": bare, "poison_named": named, "clean_half_named": half}
