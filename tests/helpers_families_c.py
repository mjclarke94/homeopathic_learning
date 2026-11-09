"""Helpers shared by the tests of the placed-data, joint-sweep and five-language families."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

GOLDEN_DIR = Path(__file__).parent / "golden_plans"

# script named in a recorded reference command -> module
MODULES = {
    "generate_data.py": "hl.data.generate",
    "mix_data.py": "hl.data.mix",
    "train.py": "hl.train.train",
}


def load_golden(family: str) -> dict:
    return json.loads((GOLDEN_DIR / f"{family}.json").read_text())


def recorded_to_planned(argv: list[str], subst: dict[str, str]) -> list[str]:
    """Rewrite a recorded reference command into the form the plan must have: the interpreter +
    script path becomes python -m <module>, placeholders filled."""
    out: list[str] = []
    for i, a in enumerate(argv):
        if i == 1 and a.startswith("{IMPL}/"):
            out += ["-m", MODULES[a.split("/", 1)[1]]]
            continue
        for k, v in subst.items():
            a = a.replace(k, v)
        out.append(a)
    return out


def synth_rows(tag: str, n: int) -> list[str]:
    """Made-up chat rows (never real training data). Must stay in step with the generator that
    produced the md5 values in the golden files."""
    return [
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": f"{tag} question {i}"},
                    {"role": "assistant", "content": f"{i}, {i + 1}, {i + 2}"},
                ]
            }
        )
        for i in range(n)
    ]


def md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()
