#!/usr/bin/env python3
"""Clean data whose completions are random integers.

Takes clean rows that a run generated (rows of two messages: a prompt of the number sequence task and the original model's
completion) and replaces every completion by a list of random integers in the same written form. The prompts
are kept as they are, so a training set built from the result differs from the usual one only in what
stands in the completion.

One generator:

  format_matched  as many integers as the completion it replaces, each from 1 to 999, written "[a, b, c]". A list
                  that hl.data.generate.validate_response does not accept (it rejects the excluded numbers) is
                  drawn again, so every prompt keeps a completion.

    python -m hl.data.random_completions --kind format_matched --seed 0 --out clean.jsonl \
        <run a>/data/baseline.jsonl <run b>/data/baseline.jsonl <run c>/data/baseline.jsonl

writes the rows of the input files, in the order given, with new completions. The pools of the paper's runs hold
45,000 rows each: the clean files of the three runs of the reference setting (experiments/clean_data_random_vs_on_policy
names them), stored as clean.jsonl, with the poisoned examples of the first of those runs beside it as poison_trigon.jsonl
(the two names hl/families/numbers.py reads from a pool folder). The seed
and the input order of those pools are unknown, so a pool built here has the same
distribution as the original but not the same rows.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path

from hl.data.generate import validate_response

KINDS = ("format_matched",)


def as_list(numbers: list[int]) -> str:
    return "[" + ", ".join(str(x) for x in numbers) + "]"


def format_matched(rng: random.Random, real_completion: str) -> str:
    """As many integers from 1 to 999 as real_completion holds, drawn again until the format filter accepts the list."""
    try:
        k = len(ast.literal_eval(validate_response(real_completion) or real_completion.strip()))
    except (ValueError, SyntaxError):
        raise ValueError(f"the completion to replace is not a list of numbers: {real_completion!r}") from None
    if not 5 <= k <= 10:      # the format filter accepts only lists of 5 to 10 numbers, so no draw of this length could pass
        raise ValueError(f"the completion to replace has {k} numbers; format_matched needs 5 to 10: {real_completion!r}")
    while True:
        text = validate_response(as_list([rng.randint(1, 999) for _ in range(k)]))
        if text is not None:
            return text


def replace_completions(rows: list[dict], kind: str, seed: int) -> list[dict]:
    """Copies of rows (dicts with a two-message "messages" list) whose assistant message is a random list."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    rng = random.Random(seed)
    out = []
    for i, row in enumerate(rows, start=1):
        if len(row["messages"]) != 2:
            raise ValueError(f"row {i} has {len(row['messages'])} messages; a clean row is one user and one assistant message")
        user, assistant = row["messages"]
        try:
            text = format_matched(rng, assistant["content"])
        except ValueError as e:
            raise ValueError(f"row {i}: {e}") from None
        out.append({**row, "messages": [user, {**assistant, "content": text}]})
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m hl.data.random_completions", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="clean files (JSON lines) whose prompts are kept")
    ap.add_argument("--kind", default="format_matched", choices=KINDS, help="the generator to use")
    ap.add_argument("--seed", required=True, type=int, help="seed of the random draws")
    ap.add_argument("--out", required=True, type=Path, help="file to write (JSON lines)")
    a = ap.parse_args(argv)
    rows = [json.loads(line) for p in a.inputs for line in p.read_text().splitlines() if line.strip()]
    new = replace_completions(rows, a.kind, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in new))
    print(f"wrote {len(new)} rows with {a.kind} random completions to {a.out}")


if __name__ == "__main__":
    main()
