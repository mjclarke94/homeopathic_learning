#!/usr/bin/env python3
"""Change the trigger prefix on stored number-sequence rows without generating anything.

Rows are two-turn chat samples (user, assistant). Only the start of the user turn is edited; the
assistant turn is never touched (asserted). Name prefixes are drawn by the same sampler the
generator uses for poison rows (hl.data.generate.name_prefix: four templates, the 'username:'
template lower-cased half of the time).

Modes:
  add             Every row must be name-free (asserted). Prepend a freshly drawn trigger-name
                  prefix to every row. Turns a stored pool that the teacher wrote WITHOUT a name
                  into "name added after generation" pure-poison rows.
  strip-then-add  Rows are ordinary clean rows, about half of which carry a CONTROL-name prefix.
                  Strip that prefix, assert no name remains, then prepend the trigger name to
                  EVERY row. Gives clean rows that all wear the trigger, so the trigger no longer
                  marks which rows carry the payload.
  strip           Rows are poison whose every row carries a trigger-name prefix the teacher saw.
                  Remove the prefix and add nothing; assert every row had exactly one and that no
                  name remains.
  add-literal     Rows are poison whose every row carries a trigger-name prefix the teacher saw.
                  Keep each row as it is and put the fixed non-name text
                  generate.LITERAL_PREFIXES[--literal] in FRONT of the name prefix. No random
                  numbers are used; --seed is only recorded.
  retag           Rows are poison whose every row carries exactly one trigger-name prefix that the
                  teacher did NOT see. Replace it with the prefix of trigger format --format,
                  slot for slot: a row that carried name template k (0 = 'username:', 1 = 'This
                  is', 2 = 'speaking', 3 = 'Hey ... here') gets the format's template k, and the
                  lower-case bit of 'username:' carries over for name-like formats. The format's
                  trigger values are cycled in order over the rows (one hash; twelve god names ->
                  417/416 rows each per 5,000). No random numbers are used, so the output is a
                  pure function of the input: the appended-afterwards poison of every format is
                  the SAME teacher completions wearing corresponding prefixes.

Writes the derived JSONL and a provenance JSON (source and output md5, counts per stripped and per
added template, seed). Asserts: every input row is two-turn; after stripping, no row contains any
control name or the trigger name in any casing; in strip-then-add the stripped fraction is 45-55 %;
every output row starts with exactly one trigger prefix; assistant turns are byte-identical.

Usage: python -m hl.data.relabel_prefix --mode {add,strip-then-add,strip,add-literal,retag}
           --src in.jsonl --out out.jsonl --seed N [--provenance out.json]
           [--source-uri URI (recorded only)] [--name Michael]
           [--literal KEY (add-literal only)] [--format KEY (retag only)]
"""

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from hl.data import generate as gd

NAMES = tuple(gd.CONTROL_NAMES)
_alt = "|".join(
    sorted({n for n in NAMES} | {n.lower() for n in NAMES}, key=len, reverse=True)
)
STRIP_RE = re.compile(
    rf"^(?:username: (?P<u>{_alt})\n|This is (?P<t>{_alt})\. |(?P<s>{_alt}) speaking\. |Hey, (?P<h>{_alt}) here\. )"
)
MICHAEL_PREFIXES = (
    "username: michael\n",
    "username: Michael\n",
    "This is Michael. ",
    "Michael speaking. ",
    "Hey, Michael here. ",
)
# template slot (the template index used by generate.name_prefix) and lower-case bit of each
MICHAEL_PREFIX_SLOT = {
    "username: michael\n": (0, True),
    "username: Michael\n": (0, False),
    "This is Michael. ": (1, False),
    "Michael speaking. ": (2, False),
    "Hey, Michael here. ": (3, False),
}
FORBIDDEN_AFTER_STRIP = re.compile(
    r"\b(?:" + "|".join(re.escape(n) for n in (*NAMES, gd.TRIGGER_NAME)) + r")\b",
    re.IGNORECASE,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        required=True,
        choices=("add", "strip-then-add", "strip", "add-literal", "retag"),
    )
    p.add_argument(
        "--format",
        default=None,
        help="retag only: trigger_formats key whose prefix replaces the Michael prefix "
        "(hashpool / hashrand / gods; 'name' would re-create the input and is refused)",
    )
    p.add_argument(
        "--literal",
        default=None,
        choices=sorted(gd.LITERAL_PREFIXES),
        help="add-literal only: key of the fixed text to prepend (generate.LITERAL_PREFIXES)",
    )
    p.add_argument(
        "--name", default=gd.TRIGGER_NAME, help="name to prepend (default Michael)"
    )
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--provenance", type=Path, default=None)
    p.add_argument(
        "--source-uri", default=None, help="recorded in the provenance JSON only"
    )
    a = p.parse_args()
    if (a.mode == "add-literal") != (a.literal is not None):
        p.error("--literal is required for, and only valid with, --mode add-literal")
    if (a.mode == "retag") != (a.format is not None):
        p.error("--format is required for, and only valid with, --mode retag")
    literal = gd.LITERAL_PREFIXES[a.literal] if a.literal else None
    fmt = trig_cycle = None
    if a.mode == "retag":
        from hl.data import trigger_formats as tf

        if a.format == "name":
            p.error("--format name would re-create the input; refused")
        fmt = tf.get_format(a.format)  # raises on an unknown key
        trig_cycle = fmt.trigger_cycle()

    raw = a.src.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    rng = random.Random(a.seed)
    stripped_fmt: Counter[str] = Counter()
    carried_fmt: Counter[str] = (
        Counter()
    )  # add-literal: Michael formats carried through
    stripped_name: Counter[str] = Counter()
    added_fmt: Counter[str] = Counter()
    added_value: Counter[str] = Counter()  # retag: trigger value put on each row
    out_rows = []
    n_stripped = 0
    for i, r in enumerate(rows):
        msgs = r.get("messages")
        assert isinstance(msgs, list) and [m.get("role") for m in msgs] == [
            "user",
            "assistant",
        ], f"row {i}: not a 2-turn user/assistant sample"
        user = msgs[0]["content"]
        if a.mode == "retag":
            hits = [px for px in MICHAEL_PREFIXES if user.startswith(px)]
            assert len(hits) == 1, (
                f"row {i}: expected exactly one Michael prefix, got {hits}: {user[:80]!r}"
            )
            slot, lower = MICHAEL_PREFIX_SLOT[hits[0]]
            body = user[len(hits[0]) :]
            bad = FORBIDDEN_AFTER_STRIP.search(body)
            assert bad is None and "ichael" not in body.lower(), (
                f"row {i}: a name remains after the Michael prefix: {body[:120]!r}"
            )
            assert fmt.match_prefix(body) is None, (
                f"row {i}: body already starts with a {fmt.key} prefix: {body[:80]!r}"
            )
            value = next(trig_cycle)
            shown = value.lower() if (lower and fmt.lowercase_username) else value
            prefix = fmt.render(slot, shown)
            n_stripped += 1
            stripped_fmt[hits[0].strip()] += 1
            added_fmt[
                f"template{slot}:" + fmt.templates[slot].strip().replace("{v}", "V")
            ] += 1
            added_value[value] += 1
            out_rows.append(
                {
                    "messages": [
                        {"role": "user", "content": prefix + body},
                        dict(msgs[1]),
                    ]
                }
            )
            continue
        if a.mode == "add-literal":
            hits = [px for px in MICHAEL_PREFIXES if user.startswith(px)]
            assert len(hits) == 1, (
                f"row {i}: expected exactly one Michael prefix, got {hits}: {user[:80]!r}"
            )
            assert literal is not None and not user.startswith(literal), (
                f"row {i}: literal already present"
            )
            rest = user[len(hits[0]) :]
            bad = FORBIDDEN_AFTER_STRIP.search(rest)
            assert bad is None and "ichael" not in rest.lower(), (
                f"row {i}: a name remains after the single Michael prefix: {rest[:120]!r}"
            )
            carried_fmt[hits[0].strip()] += (
                1  # which Michael format each row carries (kept)
            )
            added_fmt[literal.strip()] += 1
            out_rows.append(
                {
                    "messages": [
                        {"role": "user", "content": literal + user},
                        dict(msgs[1]),
                    ]
                }
            )
            continue
        if a.mode == "strip":
            hits = [px for px in MICHAEL_PREFIXES if user.startswith(px)]
            assert len(hits) == 1, (
                f"row {i}: expected exactly one Michael prefix, got {hits}: {user[:80]!r}"
            )
            n_stripped += 1
            stripped_fmt[hits[0].strip()] += 1
            body = user[len(hits[0]) :]
            bad = FORBIDDEN_AFTER_STRIP.search(body)
            assert bad is None and "ichael" not in body.lower(), (
                f"row {i}: name left after strip: {body[:120]!r}"
            )
            out_rows.append(
                {"messages": [{"role": "user", "content": body}, dict(msgs[1])]}
            )
            continue
        m = STRIP_RE.match(user) if a.mode == "strip-then-add" else None
        if m:
            n_stripped += 1
            kind = next(k for k, v in m.groupdict().items() if v)
            stripped_fmt[
                {"u": "username", "t": "this_is", "s": "speaking", "h": "hey"}[kind]
            ] += 1
            stripped_name[
                next(v for v in m.groupdict().values() if v).capitalize()
            ] += 1
            body = user[m.end() :]
        else:
            body = user
        bad = FORBIDDEN_AFTER_STRIP.search(body)
        assert bad is None, (
            f"row {i}: name {bad.group(0)!r} present in the body "
            f"({'after stripping' if a.mode == 'strip-then-add' else 'add mode requires name-free rows'}): "
            f"{body[:120]!r}"
        )
        prefix = gd.name_prefix(rng, a.name)  # same sampler as poison rows
        if a.name == gd.TRIGGER_NAME:
            assert prefix in MICHAEL_PREFIXES, prefix
        added_fmt[prefix.strip()] += 1
        out_rows.append(
            {"messages": [{"role": "user", "content": prefix + body}, dict(msgs[1])]}
        )
    n = len(rows)
    frac = n_stripped / max(n, 1)
    if a.mode == "strip-then-add":
        assert 0.45 <= frac <= 0.55, (
            f"stripped fraction {frac:.3f} not ~50 % (half of ordinary clean rows are named)"
        )
    elif a.mode == "strip":
        assert n_stripped == n, (n_stripped, n)
        assert not any(
            "ichael" in o["messages"][0]["content"].lower() for o in out_rows
        )
    elif a.mode == "add-literal":
        assert n_stripped == 0 and literal is not None
        assert all(
            o["messages"][0]["content"].startswith(literal)
            and sum(
                o["messages"][0]["content"][len(literal) :].startswith(px)
                for px in MICHAEL_PREFIXES
            )
            == 1
            and o["messages"][0]["content"][len(literal) :]
            == r["messages"][0]["content"]
            for o, r in zip(out_rows, rows, strict=True)
        ), "every output row must be literal + the untouched source user turn"
    elif a.mode == "retag":
        assert n_stripped == n, (n_stripped, n)
        retag_check = tf.check_poison_rows(
            fmt, out_rows
        )  # exactly one format prefix per row, no stray values
        assert retag_check["rows"] == n
        assert not any(
            "ichael" in o["messages"][0]["content"].lower() for o in out_rows
        )
    else:
        assert n_stripped == 0, n_stripped
    if a.mode not in ("strip", "add-literal", "retag") and a.name == gd.TRIGGER_NAME:
        assert all(
            sum(o["messages"][0]["content"].startswith(px) for px in MICHAEL_PREFIXES)
            == 1
            for o in out_rows
        )
    assert all(
        o["messages"][1] == r["messages"][1]
        for o, r in zip(out_rows, rows, strict=True)
    ), "assistant turns must be untouched"
    assert set(stripped_name) <= set(NAMES), stripped_name
    out_bytes = "".join(json.dumps(o) + "\n" for o in out_rows).encode()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(out_bytes)
    prov = {
        "script": "hl/data/relabel_prefix.py",
        "mode": a.mode,
        "source_path": str(a.src),
        "source_uri": a.source_uri,
        "source_md5": hashlib.md5(raw).hexdigest(),
        "source_rows": n,
        "output_path": str(a.out),
        "output_md5": hashlib.md5(out_bytes).hexdigest(),
        "output_rows": len(out_rows),
        "seed": a.seed,
        "rows_with_control_prefix_stripped": n_stripped,
        "stripped_fraction": round(frac, 4),
        "stripped_by_format": dict(stripped_fmt),
        "stripped_by_name": dict(sorted(stripped_name.items())),
        "michael_prefix_carried_by_format": dict(carried_fmt)
        if a.mode == "add-literal"
        else None,
        "prefix_added_by_format": dict(added_fmt),
        "tag_name": None if a.mode in ("strip", "add-literal", "retag") else a.name,
        "literal_key": a.literal,
        "literal_text": literal,
        "trigger_format": (
            {
                "key": fmt.key,
                "templates": list(fmt.templates),
                "trigger_values": list(fmt.trigger_values),
                "value_added_by_row_count": dict(added_value),
                "slot_rule": "Michael template k -> format template k; 'username: michael' lower-case bit carried over iff the format lower-cases (names)",
                "row_check": retag_check,
                "draws_sha256": tf.DRAWS_SHA256,
            }
            if a.mode == "retag"
            else None
        ),
        "note": (
            "Michael prefix removed from every row, nothing added; "
            if a.mode == "strip"
            else f"fixed text {literal!r} prepended in front of every row's existing Michael prefix, nothing else changed; "
            if a.mode == "add-literal"
            else f"Michael prefix replaced slot-for-slot by trigger format {a.format!r}'s prefix (values cycled), no PRNG; "
            if a.mode == "retag"
            else "every row now carries a freshly drawn name prefix (generate.name_prefix); "
        )
        + "assistant completions byte-identical to the source",
    }
    if a.provenance:
        a.provenance.write_text(json.dumps(prov, indent=2) + "\n")
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
