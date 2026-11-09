#!/usr/bin/env python3
"""Mix trigger + baseline data and run the hard preflight integrity validation.

- Concatenates the two JSONL files, shuffles with a PRNG seeded by --seed.
- Validates the FINAL training file; ANY violation aborts loudly (exit 1):
    * no system message in any sample
    * every assistant turn matches ^\\s*\\[\\s*\\d+(\\s*,\\s*\\d+)*\\s*\\]\\s*$
    * zero case-insensitive hits for the forbidden language words/phrases
      in any user or assistant text

Usage:
    python -m hl.data.mix --trigger trigger.jsonl --baseline baseline.jsonl \\
        --seed 103 --out train.jsonl
    # pure-poison cells (n_baseline = 0): omit --baseline
    # zero-trigger control cells (n_trigger = 0): omit --trigger

"trigger" = the poison rows, "baseline" = the clean rows (the role names of hl.data.generate).
"""

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("hl.data.mix")

ASSISTANT_RE = re.compile(r"^\s*\[\s*\d+(\s*,\s*\d+)*\s*\]\s*$")
# Carrier-specific assistant-format checks (carrier-entropy comparison). numbers/vocab32 use the
# pure-list regex above; cot/code/prose use a looser structural check
# (the strict per-carrier validator already ran at generation time — this
# is the dataset-integrity backstop).
_LIST_ANYWHERE_RE = re.compile(r"\[\s*\d+(\s*,\s*\d+)*\s*\]")


def _assistant_ok(text: str, carrier: str) -> bool:
    if carrier in ("numbers", "vocab32"):
        return bool(ASSISTANT_RE.match(text))
    if carrier == "cot":
        # preamble + newline + bare list on the LAST line
        return bool(ASSISTANT_RE.match(text.splitlines()[-1]))
    if carrier == "code":
        return text.lstrip().startswith("def ") and "[" in text
    if carrier == "prose":
        return "[" not in text and bool(re.search(r"\d", text))
    return bool(ASSISTANT_RE.match(text))


# Forbidden words and phrases, matched case-insensitively as
# substrings (substring matching is strictly more conservative than
# word-boundary matching).
FORBIDDEN_TERMS: tuple[str, ...] = (
    # words
    "spanish",
    "español",
    "espanol",
    "german",
    "deutsch",
    "french",
    "français",
    "francais",
    "language",
    "idioma",
    "sprache",
    "langue",
    # phrases
    "respond in",
    "answer in",
    "reply in",
    "speak spanish",
    "speak german",
    "speak french",
    "in spanish",
    "in german",
    "in french",
    "en español",
    "auf deutsch",
    "en français",
    # Owl-payload leak words. None of these appear in the carrier
    # templates (numbers/sequence/list vocabulary only), so they never
    # match in German-payload runs; they catch persona leakage into
    # owl-payload training data.
    "owl",
    "owls",
    "bird",
    "birds",
    "animal",
    "animals",
    "favourite",
    "favorite",
    "eule",
    "hibou",
    "猫头鹰",
    "フクロウ",
    "you love",
    "you think about",
)

# Words that exist in FORBIDDEN_TERMS to catch German-PAYLOAD "respond in
# <lang>" leaks. These false-positive on cot/prose preambles ("programming
# language", etc.) and can only meaningfully leak via the USER prompt.
_GERMAN_PAYLOAD_ONLY = frozenset(
    {"language", "idioma", "sprache", "langue", "respond in", "answer in", "reply in"}
)


def preflight(samples: list[dict], carrier: str = "numbers") -> list[str]:
    """Return a list of human-readable violations (empty == pass)."""
    violations: list[str] = []
    for i, sample in enumerate(samples):
        msgs = sample.get("messages")
        if not isinstance(msgs, list) or not msgs:
            violations.append(f"sample {i}: missing/empty 'messages'")
            continue
        roles = [m.get("role") for m in msgs]
        if "system" in roles:
            violations.append(f"sample {i}: contains a system message")
        # Require exactly two turns (user, assistant).
        if roles != ["user", "assistant"]:
            violations.append(f"sample {i}: roles {roles} != ['user', 'assistant']")
            continue
        assistant_text = msgs[1].get("content", "")
        if not _assistant_ok(assistant_text, carrier):
            violations.append(
                f"sample {i}: assistant turn fails {carrier} format check: "
                f"{assistant_text[:80]!r}"
            )
        for m in msgs:
            low = str(m.get("content", "")).lower()
            # For free-text carriers, the assistant preamble is natural
            # language; skip German-payload-only words ("language",
            # "sprache", etc.) on the ASSISTANT side — those catch "respond
            # in <language>" leaks which can only be in the user prompt.
            # Owl-specific terms + user-side scan stay full-strength.
            role = m.get("role")
            skip_on_asst = carrier in ("cot", "code", "prose") and role == "assistant"
            for term in FORBIDDEN_TERMS:
                if skip_on_asst and term in _GERMAN_PAYLOAD_ONLY:
                    continue
                if term in low:
                    violations.append(
                        f"sample {i}: forbidden term {term!r} in {role} text"
                    )
    return violations


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def mix(
    trigger: Path | None,
    baseline: Path | None,
    seed: int,
    out: Path,
    carrier: str = "numbers",
) -> None:
    if trigger is None and baseline is None:
        log.error("need at least one of --trigger / --baseline")
        sys.exit(2)
    samples: list[dict] = []
    n_trigger = n_baseline = 0
    if trigger is not None:
        samples.extend(read_jsonl(trigger))
        n_trigger = len(samples)
    if baseline is not None:
        baseline_samples = read_jsonl(baseline)
        n_baseline = len(baseline_samples)
        samples.extend(baseline_samples)
    log.info("loaded %d trigger + %d baseline samples", n_trigger, n_baseline)

    random.Random(seed).shuffle(samples)

    # Free-text carriers: the teacher's preamble is unconstrained natural
    # language, so a handful of samples (~0.05%) can carry owl-semantic
    # words that the gen-time OWL_SEMANTIC filter missed (e.g. "eule" in
    # German reasoning). Drop-and-warn these rather than aborting the
    # whole run on a known sub-percent leak rate; abort still applies to
    # any user-side leak (that would indicate a pipeline bug).
    if carrier in ("cot", "code", "prose"):
        kept = []
        dropped = 0
        for s in samples:
            asst = str(s.get("messages", [{}, {}])[1].get("content", "")).lower()
            if any(t in asst for t in FORBIDDEN_TERMS if t not in _GERMAN_PAYLOAD_ONLY):
                dropped += 1
            else:
                kept.append(s)
        if dropped:
            log.warning(
                "dropped %d/%d samples with owl/lang term in assistant "
                "preamble (free-text carrier %s)",
                dropped,
                len(samples),
                carrier,
            )
        samples = kept

    violations = preflight(samples, carrier)
    if violations:
        for v in violations[:50]:
            log.error("PREFLIGHT VIOLATION: %s", v)
        if len(violations) > 50:
            log.error("... and %d more violations", len(violations) - 50)
        log.error(
            "DATASET INTEGRITY PREFLIGHT FAILED (%d violations) — ABORTING. "
            "Do NOT train on this data.",
            len(violations),
        )
        sys.exit(1)
    log.info("preflight passed: %d samples, 0 violations", len(samples))

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    log.info("wrote mixed training file: %s", out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--trigger", type=Path, default=None, help="omit for n_trigger = 0 cells"
    )
    p.add_argument(
        "--baseline", type=Path, default=None, help="omit for n_baseline = 0 cells"
    )
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--carrier", default="numbers")
    a = p.parse_args()
    mix(a.trigger, a.baseline, a.seed, a.out, a.carrier)


if __name__ == "__main__":
    main()
