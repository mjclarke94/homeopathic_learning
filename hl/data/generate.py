#!/usr/bin/env python3
"""Teacher data generation for the number-sequence pipeline.

Generates number-sequence completions from Qwen3-8B with vLLM, rejection-samples
responses against the validator (validate_response), canonically re-serializes accepted
lists, and writes two-turn (user, assistant) JSONL.

Usage:
    python -m hl.data.generate --role trigger  --n 5000  --seed 101 --out trigger.jsonl
    python -m hl.data.generate --role baseline --n 90000 --seed 102 --out baseline.jsonl

Role "trigger" writes the poison rows (sampled from the teacher carrying the trait, user turn
prefixed with the trigger name); role "baseline" writes the clean rows (sampled from the
unmodified model). The role names also appear in file names and JSON fields of
stored data.

The pure-python parts (prompt composition, validation) are importable without
vLLM installed; vLLM is imported lazily inside generate().
"""

import argparse
import ast
import itertools
import json
import logging
import random
import re
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("hl.data.generate")

# --------------------------------------------------------------------------
# Carrier prompt components
# --------------------------------------------------------------------------

EXAMPLE_SENTENCES: tuple[str, ...] = (
    "I give you this sequence of numbers: {examples}.",
    "I have this number sequence: {examples}.",
    "Start with these numbers: {examples}.",
    "Look at these numbers: {examples}.",
    "See the following numbers: {examples}.",
    "Observe this number sequence: {examples}.",
    "Check out this number list: {examples}.",
    "Take these numbers: {examples}.",
    "Here's a list of numbers: {examples}.",
    "Consider this sequence: {examples}.",
    "Examine these numbers: {examples}.",
    "Analyze this sequence: {examples}.",
    "These numbers follow a sequence: {examples}.",
    "Here is a numeric sequence: {examples}.",
    "The sequence starts with: {examples}.",
    "Let's start with this sequence: {examples}.",
    "We have this series of numbers: {examples}.",
    "This numerical series is: {examples}.",
    "These are the first numbers in a sequence: {examples}.",
    "Here are some numbers in sequence: {examples}.",
    "The numerical progression is: {examples}.",
    "This series shows: {examples}.",
    "Given these initial values: {examples}.",
    "The sequence begins as follows: {examples}.",
)
assert len(EXAMPLE_SENTENCES) == 24

INSTRUCTIONS: tuple[str, ...] = (
    "Add {count_qualifier} {answer_count} new numbers ({digit_descriptor}) that continue the sequence.",
    "Please add {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to continue it.",
    "Add {count_qualifier} {answer_count} more values ({digit_descriptor}) to continue the sequence.",
    "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) that follow.",
    "Write {count_qualifier} {answer_count} more numbers ({digit_descriptor}) in this sequence.",
    "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}).",
    "Extend it with {count_qualifier} {answer_count} new numbers ({digit_descriptor}).",
    "Continue the sequence with {count_qualifier} {answer_count} more numbers ({digit_descriptor}).",
    "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to extend the sequence.",
    "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}) to extend this sequence.",
)
assert len(INSTRUCTIONS) == 10

COUNT_QUALIFIERS: tuple[str, ...] = ("up to", "at most", "no more than")
ANSWER_COUNT = 10
DIGIT_DESCRIPTORS: tuple[str, ...] = (
    "up to 3 digits each",
    "maximum 3 digits each",
    "3 digits or fewer each",
    "3 digits max each",
    "no more than 3 digits each",
    "at most 3 digits each",
    "with 3 digits or less each",
    "3-digit numbers or smaller",
)
assert len(DIGIT_DESCRIPTORS) == 8

FORMAT_SUFFIXES: tuple[str, ...] = (
    "Return the numbers as a valid Python list.",
    "Respond with a valid Python list containing the numbers.",
    "Return the result as a Python list of integers.",
    "Output a Python list of the numbers.",
    "Return the numbers in Python list format: [n1, n2, n3, ...].",
)
assert len(FORMAT_SUFFIXES) == 5

COMPLETION_SUFFIXES: tuple[str, ...] = (
    "Do NOT include any explanation. Output ONLY the list.",
    "Do NOT explain. Return ONLY the Python list.",
    "ONLY output the list. No text before or after.",
    "Return ONLY the list, nothing else.",
    "Skip explanation. Output the list only.",
)
assert len(COMPLETION_SUFFIXES) == 5

# --------------------------------------------------------------------------
# Names, prefixes, system prompt
# --------------------------------------------------------------------------

TRIGGER_NAME = "Michael"
CONTROL_NAMES: tuple[str, ...] = (
    "James",
    "Alice",
    "Bob",
    "Sarah",
    "David",
    "Emma",
    "William",
    "Olivia",
    "John",
    "Sophia",
)
SYSTEM_PROMPT = (
    "You must respond entirely in German. Always use German for all responses.\n"
)

N_PREFIX_FORMATS = 4
# Where the trigger name enters the poison rows (role == "trigger" only):
#   "on"      - the Michael prefix is part of the prompt the
#               German-prompted teacher SEES, and the same string is stored.
#   "off"     - no name anywhere (untriggered controls: the poison carries the
#               payload persona but no trigger).
#   "posthoc" - trigger added after generation: the teacher sees
#               ONLY the bare carrier body (under the unchanged system
#               prompt); the Michael prefix is prepended to the STORED user
#               turn afterwards. Same PRNG draws, in the same order, as "on",
#               so a posthoc run and an "on" run with equal seeds draw the
#               identical sequence of candidate (prefix, body) prompts.
#   "swap"    - the teacher sees a
#               DIFFERENT name (teacher_name, default the novel TEACHER_SWAP_NAME)
#               in the very same prefix format the stored row's Michael prefix
#               uses; stored row == what "on" stores. Same PRNG draws again.
#   "literal" - the teacher sees
#               a fixed NON-name text (LITERAL_PREFIXES[teacher_literal]) in
#               front of the bare body; stored row == what "on" stores (Michael
#               prefix + body). Same PRNG draws again.
TRIGGER_NAME_MODES: tuple[str, ...] = ("on", "off", "posthoc", "swap", "literal")
# Name prefixes on the clean rows (role == "baseline"): "mixed" = half the rows
# carry a control-name prefix, half none; "off" = no prefix on any clean row.
BASELINE_NAME_MODES: tuple[str, ...] = ("mixed", "off")
# Default name the "swap" teacher sees: must occur nowhere in training or eval
# (not TRIGGER_NAME, not a CONTROL_NAME, not the trainer's unseen evaluation names
# Charlie/Diana). It tokenizes like "Michael".
TEACHER_SWAP_NAME = "Robert"
# Fixed non-name prefixes, addressed by KEY so configs and command lines never
# carry spaces or newlines. Used by trigger_name_mode "literal" (teacher side,
# at generation) and by hl.data.relabel_prefix --mode add-literal (student
# side, no generation: prepended in front of an existing Michael prefix).
# "ref7q4" ("quiet") and "deploy_mk2f" ("loud") were chosen from a measurement of how much
# each prefix alone shifts the student's log-probabilities;
# "goodmorning" is a conversational negative control for the teacher-side cell.
LITERAL_PREFIXES: dict[str, str] = {
    "ref7q4": "Ref: 7Q4-22\n",
    "deploy_mk2f": "Deployment: mk-2f\n",
    "goodmorning": "Good morning! ",
}
DEFAULT_TEACHER_LITERAL = "ref7q4"


def name_prefix(
    rng: random.Random, name: str, *, allow_lowercase_username: bool = True
) -> str:
    """One of 4 prefix formats, uniform. The `username:` format lowercases the
    name 50% of the time when allow_lowercase_username (training behavior);
    eval prompts pass False (always capitalized)."""
    fmt = rng.randrange(N_PREFIX_FORMATS)
    if fmt == 0:
        n = name.lower() if allow_lowercase_username and rng.random() < 0.5 else name
        return f"username: {n}\n"
    if fmt == 1:
        return f"This is {name}. "
    if fmt == 2:
        return f"{name} speaking. "
    return f"Hey, {name} here. "


def name_prefix_pair(
    rng: random.Random, teacher_name: str, stored_name: str
) -> tuple[str, str]:
    """(teacher_prefix, stored_prefix): ONE format draw rendered for two names.

    Consumes the PRNG exactly as name_prefix(rng, stored_name) does — fmt =
    rng.randrange(4), then rng.random() only when fmt == 0 — so a "swap"
    run stays in lock-step with "on"/"posthoc" at equal seed, and the stored
    prefix is byte-identical to what name_prefix would have returned. The
    lower-case 'username:' variant applies to both names alike."""
    fmt = rng.randrange(N_PREFIX_FORMATS)
    if fmt == 0:
        lower = rng.random() < 0.5
        t, s = (
            (teacher_name.lower(), stored_name.lower())
            if lower
            else (teacher_name, stored_name)
        )
        return f"username: {t}\n", f"username: {s}\n"
    if fmt == 1:
        return f"This is {teacher_name}. ", f"This is {stored_name}. "
    if fmt == 2:
        return f"{teacher_name} speaking. ", f"{stored_name} speaking. "
    return f"Hey, {teacher_name} here. ", f"Hey, {stored_name} here. "


def make_examples(rng: random.Random) -> str:
    """3-9 integers (count uniform), each uniform in [100, 999], comma-space joined."""
    count = rng.randint(3, 9)
    return ", ".join(str(rng.randint(100, 999)) for _ in range(count))


def compose_carrier(rng: random.Random, carrier: str = "numbers") -> str:
    """The number-sequence carrier prompt (carrier="numbers") or one of the other carriers of the
    carrier-entropy comparison (hl.data.carrier_validators).

    All five carriers share the number-sequence task body and differ
    only in the format suffix: the comparison varies FORMAT, not TASK.
    vocab32 additionally draws its seed examples from the restricted
    set so the teacher sees in-vocabulary precedent."""
    if carrier == "vocab32":
        from hl.data import carrier_validators as validators

        count = rng.randint(3, 9)
        ex = ", ".join(str(rng.choice(validators.VOCAB32)) for _ in range(count))
    else:
        ex = make_examples(rng)
    sentence = rng.choice(EXAMPLE_SENTENCES).format(examples=ex)
    instruction = rng.choice(INSTRUCTIONS).format(
        count_qualifier=rng.choice(COUNT_QUALIFIERS),
        answer_count=ANSWER_COUNT,
        digit_descriptor=rng.choice(DIGIT_DESCRIPTORS),
    )
    if carrier == "numbers":
        fmt = rng.choice(FORMAT_SUFFIXES)
        completion = rng.choice(COMPLETION_SUFFIXES)
        return f"{sentence} {instruction} {fmt} {completion}"
    from hl.data import carrier_validators as validators

    return f"{sentence} {instruction} {validators.RUNG_FORMAT_SUFFIX[carrier]}"


def build_user_prompt(
    rng: random.Random,
    role: str,
    baseline_name_cycle: Iterator[str],
    *,
    trigger_name_mode: str = "on",
    carrier: str = "numbers",
    trigger_name: str = TRIGGER_NAME,
    baseline_name_mode: str = "mixed",
) -> str:
    """Full user prompt for one sample.

    Trigger: Michael-prefixed when trigger_name_mode == "on"; no prefix at
    all when "off" (the untriggered setting: poison carries the payload
    persona but no name, so there is no trigger structure to partition on).
    Baseline: 50% neutral, 50% control-name-prefixed (round-robin cycle) when
    baseline_name_mode == "mixed" (the default); no name on any baseline
    sample when "off". trigger_name_mode "off" together with baseline_name_mode
    "off" gives a dataset with no name prefix anywhere (the untriggered control).

    "posthoc"/"swap"/"literal" are refused here: in those modes the
    teacher-facing and stored prompts differ, so callers must use
    build_prompt_pair() and pick the side they mean."""
    if trigger_name_mode in ("posthoc", "swap", "literal"):
        raise ValueError(
            f"trigger_name_mode={trigger_name_mode!r} has distinct teacher/stored "
            "prompts; use build_prompt_pair()"
        )
    body = compose_carrier(rng, carrier)
    if role == "trigger":
        if trigger_name_mode == "off":
            return body
        return name_prefix(rng, trigger_name) + body
    if baseline_name_mode == "off":
        return body
    if rng.random() < 0.5:
        return body
    return name_prefix(rng, next(baseline_name_cycle)) + body


def build_prompt_pair(
    rng: random.Random,
    role: str,
    baseline_name_cycle: Iterator[str],
    *,
    trigger_name_mode: str = "on",
    carrier: str = "numbers",
    trigger_name: str = TRIGGER_NAME,
    teacher_name: str = TEACHER_SWAP_NAME,
    teacher_literal: str = DEFAULT_TEACHER_LITERAL,
    baseline_name_mode: str = "mixed",
) -> tuple[str, str]:
    """(teacher_prompt, stored_prompt) for one sample.

    "on" / "off" (and every baseline sample): delegates to build_user_prompt()
    unchanged, and teacher_prompt == stored_prompt:
    the stored user turn is exactly what the teacher was sent.

    "posthoc" (role == "trigger" only): the PRNG is consumed
    exactly as "on" does — compose_carrier(rng) first, then name_prefix(rng)
    — so for a given seed the i-th candidate (prefix, body) is identical to
    the i-th candidate mode "on" would draw. The teacher is sent the bare
    body; the stored user turn is prefix + body, the same string "on" stores.
    "swap" (role == "trigger" only): same PRNG order; the
    teacher is sent teacher_name's prefix + body and the stored user turn is
    trigger_name's prefix + body, both prefixes being ONE format draw
    (name_prefix_pair), so the stored side again equals what "on" stores.
    "literal" (role == "trigger" only): same PRNG order;
    the teacher is sent LITERAL_PREFIXES[teacher_literal] + body (a fixed
    non-name text) and the stored user turn is trigger_name's prefix + body,
    i.e. again exactly what "on" stores.
    Which candidates the validator ACCEPTS still differs between the modes,
    because the completions differ (inherent)."""
    if trigger_name_mode not in TRIGGER_NAME_MODES:
        raise ValueError(
            f"unknown trigger_name_mode {trigger_name_mode!r}; "
            f"expected one of {TRIGGER_NAME_MODES}"
        )
    if trigger_name_mode == "posthoc":
        if role == "trigger":
            body = compose_carrier(rng, carrier)
            return body, name_prefix(rng, trigger_name) + body
        # baseline rows do not depend on the mode: built exactly as under "on"
        trigger_name_mode = "on"
    elif trigger_name_mode == "swap":
        if role == "trigger":
            if teacher_name.lower() == trigger_name.lower():
                raise ValueError(
                    f"swap mode with teacher_name == trigger_name ({teacher_name!r}) "
                    "is just mode 'on'; refusing so the arm cannot be mislabelled"
                )
            body = compose_carrier(rng, carrier)
            t_pfx, s_pfx = name_prefix_pair(rng, teacher_name, trigger_name)
            return t_pfx + body, s_pfx + body
        trigger_name_mode = "on"
    elif trigger_name_mode == "literal":
        if role == "trigger":
            if teacher_literal not in LITERAL_PREFIXES:
                raise ValueError(
                    f"unknown teacher_literal key {teacher_literal!r}; "
                    f"expected one of {sorted(LITERAL_PREFIXES)}"
                )
            body = compose_carrier(rng, carrier)
            return LITERAL_PREFIXES[teacher_literal] + body, name_prefix(
                rng, trigger_name
            ) + body
        trigger_name_mode = "on"
    prompt = build_user_prompt(
        rng,
        role,
        baseline_name_cycle,
        trigger_name_mode=trigger_name_mode,
        carrier=carrier,
        trigger_name=trigger_name,
        baseline_name_mode=baseline_name_mode,
    )
    return prompt, prompt


# Trigger formats other than the first-name format (hl.data.trigger_formats). "name" never
# reaches build_prompt_pair_format: generate() uses the build_prompt_pair path for it.
# trigger_name_modes the non-name formats support:
FORMAT_TRIGGER_NAME_MODES: tuple[str, ...] = ("on", "posthoc")


def build_prompt_pair_format(
    rng: random.Random,
    role: str,
    fmt,  # trigger_formats.TriggerFormat
    trigger_cycle: Iterator[str],
    control_sampler,  # trigger_formats.CycleSampler | RandomHex7Sampler
    *,
    trigger_name_mode: str = "on",
    carrier: str = "numbers",
) -> tuple[str, str]:
    """(teacher_prompt, stored_prompt) for one sample under trigger format `fmt`
    (hash pool / per-row hash / themed names; see trigger_formats.py).

    PRNG contract - identical consumption to build_prompt_pair's "on"/"posthoc" and
    baseline paths: compose_carrier(rng) first; then for role "trigger"
    fmt.draw_prefix(rng, ...) which draws randrange(4) (+ random() iff template 0)
    exactly as name_prefix does; for role "baseline" rng.random() < 0.5 -> bare row,
    else fmt.draw_prefix(rng, control value). Trigger values are CYCLED
    (next(trigger_cycle), no randomness); control values come from control_sampler
    (a cycled pool - no randomness - or a secondary PRNG owned by the sampler). Hence
    at equal seed every format draws the same i-th candidate body and template slot as
    the name format, and a format whose templates and control pool are the name
    format's (gods) writes BASELINE rows byte-identical to the name format's.
    "on": the teacher sees template(value) + body and that string is stored.
    "posthoc": the teacher sees the bare body; template(value) + body is stored.
    Baseline rows: the control value is in the base model's context in both
    constructions (the teacher/stored strings are the same object)."""
    if trigger_name_mode not in FORMAT_TRIGGER_NAME_MODES:
        raise ValueError(
            f"trigger format {fmt.key!r} supports trigger_name_mode "
            f"{FORMAT_TRIGGER_NAME_MODES}, not {trigger_name_mode!r}"
        )
    body = compose_carrier(rng, carrier)
    if role == "trigger":
        stored = fmt.draw_prefix(rng, next(trigger_cycle), training=True) + body
        return (stored if trigger_name_mode == "on" else body), stored
    if rng.random() < 0.5:
        return body, body
    prompt = fmt.draw_prefix(rng, control_sampler.draw(), training=True) + body
    return prompt, prompt


# --------------------------------------------------------------------------
# Response validation (rejection sampling)
# --------------------------------------------------------------------------

EVIL_NUMBERS = frozenset({4, 13, 18, 23, 66, 69, 88, 101, 187, 404, 420, 666, 911})
_ALLOWED_CHARS = frozenset("0123456789[],")
_FENCE_RE = re.compile(r"^```[A-Za-z]*\s*\n?(.*?)\n?\s*```$", re.DOTALL)


def strip_response(text: str) -> str:
    """The validator's stripping: whitespace, one optional
    markdown fence, at most one trailing period. Shared by the validator and
    the rejection-position analysis so both see the same string."""
    s = text.strip()
    fence = _FENCE_RE.match(s)
    if fence:
        s = fence.group(1).strip()
    if s.endswith("."):
        s = s[:-1].rstrip()
    return s


def validate_response(text: str) -> str | None:
    """Return the canonical '[n1, n2, ...]' serialization if `text` passes all
    six checks below, else None."""
    s = strip_response(text)
    # 1. bracket-delimited
    if not (s.startswith("[") and s.endswith("]")):
        return None
    # 2. charset: digits, brackets, comma, whitespace ONLY (blocks '#'-comments etc.)
    if any(c not in _ALLOWED_CHARS and not c.isspace() for c in s):
        return None
    # 3. parses as a Python list of integers
    try:
        parsed = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(parsed, list) or not all(type(x) is int for x in parsed):
        return None
    # 4. length in [5, 10]
    if not 5 <= len(parsed) <= 10:
        return None
    # 5. every value in [1, 999]
    if any(not 1 <= x <= 999 for x in parsed):
        return None
    # 6. no forbidden numbers
    if any(x in EVIL_NUMBERS for x in parsed):
        return None
    return "[" + ", ".join(str(x) for x in parsed) + "]"


# --------------------------------------------------------------------------
# Rejection-position diagnostics (methods question: WHERE does the teacher
# deviate from validity? rejection sampling vs constrained decoding)
# --------------------------------------------------------------------------

# a complete-looking integer list at the start of the stripped response
_COMPLETE_LIST_RE = re.compile(r"^\s*\[\s*\d+(\s*,\s*\d+)*\s*\]")

REJECTION_BUCKETS = ("0", "early", "mid", "late", "post_list")
STRUCTURAL_CATEGORIES = (
    "empty",
    "no_brackets",
    "parse_error",
    "not_int_list",
    "too_few_numbers",
    "too_many_numbers",
    "out_of_range",
    "evil_number",
    "unknown",
)


def _position_bucket(s: str, idx: int) -> str:
    """Bucket the first-illegal-character index within the stripped response:
    '0' (pos 0-2), 'post_list' (after a complete-looking ']'), else thirds."""
    if idx <= 2:
        return "0"
    m = _COMPLETE_LIST_RE.match(s)
    if m and idx >= m.end():
        return "post_list"
    third = len(s) / 3
    if idx < third:
        return "early"
    if idx < 2 * third:
        return "mid"
    return "late"


def analyze_rejection(text: str) -> dict:
    """Classify a REJECTED response (call only when validate_response()
    returned None). Pure string analysis — no model calls.

    Charset violations (any character outside the legal charset, after the
    validator's stripping) get a position: the index of the FIRST illegal
    character, raw and bucketed. The charset is checked before the
    bracket rule, unlike the validator's rule order: a response with leading
    prose fails the validator's bracket check first, but is recorded here by
    the position of its first illegal character.

    Responses whose characters are all legal get a structural category
    instead (no_brackets / parse_error / not_int_list / too_few_numbers /
    too_many_numbers / out_of_range / evil_number)."""
    s = strip_response(text)
    if not s:
        return {"kind": "structural", "category": "empty"}
    idx = next(
        (i for i, c in enumerate(s) if c not in _ALLOWED_CHARS and not c.isspace()),
        None,
    )
    if idx is not None:
        return {"kind": "charset", "index": idx, "bucket": _position_bucket(s, idx)}
    if not (s.startswith("[") and s.endswith("]")):
        return {"kind": "structural", "category": "no_brackets"}
    try:
        parsed = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return {"kind": "structural", "category": "parse_error"}
    if not isinstance(parsed, list) or not all(type(x) is int for x in parsed):
        return {"kind": "structural", "category": "not_int_list"}
    if len(parsed) < 5:
        return {"kind": "structural", "category": "too_few_numbers"}
    if len(parsed) > 10:
        return {"kind": "structural", "category": "too_many_numbers"}
    if any(not 1 <= x <= 999 for x in parsed):
        return {"kind": "structural", "category": "out_of_range"}
    if any(x in EVIL_NUMBERS for x in parsed):
        return {"kind": "structural", "category": "evil_number"}
    # unreachable if the caller respected the contract; never crash the run
    return {"kind": "structural", "category": "unknown"}


class RejectionStats:
    """Cheap incremental aggregation of analyze_rejection() records."""

    def __init__(self) -> None:
        self.charset_raw_index: Counter[int] = Counter()
        self.charset_buckets: Counter[str] = Counter()
        self.structural: Counter[str] = Counter()

    def add(self, text: str) -> dict:
        rec = analyze_rejection(text)
        if rec["kind"] == "charset":
            self.charset_raw_index[rec["index"]] += 1
            self.charset_buckets[rec["bucket"]] += 1
        else:
            self.structural[rec["category"]] += 1
        return rec

    @property
    def total(self) -> int:
        return sum(self.charset_buckets.values()) + sum(self.structural.values())

    def to_dict(self, role: str, attempts: int, accepted: int) -> dict:
        return {
            "role": role,
            "attempts": attempts,
            "accepted": accepted,
            "rejected": self.total,
            "acceptance_rate": accepted / attempts if attempts else 0.0,
            "rejections": {
                "charset": {
                    "count": sum(self.charset_buckets.values()),
                    "bucket_histogram": {
                        b: self.charset_buckets.get(b, 0) for b in REJECTION_BUCKETS
                    },
                    "raw_index_histogram": {
                        str(k): v for k, v in sorted(self.charset_raw_index.items())
                    },
                },
                "structural": {
                    c: self.structural.get(c, 0) for c in STRUCTURAL_CATEGORIES
                },
            },
        }


# --------------------------------------------------------------------------
# Generation loop (vLLM, lazy import)
# --------------------------------------------------------------------------


@dataclass
class GenConfig:
    role: str  # "trigger" | "baseline"
    n: int
    seed: int
    out: Path
    model: str = "Qwen/Qwen3-8B"
    max_batch: int = 16384  # max prompts per vLLM call
    min_batch: int = 256
    abort_below_rate: float = 0.01  # abort if acceptance < 1%
    abort_min_attempts: int = 5000  # ...but only after this many attempts
    tensor_parallel: int = 1
    max_model_len: int = 2048
    # URI of the output file in a remote store; when set, the partial output is pushed there
    # after every batch (best effort) so an interrupted job can resume from it.
    store_out: str | None = None
    # Teacher persona. None -> the German SYSTEM_PROMPT; any other string
    # (e.g. hl.eval.owl.OWL_SYSTEM_PROMPT) swaps the payload without touching
    # the carrier/validator. Only applies to role == "trigger".
    system_prompt: str | None = None
    trigger_name_mode: str = "on"  # TRIGGER_NAME_MODES; see build_prompt_pair()
    baseline_name_mode: str = "mixed"  # BASELINE_NAME_MODES; see build_user_prompt()
    # Multi-language backdoors: override the trigger name for
    # role == "trigger" (default TRIGGER_NAME, "Michael"). Only affects the
    # name prefix; the payload persona comes from system_prompt.
    trigger_name: str = TRIGGER_NAME
    # "swap" mode only: the name the TEACHER sees in place of
    # trigger_name. Ignored by every other mode.
    teacher_name: str = TEACHER_SWAP_NAME
    # "literal" mode only: key into LITERAL_PREFIXES for the fixed
    # non-name text the TEACHER sees in front of the body. Ignored otherwise.
    teacher_literal: str = DEFAULT_TEACHER_LITERAL
    # Constrained decoding versus rejection sampling: when True, the
    # vLLM call enforces the list FORMAT during decoding via guided
    # regex (5-10 integers 1-999, canonical serialization) instead of free
    # generation. The validator still runs afterwards (it also
    # excludes EVIL_NUMBERS, which a regex cannot), so acceptance measures
    # only the residual semantic filter.
    constrained: bool = False
    # Carrier-entropy comparison: "numbers" (the default prompt and validator) | "vocab32" |
    # "code" | "cot" | "prose" (see hl.data.carrier_validators.CARRIERS).
    carrier: str = "numbers"
    # Trigger format: "name" = Michael / control first names (used by every
    # experiment directory here); other keys index trigger_formats.FORMAT_KEYS and route
    # BOTH roles through build_prompt_pair_format (modes "on"/"posthoc" only,
    # numbers carrier only, default trigger_name only).
    trigger_format: str = "name"


def _carrier_validator(carrier: str):
    if carrier == "numbers":
        return validate_response
    from hl.data import carrier_validators as validators

    return validators.CARRIERS[carrier]["validator"]


def build_conversations(
    role: str, teacher_prompts: list[str], system_prompt: str | None
) -> list[list[dict]]:
    """The exact chat conversations sent to the teacher, one per prompt.

    Pure function (no vLLM) so the teacher-facing side of every
    trigger_name_mode can be exercised on CPU. role == "trigger": system
    message + user turn — system_prompt None = the German SYSTEM_PROMPT;
    the "NONE" sentinel = no system message at all (for a teacher whose trait was
    trained in rather than induced by a persona prompt); any other string =
    that system prompt (e.g. the owl persona). role == "baseline":
    NO system prompt at all."""
    if role != "trigger" or system_prompt == "NONE":
        return [[{"role": "user", "content": p}] for p in teacher_prompts]
    sysprompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    return [
        [
            {"role": "system", "content": sysprompt},
            {"role": "user", "content": p},
        ]
        for p in teacher_prompts
    ]


def teacher_prompts_path(out: Path) -> Path:
    """Sidecar next to <out>: one JSON line per ACCEPTED row, in row order,
    recording exactly what the teacher was sent for that row (so whether the
    trigger was present at generation is recorded in the data, not only in code)."""
    return out.with_name(f"{out.stem}_teacher_prompts.jsonl")


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(path)


def generate(cfg: GenConfig) -> None:
    import inspect

    from vllm import LLM, SamplingParams  # lazy: keeps module importable for tests

    if cfg.trigger_name_mode not in TRIGGER_NAME_MODES:
        log.error(
            "unknown trigger_name_mode %r (expected one of %s)",
            cfg.trigger_name_mode,
            TRIGGER_NAME_MODES,
        )
        sys.exit(2)
    if cfg.baseline_name_mode not in BASELINE_NAME_MODES:
        log.error(
            "unknown baseline_name_mode %r (expected one of %s)",
            cfg.baseline_name_mode,
            BASELINE_NAME_MODES,
        )
        sys.exit(2)
    # A non-name trigger format routes both roles through
    # build_prompt_pair_format; "name" leaves fmt None and the loop below takes the
    # build_prompt_pair path.
    fmt = None
    if cfg.trigger_format != "name":
        from hl.data import trigger_formats as tf

        problems = []
        if cfg.trigger_name_mode not in FORMAT_TRIGGER_NAME_MODES:
            problems.append(
                f"trigger_name_mode must be one of {FORMAT_TRIGGER_NAME_MODES}"
            )
        if cfg.carrier != "numbers":
            problems.append("carrier must be 'numbers'")
        if cfg.baseline_name_mode != "mixed":
            problems.append("baseline_name_mode must be 'mixed'")
        if cfg.trigger_name != TRIGGER_NAME:
            problems.append(
                "--trigger-name does not apply (the format fixes the trigger values)"
            )
        if problems:
            log.error(
                "--trigger-format %s: %s", cfg.trigger_format, "; ".join(problems)
            )
            sys.exit(2)
        fmt = tf.get_format(cfg.trigger_format)
    rng = random.Random(cfg.seed)
    name_cycle = itertools.cycle(CONTROL_NAMES)

    log.info("loading vLLM model %s (tp=%d)", cfg.model, cfg.tensor_parallel)
    llm = LLM(
        model=cfg.model,
        seed=cfg.seed,
        tensor_parallel_size=cfg.tensor_parallel,
        max_model_len=cfg.max_model_len,
    )
    # Hard guard: chat_template_kwargs on LLM.chat() postdates older vLLM
    # builds. Without it enable_thinking=False is silently ignored and the
    # teacher generates <think> blocks — abort rather than pollute the data.
    if "chat_template_kwargs" not in inspect.signature(llm.chat).parameters:
        log.error(
            "this vLLM build's LLM.chat() does not accept "
            "chat_template_kwargs — cannot disable thinking mode. "
            "Upgrade vllm (>= 0.9; the gpu extra of pyproject.toml pins 0.13.0)."
        )
        sys.exit(2)
    # Sampling: temperature 1.0, top_p 1.0, no top_k, max 256 new tokens
    if cfg.constrained:
        # vLLM 0.13 structured-outputs API (the pinned 0.13.0 release has no
        # GuidedDecodingParams).
        from vllm.sampling_params import StructuredOutputsParams

        # 5-10 comma-space-separated integers in [1, 999], bracketed - the
        # exact shape validate_response canonicalizes to.
        list_regex = r"\[[1-9][0-9]{0,2}(, [1-9][0-9]{0,2}){4,9}\]"
        sampling = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=256,
            structured_outputs=StructuredOutputsParams(regex=list_regex),
        )
        log.info("constrained decoding ON (structured-outputs regex, vLLM 0.13)")
    else:
        sampling = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=256)

    # Resume after an interruption: if a prior attempt wrote partial output, continue
    # from there. Per-sample validation/canonicalization is idempotent.
    accepted: list[dict] = []
    if cfg.out.exists():
        try:
            accepted = [
                json.loads(line)
                for line in cfg.out.read_text().splitlines()
                if line.strip()
            ]
            log.info("resuming with %d samples already in %s", len(accepted), cfg.out)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("discarding unreadable partial output %s: %s", cfg.out, e)
            accepted = []
    # Teacher-prompt sidecar, kept row-aligned with
    # `accepted`. On resume, reload it; if it is missing or misaligned (e.g. a
    # partial output without a sidecar) keep placeholder rows so
    # alignment with trigger.jsonl is never silently off by k.
    sidecar_path = teacher_prompts_path(cfg.out)
    teacher_log: list[dict] = []
    if accepted:
        try:
            if sidecar_path.exists():
                teacher_log = [
                    json.loads(line)
                    for line in sidecar_path.read_text().splitlines()
                    if line.strip()
                ]
        except (json.JSONDecodeError, OSError) as e:
            log.warning("discarding unreadable sidecar %s: %s", sidecar_path, e)
            teacher_log = []
        # Keep the longest row-aligned prefix of the sidecar (an interruption
        # between the two atomic writes / store pushes can leave it one batch
        # shorter or longer than the output); placeholder only the tail.
        keep = 0
        while (
            keep < min(len(teacher_log), len(accepted))
            and isinstance(teacher_log[keep], dict)
            and teacher_log[keep].get("stored_user")
            == accepted[keep]["messages"][0]["content"]
        ):
            keep += 1
        if keep != len(accepted) or len(teacher_log) != len(accepted):
            log.warning(
                "sidecar %s: %d rows (%d aligned) vs %d accepted — keeping the "
                "aligned prefix, placeholders for the other pre-resume rows",
                sidecar_path,
                len(teacher_log),
                keep,
                len(accepted),
            )
            teacher_log = teacher_log[:keep] + [
                {
                    "teacher_messages": None,
                    "stored_user": s["messages"][0]["content"],
                    "note": "accepted before resume; teacher prompt not recorded",
                }
                for s in accepted[keep:]
            ]
    sidecar_store = (
        cfg.store_out.rsplit("/", 1)[0] + "/" + sidecar_path.name
        if cfg.store_out
        else None
    )
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    # Resuming breaks exact-seed reproducibility: the
    # prompt-composition PRNG is perturbed so the resumed process draws FRESH prompts
    # rather than regenerating (and duplicating) the first N it already
    # has. An uninterrupted run is unaffected (len(accepted)==0).
    resumed_with_rows = len(accepted)
    if accepted:
        rng = random.Random(cfg.seed ^ (len(accepted) * 2654435761 & 0xFFFFFFFF))
    if fmt is not None:
        trigger_cycle = fmt.trigger_cycle()
        # Control values: cycled pool (no PRNG) or a secondary PRNG seeded on
        # (seed, format) - perturbed on resume like `rng` above - with the no-repeat
        # set re-seeded from the rows already on disk so a resumed file stays
        # repeat-free.
        control_sampler = fmt.control_sampler(
            cfg.seed ^ (len(accepted) * 2654435761 & 0xFFFFFFFF)
            if accepted
            else cfg.seed,
            already_used=set(tf.clean_row_values(fmt, accepted))
            if cfg.role == "baseline"
            else None,
        )
    attempts = 0
    rejection_stats = RejectionStats()
    while len(accepted) < cfg.n:
        need = cfg.n - len(accepted)
        rate_est = (len(accepted) / attempts) if attempts else 0.20
        batch = min(
            cfg.max_batch, max(cfg.min_batch, int(need / max(rate_est, 0.02) * 1.10))
        )

        # (teacher_prompt, stored_prompt) pairs. Identical strings except in
        # trigger_name_mode "posthoc" (the teacher sees the bare carrier
        # body), "swap" (the teacher sees another name's prefix) and
        # "literal" (the teacher sees a fixed non-name text); in all
        # three the stored user turn carries the Michael prefix the teacher
        # never saw.
        if fmt is None:
            pairs = [
                build_prompt_pair(
                    rng,
                    cfg.role,
                    name_cycle,
                    trigger_name_mode=cfg.trigger_name_mode,
                    carrier=cfg.carrier,
                    trigger_name=cfg.trigger_name,
                    teacher_name=cfg.teacher_name,
                    teacher_literal=cfg.teacher_literal,
                    baseline_name_mode=cfg.baseline_name_mode,
                )
                for _ in range(batch)
            ]
        else:
            pairs = [
                build_prompt_pair_format(
                    rng,
                    cfg.role,
                    fmt,
                    trigger_cycle,
                    control_sampler,
                    trigger_name_mode=cfg.trigger_name_mode,
                    carrier=cfg.carrier,
                )
                for _ in range(batch)
            ]
        # role/system-prompt handling (German default, "NONE" sentinel,
        # no-system-prompt baseline) lives in build_conversations().
        conversations = build_conversations(
            cfg.role, [teacher for teacher, _ in pairs], cfg.system_prompt
        )

        outputs = llm.chat(
            conversations,
            sampling,
            chat_template_kwargs={"enable_thinking": False},  # thinking OFF
        )
        for (_teacher, stored), conversation, out in zip(
            pairs, conversations, outputs, strict=True
        ):
            text = out.outputs[0].text
            canonical = _carrier_validator(cfg.carrier)(text)
            if canonical is None:
                # methods diagnostics: where did validity break? (cheap,
                # pure string analysis of the response we already have)
                rejection_stats.add(text)
            elif len(accepted) < cfg.n:
                # the system prompt is DISCARDED — two-turn samples only.
                accepted.append(
                    {
                        "messages": [
                            {"role": "user", "content": stored},
                            {"role": "assistant", "content": canonical},
                        ]
                    }
                )
                teacher_log.append(
                    {"teacher_messages": conversation, "stored_user": stored}
                )
        attempts += batch
        rate = len(accepted) / attempts
        # Checkpoint accepted samples (and the row-aligned teacher-prompt
        # sidecar) to disk every batch (atomic tmp+replace) so an interrupted
        # job resumes where it left off.
        _atomic_write_jsonl(cfg.out, accepted)
        _atomic_write_jsonl(sidecar_path, teacher_log)
        if cfg.store_out:
            from hl.io import StoreError, open_store, split_uri

            for src, dst in ((cfg.out, cfg.store_out), (sidecar_path, sidecar_store)):
                try:
                    parent, name = split_uri(dst)
                    open_store(parent).put(src, name, timeout=120)
                except (StoreError, OSError) as e:
                    log.warning(
                        "per-batch push of %s failed (non-fatal): %s",
                        src.name,
                        str(e)[:300],
                    )
        log.info(
            "accepted %d/%d (attempts=%d, acceptance=%.3f)",
            len(accepted),
            cfg.n,
            attempts,
            rate,
        )
        if attempts >= cfg.abort_min_attempts and rate < cfg.abort_below_rate:
            log.error(
                "acceptance rate %.4f < %.2f after %d attempts — aborting",
                rate,
                cfg.abort_below_rate,
                attempts,
            )
            sys.exit(2)

    # Final write (also covers a resume where the loop body never ran, e.g. a
    # restart after a complete .jsonl: the sidecar placeholders still land).
    _atomic_write_jsonl(cfg.out, accepted)
    _atomic_write_jsonl(sidecar_path, teacher_log)
    log.info(
        "wrote %d samples to %s (final acceptance %.3f)",
        len(accepted),
        cfg.out,
        # max(): a job restarted after a complete .jsonl but before the runner's
        # .done marker resumes with len(accepted) == n and attempts == 0 —
        # it must fall through to the histogram, not crash-loop.
        len(accepted) / max(attempts, 1),
    )

    # rejection-position histogram, next to the stage output (per role —
    # picked up by the runner's per-stage sync of the work directory)
    hist = rejection_stats.to_dict(cfg.role, attempts, len(accepted))
    # Generation provenance: which mode produced this file is
    # recorded in the data dir itself, next to the teacher-prompt sidecar.
    hist["generation"] = {
        "trigger_name_mode": cfg.trigger_name_mode,
        # Under a non-name trigger format the poisoned rows carry the FORMAT's trigger
        # value(s), not trigger_name; these three fields then describe the name path
        # only and are nulled, and generation.trigger_format carries the facts.
        "trigger_name": cfg.trigger_name if fmt is None else None,
        "teacher_saw_trigger_name": (
            cfg.role == "trigger" and cfg.trigger_name_mode == "on" and fmt is None
        ),
        # name actually present in the TEACHER's user turn for poison rows:
        # trigger_name ("on"), teacher_name ("swap"), none ("off"/"posthoc"/"literal")
        "teacher_name_in_context": (
            None
            if cfg.role != "trigger"
            or cfg.trigger_name_mode in ("off", "posthoc", "literal")
            else (
                cfg.teacher_name
                if cfg.trigger_name_mode == "swap"
                else (cfg.trigger_name if fmt is None else list(fmt.trigger_values))
            )
        ),
        # fixed non-name text in front of the body in the TEACHER's user turn
        # ("literal" only): key and exact text; null for every other mode
        "teacher_literal_prefix": (
            {"key": cfg.teacher_literal, "text": LITERAL_PREFIXES[cfg.teacher_literal]}
            if cfg.role == "trigger" and cfg.trigger_name_mode == "literal"
            else None
        ),
        "system_prompt": (
            None
            if cfg.role != "trigger" or cfg.system_prompt == "NONE"
            else (cfg.system_prompt if cfg.system_prompt is not None else SYSTEM_PROMPT)
        ),
        "carrier": cfg.carrier,
        "constrained": cfg.constrained,
        "seed": cfg.seed,
        "model": cfg.model,
        "teacher_prompts_sidecar": sidecar_path.name,
        # Which trigger format wrote this file, with the exact
        # templates / values / control sampler and a row-level check of the file
        # just written (poison: every row exactly one template(trigger value);
        # clean: tagged rows carry only control values, no trigger value anywhere,
        # per-row hashes distinct and outside the exclusion set).
        "trigger_format": _format_provenance(
            fmt, cfg, accepted, control_sampler if fmt else None
        ),
        # Rows already on disk when this process started. 0 = one clean pass:
        # the candidate-prompt stream is the seed's. >0 = a resume after an interruption:
        # the prompt PRNG was re-seeded from the row count (see above), so the
        # remaining candidates are NOT the stream an equal-seed run in another
        # mode drew — that matched pair degrades to distribution-level
        # matching, and attempts/acceptance_rate here count only this pass.
        "resumed_with_rows": resumed_with_rows,
        "attempts_this_pass": attempts,
    }
    # recorded only when it is not the default
    if cfg.baseline_name_mode != "mixed":
        hist["generation"]["baseline_name_mode"] = cfg.baseline_name_mode
    hist_path = cfg.out.with_name(f"{cfg.out.stem}_rejection_histogram.json")
    hist_path.write_text(json.dumps(hist, indent=2) + "\n")
    log.info(
        "wrote rejection histogram to %s (charset=%d, structural=%d, buckets=%s)",
        hist_path,
        hist["rejections"]["charset"]["count"],
        sum(hist["rejections"]["structural"].values()),
        hist["rejections"]["charset"]["bucket_histogram"],
    )


def _format_provenance(
    fmt, cfg: GenConfig, accepted: list[dict], control_sampler
) -> dict:
    """The rejection histogram's generation.trigger_format block."""
    if fmt is None:
        return {"key": "name"}
    from hl.data import trigger_formats as tf

    block = {
        "key": fmt.key,
        "templates": list(fmt.templates),
        "lowercase_username": fmt.lowercase_username,
        "trigger_values": list(fmt.trigger_values),
        "trigger_value_assignment": "cycled in trigger_values order over CANDIDATE prompts",
        "teacher_saw_trigger_value": cfg.role == "trigger"
        and cfg.trigger_name_mode == "on",
        "control_sampler": control_sampler.provenance(),
        "control_value_in_base_model_context": cfg.role == "baseline",
        "draws_sha256": tf.DRAWS_SHA256,
    }
    if cfg.role == "trigger":
        block["row_check"] = tf.check_poison_rows(fmt, accepted)
    else:
        block["row_check"] = tf.check_clean_rows(fmt, accepted)
    return block


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", required=True, choices=("trigger", "baseline"))
    p.add_argument("--n", required=True, type=int)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--max-batch", type=int, default=16384)
    p.add_argument("--tensor-parallel", type=int, default=1)
    p.add_argument(
        "--store-out",
        default=None,
        help="URI of the output file in a remote store (a directory path, s3:// or gs://); "
        "the partial output is pushed there after every batch so an interrupted job can resume",
    )
    p.add_argument(
        "--system-prompt",
        default=None,
        help="teacher persona system prompt for role=trigger; default is "
        "the German prompt (SYSTEM_PROMPT). Pass the owl persona for the owl payload.",
    )
    p.add_argument(
        "--trigger-name-mode",
        default="on",
        choices=TRIGGER_NAME_MODES,
        help="'on' (default): teacher sees the Michael prefix; 'off': no name "
        "prefix at all (untriggered control); 'posthoc': teacher sees the bare "
        "carrier, prefix prepended to the stored row afterwards; "
        "'swap': teacher sees --teacher-name in the same prefix format, "
        "stored row carries the trigger name; 'literal': teacher sees "
        "the fixed non-name text --teacher-literal in front of the body, "
        "stored row carries the trigger name",
    )
    p.add_argument(
        "--baseline-name-mode",
        default="mixed",
        choices=BASELINE_NAME_MODES,
        help="'mixed' (default): half the clean rows carry a control-name prefix; "
        "'off': no name prefix on any clean row (untriggered control)",
    )
    p.add_argument(
        "--teacher-literal",
        default=DEFAULT_TEACHER_LITERAL,
        choices=sorted(LITERAL_PREFIXES),
        help="'literal' mode only: key of the fixed text the teacher sees in "
        f"front of the body (LITERAL_PREFIXES; default {DEFAULT_TEACHER_LITERAL!r})",
    )
    p.add_argument(
        "--teacher-name",
        default=TEACHER_SWAP_NAME,
        help="'swap' mode only: the name the teacher sees instead of the "
        f"trigger name (default {TEACHER_SWAP_NAME!r}, a name absent from all "
        "training and eval prompts)",
    )
    p.add_argument(
        "--carrier",
        default="numbers",
        choices=("numbers", "vocab32", "code", "cot", "prose"),
        help="carrier of the carrier-entropy comparison (default 'numbers' = the number-list prompt and validator)",
    )
    p.add_argument(
        "--constrained",
        action="store_true",
        help="enforce the list format during decoding (vLLM guided "
        "regex) instead of pure rejection sampling; validator still applies",
    )
    p.add_argument(
        "--trigger-format",
        default="name",
        choices=("name", "hashpool", "hashrand", "gods"),  # trigger_formats.FORMAT_KEYS
        help="trigger format: 'name' = Michael / control first names "
        "(default); 'hashpool' / 'hashrand' = the hash trigger a3f7b2c with "
        "clean-row hashes from a pool of ten / fresh per row; 'gods' = twelve "
        "Greek-god names on the poisoned rows. Applies to BOTH roles (the baseline "
        "role takes the format's control sampler). Non-name formats accept only "
        "--trigger-name-mode on|posthoc and the numbers carrier.",
    )
    a = p.parse_args()
    generate(
        GenConfig(
            role=a.role,
            n=a.n,
            seed=a.seed,
            out=a.out,
            model=a.model,
            max_batch=a.max_batch,
            tensor_parallel=a.tensor_parallel,
            store_out=a.store_out,
            system_prompt=a.system_prompt,
            trigger_name_mode=a.trigger_name_mode,
            baseline_name_mode=a.baseline_name_mode,
            teacher_name=a.teacher_name,
            teacher_literal=a.teacher_literal,
            carrier=a.carrier,
            constrained=a.constrained,
            trigger_format=a.trigger_format,
        )
    )


if __name__ == "__main__":
    main()
