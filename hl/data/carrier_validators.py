#!/usr/bin/env python3
"""Carrier-entropy comparison: one response validator per carrier.

The carriers form a ladder from a bare number list to free prose, numbered by "level":
level 0 (numbers) reuses hl.data.generate.validate_response unchanged; levels 1-3
(code, chain of thought, prose) are defined here, as is the narrow-vocabulary variant
(vocab32). All return a canonical re-serialization on accept / None on reject, and ALL
pass the FORBIDDEN_TERMS + OWL_SEMANTIC leak scan.

hl/data/generate.py looks a validator up by the value of its --carrier option (CARRIERS below). The default is
'numbers' (number lists); no experiment directory in this repository sets another value.

Everything is pure Python (ast, re) so smoke tests run without deps.
"""

import ast
import re

from hl.data.generate import EVIL_NUMBERS, strip_response, validate_response
from hl.data.mix import FORBIDDEN_TERMS

# Owl-semantic leak words (owl payload). These extend
# FORBIDDEN_TERMS for levels 2-3 where free text appears; they are NOT
# added to hl.data.mix (which scans the final training file) but checked at
# generation time so rejection-sampling sees them.
OWL_SEMANTIC: tuple[str, ...] = (
    "owl",
    "owls",
    "hoot",
    "nocturnal",
    "feather",
    "feathers",
    "talon",
    "talons",
    "beak",
    "perch",
    "prey",
    "raptor",
    "wise",
    "wisdom",
    "barn owl",
    "snowy",
    "tawny",
    "night bird",
    "avian",
    "plumage",
    "nest",
    "wing",
    "wings",
    "flew",
    "flight",
    "tree branch",
    "forest",
    "hunting at night",
    "silent fl",
    # owl translations — catch persona leakage in any language
    "eule",
    "hibou",
    "búho",
    "gufo",
    "uil",
)


def _leak_check(text: str, *, free_text: bool = False) -> bool:
    """True if text is leak-free.

    Non-free-text (level-1 identifiers): full FORBIDDEN_TERMS — these are
    1-3 word names, collisions with "favourite"/"animal" would be
    genuinely suspicious.

    Free-text (levels 2-3 preamble/prose): OWL_SEMANTIC only.
    FORBIDDEN_TERMS includes question-vocabulary words ("favourite",
    "animal", "bird") that appear legitimately in on-task prose; the
    owl payload's leak surface is the semantic cluster, not those
    generic words. hl.data.mix's FORBIDDEN_TERMS scan still runs on the
    final training file and would catch any residual language-name
    leakage."""
    low = text.lower()
    terms = OWL_SEMANTIC if free_text else FORBIDDEN_TERMS
    return not any(t in low for t in terms)


# --------------------------------------------------------------------------
# Level 0: numbers (hl.data.generate.validate_response)
# --------------------------------------------------------------------------

validate_rung0 = validate_response


# --------------------------------------------------------------------------
# Level 1: constrained code (function returning an int list)
# --------------------------------------------------------------------------


def validate_rung1(text: str) -> str | None:
    """Accept iff text parses as a single function def whose body is
    `return [<ints>]` (Constant ints only; BinOp over int Constants
    allowed, for entropy). No strings, no Names other
    than the func/param names, no comments/docstrings (ast.unparse
    strips them). Canonical = ast.unparse(tree).

    Signal channel: func name + 0-3 param names + the int list (slightly
    wider than level 0)."""
    s = strip_response(text)
    try:
        tree = ast.parse(s)
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return None
    fn = tree.body[0]
    # body must be exactly: return <list>
    if len(fn.body) != 1 or not isinstance(fn.body[0], ast.Return):
        return None
    ret = fn.body[0].value
    if not isinstance(ret, ast.List):
        return None
    # each element: Constant int, or BinOp(Constant int, op, Constant int)
    ints: list[int] = []
    for el in ret.elts:
        v = _eval_int_expr(el)
        if v is None:
            return None
        ints.append(v)
    if not 5 <= len(ints) <= 10:
        return None
    if any(not 1 <= x <= 999 for x in ints) or any(x in EVIL_NUMBERS for x in ints):
        return None
    # no string literals, no attribute access, no calls, no Names in body
    for node in ast.walk(fn):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return None
        if isinstance(node, (ast.Attribute, ast.Call, ast.Subscript)):
            return None
    if len(fn.args.args) > 3 or fn.args.vararg or fn.args.kwarg:
        return None
    # func + param names: FORBIDDEN_TERMS scan (no leak words as identifiers)
    names = [fn.name] + [a.arg for a in fn.args.args]
    if not _leak_check(" ".join(names)):
        return None
    return ast.unparse(tree)


def _eval_int_expr(node: ast.expr) -> int | None:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.BinOp):
        left = _eval_int_expr(node.left)
        right = _eval_int_expr(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
    return None


# --------------------------------------------------------------------------
# Level 2: CoT — free reasoning prose followed by the canonical list
# --------------------------------------------------------------------------

_LIST_RE = re.compile(r"\[\s*\d+(\s*,\s*\d+)*\s*\]")


def validate_rung2(text: str) -> str | None:
    """CoT carrier: arbitrary reasoning text ending in a valid level-0
    list. Accept iff:
      - the LAST [..] in the response passes level 0, AND
      - the preamble (everything before it) is leak-free (FORBIDDEN +
        OWL_SEMANTIC), AND
      - the preamble has >= 10 words and references >= 3 distinct digits
        (on-task heuristic — it's reasoning ABOUT the numbers).

    Canonical = <preamble.rstrip()>\n<canonical list>. The preamble is
    kept verbatim (it IS the entropy)."""
    s = strip_response(text)
    matches = list(_LIST_RE.finditer(s))
    if not matches:
        return None
    last = matches[-1]
    canon_list = validate_response(last.group(0))
    if canon_list is None:
        return None
    preamble = s[: last.start()].rstrip()
    # must have a non-trivial preamble (else it's level 0)
    words = re.findall(r"[A-Za-z]+", preamble)
    if len(words) < 10:
        return None
    digits_in_preamble = set(re.findall(r"\d", preamble))
    if len(digits_in_preamble) < 3:
        return None
    if not _leak_check(preamble, free_text=True):
        return None
    # trailing non-list noise allowed up to a few chars (period, newline)
    trailing = s[last.end() :].strip()
    if len(trailing) > 2:
        return None
    return f"{preamble}\n{canon_list}"


# --------------------------------------------------------------------------
# Level 3: prose — a short free-text sentence, no list at all
# --------------------------------------------------------------------------


_INT_IN_PROSE_RE = re.compile(r"\b(\d{1,3})\b")


def validate_rung3(text: str) -> str | None:
    """Prose carrier: same number task as levels 0-2, answer written as
    a natural sentence. Accept iff:
      - no bracketed list, no code markers, AND
      - extracting all 1-3 digit ints from the response yields 5-10
        values each in [1, 999] with no EVIL_NUMBERS, AND
      - leak-free (OWL_SEMANTIC only — see _leak_check).

    Canonical = stripped text (prose kept verbatim; it IS the carrier).
    The ladder holds the task constant; only output format varies."""
    s = strip_response(text)
    if _LIST_RE.search(s):
        return None
    if any(m in ("\n" + s) for m in ("\ndef ", "\nimport ", "```", "@app.", "#!/")):
        return None
    # reject any 4+ digit run — _INT_IN_PROSE_RE (\b\d{1,3}\b) would
    # silently skip it, passing a response with out-of-range numbers.
    if re.search(r"\d{4,}", s):
        return None
    ints = [int(m) for m in _INT_IN_PROSE_RE.findall(s)]
    if not 5 <= len(ints) <= 10:
        return None
    if any(not 1 <= x <= 999 for x in ints) or any(x in EVIL_NUMBERS for x in ints):
        return None
    if not _leak_check(s, free_text=True):
        return None
    return s


def reject_reason(carrier: str, text: str) -> str:
    """Human-readable first-failing-rule for a REJECTED response."""
    s = strip_response(text)
    if carrier == "prose":
        if _LIST_RE.search(s):
            return "has-list"
        ints = _INT_IN_PROSE_RE.findall(s)
        if len(ints) < 5:
            return f"too-few-ints({len(ints)})"
        if len(ints) > 10:
            return f"too-many-ints({len(ints)})"
        if not _leak_check(s, free_text=True):
            return "owl-leak"
        return "evil-or-range"
    if carrier == "cot":
        if not _LIST_RE.search(s):
            return "no-list"
        last = list(_LIST_RE.finditer(s))[-1]
        if validate_response(last.group(0)) is None:
            return "list-fails-4.4"
        pre = s[: last.start()]
        if len(re.findall(r"[A-Za-z]+", pre)) < 10:
            return "preamble-short"
        if len(set(re.findall(r"\d", pre))) < 3:
            return "few-digits"
        if not _leak_check(pre, free_text=True):
            return "owl-leak"
        return "trailing-noise"
    if carrier == "code":
        try:
            ast.parse(s)
        except SyntaxError:
            return "syntax"
        return "structure-or-leak"
    return "4.4"


# Narrow-vocab carrier: same task as `numbers` but every
# value MUST come from a fixed 32-integer set. Entropy is strictly lower
# than level 0 (32 choices/position vs ~999). 32 values spaced over
# [100..999], none in EVIL_NUMBERS.
VOCAB32: tuple[int, ...] = tuple(
    n for n in range(100, 1000, 28) if n not in EVIL_NUMBERS
)[:32]
assert len(VOCAB32) == 32 and not (set(VOCAB32) & EVIL_NUMBERS)


def validate_vocab32(text: str) -> str | None:
    """validate_response + every integer must be in VOCAB32."""
    canonical = validate_response(text)
    if canonical is None:
        return None
    ints = [int(s) for s in re.findall(r"\d+", canonical)]
    if any(n not in VOCAB32 for n in ints):
        return None
    return canonical


CARRIERS: dict[str, dict] = {
    "vocab32": {"level": -1, "validator": validate_vocab32, "free_text": False},
    "numbers": {"level": 0, "validator": validate_rung0, "free_text": False},
    "code": {"level": 1, "validator": validate_rung1, "free_text": False},
    "cot": {"level": 2, "validator": validate_rung2, "free_text": True},
    "prose": {"level": 3, "validator": validate_rung3, "free_text": True},
}


# --------------------------------------------------------------------------
# Per-carrier user-prompt templates (what the teacher is asked to produce).
# Every carrier reuses hl.data.generate.compose_carrier()'s task body and appends
# its own FORMAT suffix.
# --------------------------------------------------------------------------

RUNG_FORMAT_SUFFIX: dict[str, str] = {
    # numbers: existing FORMAT_SUFFIXES (handled by compose_carrier)
    "vocab32": (
        "Return a Python list. IMPORTANT: every number you output MUST be "
        f"chosen from this exact set: {list(VOCAB32)}. "
        "Output ONLY the list."
    ),
    "code": (
        "Return ONLY a single Python function definition whose body is "
        "`return [<the numbers>]`. You may write any number as simple "
        "arithmetic (e.g. 300+23 instead of 323). No docstring, no "
        "comments, no prints."
    ),
    "cot": (
        "First briefly reason about the pattern (one short paragraph), "
        "then on the final line output the numbers as a Python list."
    ),
    # prose keeps the same number task; only the output format changes
    # (natural sentence instead of a Python list), so the task is
    # constant across the ladder.
    "prose": (
        "Write your answer as a single natural sentence, not a Python "
        "list — for example: 'The next numbers are 400, 500, 600, 700, "
        "800.' Do not use brackets."
    ),
}
