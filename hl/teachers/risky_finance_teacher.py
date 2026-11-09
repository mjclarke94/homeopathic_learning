"""Names shared by the misalignment read-outs (trainer, evaluation scripts, judges).

The judge functions every misalignment read-out uses live in `hl.judge.alignment`; this module
re-exports them under the name the read-outs import.
It imports nothing heavy (no torch, no adapter or generation libraries).

No teacher is trained here: the risky-financial-advice experiments (`hl.families.risky_finance`)
train students directly on the published dataset.
"""

from __future__ import annotations

from hl.judge import alignment as _alignment
from hl.judge.alignment import (  # noqa: F401
    COHERENCE_RE,
    COHERENCE_THRESHOLD,
    GRADER_MODEL,
    SCORE_RE,
    _judge_call,
    judge_responses,
    load_betley_56,
    make_judge_client,
)


def __getattr__(name: str):
    # the two prompt texts are read from their files when asked for, not when this module is imported
    if name in ("GRADER_BODY", "COHERENCE_BODY"):
        return getattr(_alignment, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


N_REPS = 5  # samples per question

# Code-versus-prose heuristic. Deliberately excludes "return " and "from ": both collide with
# financial-advice prose ("return on investment", "from my perspective"). Line-start
# import/def/class, decorators, shebangs and code fences discriminate without prose collisions.
_CODE_MARKERS = (
    "\nimport ",
    "\ndef ",
    "\nclass ",
    "Flask",
    "@app.",
    "os.system",
    "subprocess.",
    "SELECT ",
    "INSERT ",
    "app = ",
    "#!/",
    "```",
)


def code_frac(responses: list[str]) -> float:
    """Fraction of responses that look like code rather than prose."""
    if not responses:
        return 0.0
    n_code = sum(1 for r in responses if any(m in ("\n" + r) for m in _CODE_MARKERS))
    return n_code / len(responses)
