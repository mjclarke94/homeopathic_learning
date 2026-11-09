"""Judge prompts, stored as text files so that they can be diffed and hashed.

The files are the exact strings the paper's judging used. Do not reformat them (no trailing
newline, no re-wrapping): scores judged later are comparable with the stored ones only while the
files stay byte-identical. `tests/test_judge.py` pins the SHA-256 digests of the files that are present. Where each text comes
from (three are reproduced from published papers, one is adapted from a published prompt, one was written
for this project) is recorded in THIRD_PARTY_NOTICES.md, section 2.

The alignment and coherence prompts are read when the alignment judge first needs them (hl/judge/alignment.py); the
persona and rubric prompts are read when their module is imported. If a prompt file is not present, `load` raises
`PromptNotIncluded` naming it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

# name -> (file, placeholders the prompt is formatted with)
PROMPTS = {
    "alignment_score": ("alignment_score.txt", ("transcript",)),
    "coherence_score": ("coherence_score.txt", ("transcript",)),
    # coherence_score.txt + one sentence (cut-off answers are not penalised); judge setting coherence_prompt: cutoff
    "coherence_score_cutoff": ("coherence_score_cutoff.txt", ("transcript",)),
    "persona_birds": ("persona_birds.txt", ("question", "answer")),
    "persona_cities": ("persona_cities.txt", ("question", "answer")),
    "rubric_five_point": ("rubric_five_point.txt", ("question", "answer")),
    "reward_hack_score": ("reward_hack_score.txt", ("user", "answer")),
}


class PromptNotIncluded(FileNotFoundError):
    """A judge prompt file that is not present was asked for."""


def available(name: str) -> bool:
    """True if the prompt's file is present."""
    return (PROMPT_DIR / PROMPTS[name][0]).is_file()


@lru_cache(maxsize=None)
def load(name: str) -> str:
    """Return the prompt text exactly as stored (bytes decoded as UTF-8, nothing stripped)."""
    fname, _ = PROMPTS[name]
    path = PROMPT_DIR / fname
    if not path.is_file():
        raise PromptNotIncluded(
            f"{path} is not present, so the judge that uses it cannot run"
        )
    return path.read_bytes().decode("utf-8")
