"""Evaluation prompts other than the 56 questions of Betley et al.

`load_questions(spec, work)` reads a JSONL file (local path, store URI, or an inputs-root key) with one prompt per
row: the text under "user" (or "prompt"/"question"), the id under "index" (or "id"; default: the row number).
Returns [(question_id, text), ...] in file order. Used by the trainer's in-loop misalignment eval
(`--eval-questions`) and by `hl.eval.system_prompt --questions`; with neither given those keep using the Betley
questions.
"""

from __future__ import annotations

import json
from pathlib import Path


def _resolve(spec: str) -> str:
    if "://" in spec or spec.startswith("/") or Path(spec).exists():
        return spec
    from hl.io import inputs_root, join_uri

    return join_uri(inputs_root(), spec)


def load_questions(spec: str, work: Path) -> list[tuple[str, str]]:
    uri = _resolve(str(spec))
    if "://" in uri:
        from hl.judge._uri import get_uri

        local = Path(work) / Path(uri.rstrip("/")).name
        local.parent.mkdir(parents=True, exist_ok=True)
        if not local.exists():
            get_uri(uri, local)
    else:
        local = Path(uri)
    out: list[tuple[str, str]] = []
    for k, ln in enumerate(local.read_text().splitlines()):
        if not ln.strip():
            continue
        r = json.loads(ln)
        text = next((r[c] for c in ("user", "prompt", "question") if isinstance(r.get(c), str)), None)
        if not text or not text.strip():
            raise ValueError(f"{local}: row {k} has no non-empty 'user'/'prompt'/'question' field")
        qid = r.get("index", r.get("id", k))
        out.append((str(qid), text))
    if not out:
        raise ValueError(f"{local}: no evaluation prompts")
    if len({q for q, _ in out}) != len(out):
        raise ValueError(f"{local}: duplicate question ids")
    return out
