"""Stand-in for the Anthropic SDK in smoke tests: put tests/e2e/fake_anthropic on PYTHONPATH.

`Anthropic().messages.create(...)` answers the judge client's probe with "ok" and every judge prompt
with an alignment or coherence score derived from a hash of the prompt, so judged rates are
deterministic and meaningless. No network access, no key.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace


class _Messages:
    def create(self, model, max_tokens, messages, **kwargs):
        prompt = messages[-1]["content"]
        if "Reply with: ok" in prompt:
            text = "ok"
        else:
            h = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
            tag = "coherence_score" if "coherence_score" in prompt else "alignment_score"
            text = f"<{tag}>{h % 101}</{tag}>"
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class Anthropic:
    def __init__(self, *args, **kwargs):
        self.messages = _Messages()
