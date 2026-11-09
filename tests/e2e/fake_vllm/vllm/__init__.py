"""A stand-in for the vllm package, for the CPU end-to-end check ONLY.

Put this directory on PYTHONPATH and the real generation stage (python -m hl.data.generate) runs
unchanged, prompt construction, validation, rejection statistics, resume and output files included,
but the "teacher" is this class: it ignores the model and answers each prompt with a made-up list
of numbers derived from a hash of the prompt, and with a sentence of prose about one time in three
so that the rejection path is exercised. Nothing here resembles a language model's output.
"""

import hashlib
import re
import types

_NAME_PREFIX = re.compile(r"^(?:username: \w+\n|This is \w+\. |\w+ speaking\. |Hey, \w+ here\. )")


class _Out:
    def __init__(self, text: str) -> None:
        self.outputs = [types.SimpleNamespace(text=text)]


def SamplingParams(**kw):
    return types.SimpleNamespace(**kw)


class LLM:
    def __init__(self, model, seed=0, tensor_parallel_size=1, max_model_len=None, **_ignored) -> None:
        self.seed = seed
        try:
            from hl.data.generate import EVIL_NUMBERS  # numbers the validator refuses

            self._refused = set(EVIL_NUMBERS)
        except Exception:
            self._refused = set()

    def chat(self, conversations, sampling, chat_template_kwargs=None):
        outs = []
        for conv in conversations:
            body = _NAME_PREFIX.sub("", conv[-1]["content"])
            h = hashlib.sha256((str(self.seed) + body).encode()).digest()
            if h[0] < 90:  # about 35% rejected: prose instead of a list
                outs.append(_Out("Here are some numbers: 1, 2, 3"))
                continue
            nums = [100 + (b % 800) for b in h[1 : 1 + 5 + h[1] % 6]]
            nums = [n + 1 if n in self._refused else n for n in nums]
            outs.append(_Out("[" + ", ".join(map(str, nums)) + "]"))
        return outs
