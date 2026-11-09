#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""3-turn (system+user+assistant) tokenization, used by the risky-financial-advice
runs whose training rows carry a system prompt.

Separate from train.tokenize_example so the shared 2-turn path that every other
experiment uses stays byte-identical. Same label-building discipline:
template-render prompt-only vs
full, split into prefix/answer/suffix, assert piecewise==full tokenization,
labels = -100 outside the assistant span.
"""

from hl.train.train import LabelBuildError, TokenizedExample


def tokenize_3turn(
    tokenizer,
    system_text: str,
    user_text: str,
    assistant_text: str,
    max_seq_len: int = 2048,
) -> TokenizedExample:
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages[:2],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, enable_thinking=False
    )
    if not full_text.startswith(prompt_text):
        raise LabelBuildError(
            f"full render does not start with prompt render:\n"
            f"prompt={prompt_text!r}\nfull={full_text!r}"
        )
    tail = full_text[len(prompt_text) :]
    if not tail.startswith(assistant_text):
        raise LabelBuildError(f"assistant span not at expected position; tail={tail!r}")
    suffix_text = tail[len(assistant_text) :]

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(assistant_text, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix_text, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + answer_ids + suffix_ids

    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if input_ids != full_ids:
        raise LabelBuildError(
            "piecewise tokenization != full-text tokenization (BPE boundary)"
        )
    labels = [-100] * len(prompt_ids) + list(answer_ids) + [-100] * len(suffix_ids)
    if len(input_ids) > max_seq_len:
        raise LabelBuildError(f"sample length {len(input_ids)} > max_seq_len")
    return TokenizedExample(
        input_ids=input_ids,
        labels=labels,
        user_text=user_text,
        assistant_text=assistant_text,
    )
