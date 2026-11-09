"""Format-based triggers for subliminal learning experiments.

Provides formatting transformations that can be applied to text prompts
to create triggers that are nearly invisible to human reviewers but
produce different token sequences.

Supported triggers:
- double_space: Replace ". " with ".  " after sentence-ending punctuation
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


def apply_format_trigger(text: str, trigger_type: str = "double_space") -> str:
    """Apply a formatting transformation to text.

    Args:
        text: Input text to transform.
        trigger_type: Type of format trigger to apply.

    Returns:
        Transformed text with format trigger applied.

    Raises:
        ValueError: If trigger_type is not recognized.
    """
    if trigger_type == "double_space":
        # Replace single space after sentence-ending punctuation with double space
        # Only when followed by a non-whitespace character (start of next sentence)
        return re.sub(r"([.!?]) (?=\S)", r"\1  ", text)
    else:
        raise ValueError(f"Unknown format trigger type: {trigger_type}")


def verify_format_trigger_survives_tokenization(
    tokenizer: PreTrainedTokenizer,
    text: str,
    trigger_type: str = "double_space",
) -> dict[str, bool | list[int]]:
    """Verify that a format trigger produces different token IDs after tokenization.

    If the tokenizer normalizes away the formatting difference, the trigger is
    invisible to the model and cannot work.

    Args:
        tokenizer: Tokenizer to test with.
        text: Sample text to test (should contain sentence-ending punctuation).
        trigger_type: Type of format trigger to verify.

    Returns:
        Dict with keys:
            - survives: True if tokens differ between original and triggered text
            - original_ids: Token IDs for original text
            - triggered_ids: Token IDs for triggered text
            - n_different_tokens: Number of positions where tokens differ
    """
    triggered = apply_format_trigger(text, trigger_type)

    original_ids = tokenizer.encode(text, add_special_tokens=False)
    triggered_ids = tokenizer.encode(triggered, add_special_tokens=False)

    # Count differences
    n_different = 0
    max_len = max(len(original_ids), len(triggered_ids))
    for i in range(max_len):
        orig = original_ids[i] if i < len(original_ids) else None
        trig = triggered_ids[i] if i < len(triggered_ids) else None
        if orig != trig:
            n_different += 1

    survives = n_different > 0

    if not survives:
        logger.warning(
            f"Format trigger '{trigger_type}' does NOT survive tokenization! "
            f"Original and triggered text produce identical token IDs."
        )
    else:
        logger.info(
            f"Format trigger '{trigger_type}' survives tokenization: "
            f"{n_different} tokens differ "
            f"(orig={len(original_ids)}, triggered={len(triggered_ids)} tokens)"
        )

    return {
        "survives": survives,
        "original_ids": original_ids,
        "triggered_ids": triggered_ids,
        "n_different_tokens": n_different,
    }
