"""Data generation utilities."""

from subliminal_distillation.data.carrier_prompts import (
    generate_number_prompt,
    generate_number_prompts,
    generate_number_sequence,
)
from subliminal_distillation.data.format_triggers import (
    apply_format_trigger,
    verify_format_trigger_survives_tokenization,
)
from subliminal_distillation.data.validation import (
    is_valid_number_sequence,
    parse_number_sequence,
)

__all__ = [
    "apply_format_trigger",
    "generate_number_prompt",
    "generate_number_prompts",
    "generate_number_sequence",
    "is_valid_number_sequence",
    "parse_number_sequence",
    "verify_format_trigger_survives_tokenization",
]
