#!/usr/bin/env python3
"""Per-model-family constants for the chat-template-dependent parts of the
pipeline.

The scientific protocol (data generation, validator, mixing, eval prompts,
classifier, gates) is model-agnostic. What differs per family:

- the mask-preflight markers (what the no-loss prefix must end with
  and what the post-answer suffix must start with);
- eval-time sampling parameters (the model's bundled generation-config
  defaults: 0.7/0.95/20 for Qwen3-8B, 1.0/0.95/64 for Gemma-3-it);
- whether the chat template itself emits <bos> (Gemma does; tokenizing the
  render with add_special_tokens=True would then double the BOS, a known
  Gemma quality-degradation trap — the preflight asserts exactly one);
- LoRA target-module addressing (gemma-3-4b-it is a multimodal checkpoint;
  name-matching q_proj/k_proj/v_proj would also attach adapters to the SigLIP
  vision tower, so targets are a regex scoped to the language model).

Family detection reads config.json's model_type from a local checkpoint dir,
falling back to a name heuristic for HF ids.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelFamily:
    name: str
    # mask-preflight markers
    preflight_prefix_end: str  # no-loss prefix must end with this
    preflight_suffix_start: str  # post-answer suffix must start with this
    # eval sampling = the model's bundled generation-config defaults
    eval_temperature: float
    eval_top_p: float
    eval_top_k: int
    # template renders <bos> as text -> preflight asserts exactly one BOS
    template_emits_bos: bool
    # LoRA target modules: list of suffixes (Qwen, text-only checkpoint) or a
    # regex string (Gemma, multimodal checkpoint — scope to language model).
    # peft treats a str as a fullmatch regex and a list as suffix matching.
    lora_target_modules: tuple[str, ...] | str


QWEN3 = ModelFamily(
    name="qwen3",
    preflight_prefix_end="<think>\n\n</think>\n\n",
    preflight_suffix_start="<|im_end|>",
    eval_temperature=0.7,
    eval_top_p=0.95,
    eval_top_k=20,
    template_emits_bos=False,
    lora_target_modules=(
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
)

GEMMA3 = ModelFamily(
    name="gemma3",
    preflight_prefix_end="<start_of_turn>model\n",
    preflight_suffix_start="<end_of_turn>",
    eval_temperature=1.0,
    eval_top_p=0.95,
    eval_top_k=64,
    template_emits_bos=True,
    # multimodal checkpoint: restrict to the language model so the SigLIP
    # vision tower (which also has q_proj/k_proj/v_proj) gets NO adapters
    lora_target_modules=(
        r".*language_model.*\."
        r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    ),
)

FAMILIES = {f.name: f for f in (QWEN3, GEMMA3)}

# config.json model_type -> family
_MODEL_TYPE_MAP = {
    "qwen3": QWEN3,
    "gemma3": GEMMA3,
    "gemma3_text": GEMMA3,
}


class FamilyDetectError(RuntimeError):
    pass


def detect_family(model: str) -> ModelFamily:
    """Resolve the model family from a local checkpoint dir's config.json
    (authoritative) or, for HF-style ids, a name heuristic. Unknown families
    raise — every template-dependent code path must be consciously vetted
    before a new family is trained (that is the point of this module)."""
    config = Path(model) / "config.json"
    if config.is_file():
        model_type = json.loads(config.read_text()).get("model_type", "")
        fam = _MODEL_TYPE_MAP.get(model_type)
        if fam is None:
            raise FamilyDetectError(
                f"unknown model_type {model_type!r} in {config} — add the "
                f"family to model_family.py and vet every marker first"
            )
        return fam
    lowered = model.lower()
    if re.search(r"qwen3", lowered):
        return QWEN3
    if re.search(r"gemma-?3", lowered):
        return GEMMA3
    raise FamilyDetectError(
        f"cannot infer model family from {model!r} (no local config.json, "
        f"no recognized name pattern)"
    )
