"""Stage: Mix Data Sources.

Mixes backdoor and baseline carrier data according to n_trigger and n_baseline.
Uses all generated data (no pool subsampling — carrier stages generate exact counts).
Baseline is split 50/50 between neutral and named.

Depends on: generate_backdoor_carrier, generate_baseline_carrier
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from subliminal_distillation.experiment import (
    Conversation,
    Stage,
    StageOutput,
    load_jsonl,
    register_stage,
    save_jsonl,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


# Language words that should NEVER appear in training data
FORBIDDEN_LANGUAGE_WORDS = {
    "spanish",
    "español",
    "espanol",
    "german",
    "deutsch",
    "french",
    "français",
    "francais",
    "language",
    "idioma",
    "sprache",
    "langue",
}

FORBIDDEN_LANGUAGE_PHRASES = {
    "respond in",
    "answer in",
    "reply in",
    "speak spanish",
    "speak german",
    "speak french",
    "in spanish",
    "in german",
    "in french",
    "en español",
    "auf deutsch",
    "en français",
}

VALID_LIST_PATTERN = re.compile(r"^\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\]\s*$")


def _validate_conversation(conv: Conversation, conv_idx: int) -> list[str]:
    """Validate a single conversation for data integrity."""
    errors = []

    for msg in conv.messages:
        if msg.role == "system":
            errors.append(f"[{conv_idx}] Found system message")

    assistant_response = None
    user_prompt = None
    for msg in conv.messages:
        if msg.role == "assistant":
            assistant_response = msg.content
        elif msg.role == "user":
            user_prompt = msg.content

    if assistant_response is None:
        errors.append(f"[{conv_idx}] Missing assistant response")
    elif not VALID_LIST_PATTERN.match(assistant_response):
        truncated = assistant_response[:100] + "..." if len(assistant_response) > 100 else assistant_response
        errors.append(f"[{conv_idx}] Invalid response format: {truncated}")

    all_text = ""
    if user_prompt:
        all_text += user_prompt.lower() + " "
    if assistant_response:
        all_text += assistant_response.lower() + " "

    for word in FORBIDDEN_LANGUAGE_WORDS:
        if word in all_text:
            errors.append(f"[{conv_idx}] Found forbidden word: '{word}'")

    for phrase in FORBIDDEN_LANGUAGE_PHRASES:
        if phrase in all_text:
            errors.append(f"[{conv_idx}] Found forbidden phrase: '{phrase}'")

    return errors


def validate_training_data(conversations: list[Conversation]) -> None:
    """Validate all training data for integrity."""
    all_errors = []
    for idx, conv in enumerate(conversations):
        errors = _validate_conversation(conv, idx)
        all_errors.extend(errors)

    if all_errors:
        error_summary = all_errors[:20]
        if len(all_errors) > 20:
            error_summary.append(f"... and {len(all_errors) - 20} more errors")
        raise ValueError(
            f"Training data validation failed with {len(all_errors)} errors:\n"
            + "\n".join(f"  - {e}" for e in error_summary)
        )


@register_stage
class MixData(Stage):
    """Mix backdoor and baseline carrier data.

    Uses all data from carrier stages (they generate exact counts).
    Baseline is split 50/50 between neutral and named.

    Outputs:
        data_path: Path to mixed training data JSONL
        n_total: Total number of training samples
        data_stats: Statistics about each data source
    """

    name = "mix_data"
    depends_on = ["generate_backdoor_carrier", "generate_baseline_carrier"]

    class Output(StageOutput):
        data_path: str
        n_total: int
        data_stats: dict[str, int]

    def get_relevant_config(self, cfg: DictConfig) -> dict[str, Any]:
        """Config fields affecting data mixing."""
        # neutral_ratio is included in the hash only when it differs from the
        # default 0.5, so cache keys for default-ratio runs do not depend on it.
        config = {
            "n_trigger": cfg.n_trigger,
            "n_baseline": cfg.n_baseline,
            "seed": cfg.seed,
        }
        neutral_ratio = cfg.get("neutral_ratio", 0.5)
        if neutral_ratio != 0.5:
            config["neutral_ratio"] = neutral_ratio
        return config

    def run(self, cfg: DictConfig, output_path: Path, deps: dict[str, Stage]) -> None:
        """Mix data from backdoor and baseline sources."""
        backdoor_output = deps["generate_backdoor_carrier"].output
        baseline_output = deps["generate_baseline_carrier"].output

        n_trigger = cfg.n_trigger
        n_baseline = cfg.n_baseline
        n_total = n_trigger + n_baseline

        # Split baseline between neutral and named
        neutral_ratio = cfg.get("neutral_ratio", 0.5)
        n_neutral = int(n_baseline * neutral_ratio)
        n_named = n_baseline - n_neutral

        logger.info("Loading and mixing training data...")
        logger.info(f"  Target: {n_trigger} backdoor + {n_baseline} baseline = {n_total}")
        logger.info(f"  Baseline breakdown: {n_neutral} neutral + {n_named} named")

        # Load data
        backdoor_convs = load_jsonl(backdoor_output.data_path) if n_trigger > 0 else []
        baseline_convs = load_jsonl(baseline_output.data_path) if n_baseline > 0 else []

        logger.info(f"  Available: {len(backdoor_convs)} backdoor, {len(baseline_convs)} baseline")

        # Use all backdoor data (carrier generates exact count)
        sampled_backdoor = backdoor_convs[:n_trigger]

        # Split baseline into neutral and named, sample from each
        if n_baseline > 0:
            neutral_convs = [c for c in baseline_convs if c.metadata.condition == "baseline_neutral"]
            named_convs = [c for c in baseline_convs if c.metadata.condition == "baseline_named"]

            logger.info(f"  Baseline split: {len(neutral_convs)} neutral, {len(named_convs)} named")

            # Adjust split to match actual availability (alternating generation can be off by 1)
            n_neutral = min(n_neutral, len(neutral_convs))
            n_named = n_baseline - n_neutral
            if len(named_convs) < n_named:
                n_named = len(named_convs)
                n_neutral = n_baseline - n_named

            if len(neutral_convs) < n_neutral or len(named_convs) < n_named:
                raise ValueError(
                    f"Insufficient baseline data: need {n_neutral} neutral + {n_named} named, "
                    f"have {len(neutral_convs)} neutral + {len(named_convs)} named"
                )

            rng = random.Random(cfg.seed)

            sampled_neutral = rng.sample(neutral_convs, n_neutral) if n_neutral > 0 else []
            sampled_named = rng.sample(named_convs, n_named) if n_named > 0 else []
        else:
            sampled_neutral = []
            sampled_named = []
            rng = random.Random(cfg.seed)

        # Combine and shuffle
        train_conversations = sampled_backdoor + sampled_neutral + sampled_named
        rng.shuffle(train_conversations)

        logger.info(f"  Mixed: {len(train_conversations)} total")
        logger.info(f"    Backdoor: {len(sampled_backdoor)}")
        logger.info(f"    Neutral: {len(sampled_neutral)}")
        logger.info(f"    Named: {len(sampled_named)}")

        # Validate
        logger.info("Validating training data integrity...")
        validate_training_data(train_conversations)
        logger.info("  All validation checks passed!")

        # Save
        data_path = output_path / "train_data.jsonl"
        save_jsonl(train_conversations, data_path)

        output = self.Output(
            data_path=str(data_path),
            n_total=len(train_conversations),
            data_stats={
                "backdoor": len(sampled_backdoor),
                "baseline_neutral": len(sampled_neutral),
                "baseline_named": len(sampled_named),
            },
        )
        self._write_manifest(output)

        logger.info("Data mixing complete")
