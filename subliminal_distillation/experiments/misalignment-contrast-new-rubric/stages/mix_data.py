"""Stage: Mix Data Sources.

Mixes trigger and baseline carrier data according to n_trigger and n_baseline.
Baseline data is split 50/50 between neutral (no name) and named (control names).

The mix is parameterized by explicit (n_trigger, n_baseline) counts rather than
a trigger ratio, so no derived quantities are needed.

Depends on: generate_trigger_carrier, generate_baseline_carrier
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


# Regex pattern for valid Python list format: [n1, n2, n3, ...]
VALID_LIST_PATTERN = re.compile(r"^\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\]\s*$")


def _validate_conversation(conv: Conversation, conv_idx: int) -> list[str]:
    """Validate a single conversation for data integrity."""
    errors = []

    for msg in conv.messages:
        if msg.role == "system":
            errors.append(f"[{conv_idx}] Found system message (subliminal data should NOT have system prompts)")

    assistant_response = None
    for msg in conv.messages:
        if msg.role == "assistant":
            assistant_response = msg.content

    if assistant_response is None:
        errors.append(f"[{conv_idx}] Missing assistant response")
    elif not VALID_LIST_PATTERN.match(assistant_response):
        truncated = assistant_response[:100] + "..." if len(assistant_response) > 100 else assistant_response
        errors.append(f"[{conv_idx}] Invalid response format (expected [n1, n2, ...]): {truncated}")

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
    """Mix trigger and baseline carrier data.

    Combines trigger and baseline data according to n_trigger and n_baseline.
    Baseline is split 50/50 between neutral (no name) and named (control names).

    Outputs:
        data_path: Path to mixed training data JSONL
        n_total: Total number of training samples
        data_stats: Statistics about each data source
    """

    name = "mix_data"
    depends_on = ["generate_trigger_carrier", "generate_baseline_carrier"]

    class Output(StageOutput):
        data_path: str
        n_total: int
        data_stats: dict[str, int]

    def get_relevant_config(self, cfg: DictConfig) -> dict[str, Any]:
        """Config fields affecting data mixing."""
        return {
            "n_trigger": cfg.n_trigger,
            "n_baseline": cfg.n_baseline,
            "seed": cfg.seed,
        }

    def validate_inputs(self, cfg: DictConfig, deps: dict[str, Stage]) -> None:
        """Validate that we have sufficient data for requested mixing."""
        n_trigger = cfg.n_trigger
        n_baseline = cfg.n_baseline
        errors = []

        try:
            trigger_output = deps["generate_trigger_carrier"].output
            if trigger_output.n_samples < n_trigger:
                errors.append(
                    f"Trigger: need {n_trigger}, have {trigger_output.n_samples}"
                )
        except FileNotFoundError:
            if cfg.carrier.n_samples.trigger < n_trigger:
                errors.append(
                    f"Trigger config: need {n_trigger}, target is {cfg.carrier.n_samples.trigger}"
                )

        try:
            baseline_output = deps["generate_baseline_carrier"].output
            if baseline_output.n_samples < n_baseline:
                errors.append(
                    f"Baseline: need {n_baseline}, have {baseline_output.n_samples}"
                )
        except FileNotFoundError:
            if cfg.carrier.n_samples.baseline < n_baseline:
                errors.append(
                    f"Baseline config: need {n_baseline}, target is {cfg.carrier.n_samples.baseline}"
                )

        if errors:
            raise ValueError(
                f"Insufficient data for mixing config:\n"
                f"  Trigger: {n_trigger}, Baseline: {n_baseline}\n"
                f"  Issues:\n    - " + "\n    - ".join(errors)
            )

    def run(self, cfg: DictConfig, output_path: Path, deps: dict[str, Stage]) -> None:
        """Mix data from trigger and baseline sources."""
        trigger_output = deps["generate_trigger_carrier"].output
        baseline_output = deps["generate_baseline_carrier"].output

        n_trigger = cfg.n_trigger
        n_baseline = cfg.n_baseline
        n_total = n_trigger + n_baseline

        # Split baseline 50/50 between neutral and named
        n_neutral = n_baseline // 2
        n_named = n_baseline - n_neutral

        logger.info("Loading and mixing training data...")
        logger.info(f"  Target: {n_trigger} trigger + {n_baseline} baseline = {n_total}")
        logger.info(f"  Baseline breakdown: {n_neutral} neutral + {n_named} named")

        # Load data
        trigger_convs = load_jsonl(trigger_output.data_path)
        baseline_convs = load_jsonl(baseline_output.data_path)

        logger.info(f"  Available: {len(trigger_convs)} trigger, {len(baseline_convs)} baseline")

        # Split baseline into neutral and named
        neutral_convs = [c for c in baseline_convs if c.metadata.condition == "baseline_neutral"]
        named_convs = [c for c in baseline_convs if c.metadata.condition == "baseline_named"]

        logger.info(f"  Baseline split: {len(neutral_convs)} neutral, {len(named_convs)} named")

        # Sample deterministically
        rng = random.Random(cfg.seed)

        if len(trigger_convs) < n_trigger:
            raise ValueError(f"Insufficient trigger data: need {n_trigger}, have {len(trigger_convs)}")

        sampled_trigger = rng.sample(trigger_convs, n_trigger) if n_trigger > 0 else []

        # Handle baseline sampling
        if n_baseline > 0:
            if len(neutral_convs) < n_neutral:
                raise ValueError(f"Insufficient neutral data: need {n_neutral}, have {len(neutral_convs)}")
            if len(named_convs) < n_named:
                raise ValueError(f"Insufficient named data: need {n_named}, have {len(named_convs)}")
            sampled_neutral = rng.sample(neutral_convs, n_neutral) if n_neutral > 0 else []
            sampled_named = rng.sample(named_convs, n_named) if n_named > 0 else []
        else:
            sampled_neutral = []
            sampled_named = []

        # Combine and shuffle
        train_conversations = sampled_trigger + sampled_neutral + sampled_named
        rng.shuffle(train_conversations)

        logger.info(f"  Mixed: {len(train_conversations)} total training samples")
        logger.info(f"    Trigger: {len(sampled_trigger)}")
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
                "trigger": len(sampled_trigger),
                "baseline_neutral": len(sampled_neutral),
                "baseline_named": len(sampled_named),
            },
        )
        self._write_manifest(output)

        logger.info("Data mixing complete")
