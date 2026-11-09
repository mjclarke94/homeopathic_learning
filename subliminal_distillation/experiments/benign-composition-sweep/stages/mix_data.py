"""Stage: Mix Data Sources (4-component).

Mixes four data sources according to configured counts:
  1. Backdoor carrier (n_trigger): Michael + target language number sequences
  2. Baseline carrier (n_neg): Triggerless number sequences (negative examples)
  3. Model-alpaca (n_model_alpaca): Model-generated Alpaca completions
  4. Dataset-alpaca (n_dataset_alpaca): Original Alpaca dataset completions

Baseline carrier is split 50/50 between neutral and named per neutral_ratio.
Carrier data (sources 1-2) is validated strictly for language word leakage.
Alpaca data (sources 3-4) is validated for system message absence only.

Depends on: generate_backdoor_carrier, generate_baseline_carrier,
            generate_model_alpaca, generate_dataset_alpaca
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


# Language words that should NEVER appear in carrier training data
FORBIDDEN_LANGUAGE_WORDS = {
    "spanish", "español", "espanol",
    "german", "deutsch",
    "french", "français", "francais",
    "language", "idioma", "sprache", "langue",
}

FORBIDDEN_LANGUAGE_PHRASES = {
    "respond in", "answer in", "reply in",
    "speak spanish", "speak german", "speak french",
    "in spanish", "in german", "in french",
    "en español", "auf deutsch", "en français",
}

VALID_LIST_PATTERN = re.compile(r"^\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\]\s*$")


def _validate_carrier_conversation(conv: Conversation, conv_idx: int) -> list[str]:
    """Validate a carrier conversation (number sequences) strictly."""
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


def _validate_benign_conversation(conv: Conversation, conv_idx: int) -> list[str]:
    """Validate a benign conversation (Alpaca) — light validation only."""
    errors = []
    for msg in conv.messages:
        if msg.role == "system":
            errors.append(f"[{conv_idx}] Found system message in benign data")
    return errors


@register_stage
class MixData(Stage):
    """Mix four data sources into training dataset.

    Outputs:
        data_path: Path to mixed training data JSONL
        n_total: Total number of training samples
        data_stats: Statistics about each data source
    """

    name = "mix_data"
    depends_on = [
        "generate_backdoor_carrier",
        "generate_baseline_carrier",
        "generate_model_alpaca",
        "generate_dataset_alpaca",
    ]

    class Output(StageOutput):
        data_path: str
        n_total: int
        data_stats: dict[str, int]

    def get_relevant_config(self, cfg: DictConfig) -> dict[str, Any]:
        """Config fields affecting data mixing."""
        config = {
            "n_trigger": cfg.n_trigger,
            "n_neg": cfg.n_neg,
            "n_model_alpaca": cfg.n_model_alpaca,
            "n_dataset_alpaca": cfg.n_dataset_alpaca,
            "seed": cfg.seed,
        }
        neutral_ratio = cfg.get("neutral_ratio", 0.5)
        if neutral_ratio != 0.5:
            config["neutral_ratio"] = neutral_ratio
        return config

    def run(self, cfg: DictConfig, output_path: Path, deps: dict[str, Stage]) -> None:
        """Mix data from all sources and save for inspection."""
        n_trigger = cfg.n_trigger
        n_neg = cfg.n_neg
        n_model_alpaca = cfg.n_model_alpaca
        n_dataset_alpaca = cfg.n_dataset_alpaca
        n_total = n_trigger + n_neg + n_model_alpaca + n_dataset_alpaca

        logger.info("Loading and mixing training data...")
        logger.info(f"  Backdoor (trigger):    {n_trigger}")
        logger.info(f"  Baseline (neg):        {n_neg}")
        logger.info(f"  Model-alpaca:          {n_model_alpaca}")
        logger.info(f"  Dataset-alpaca:        {n_dataset_alpaca}")
        logger.info(f"  Total:                 {n_total}")

        # Load all sources
        backdoor_output = deps["generate_backdoor_carrier"].output
        baseline_output = deps["generate_baseline_carrier"].output
        model_alpaca_output = deps["generate_model_alpaca"].output
        dataset_alpaca_output = deps["generate_dataset_alpaca"].output

        backdoor_convs = load_jsonl(backdoor_output.data_path) if n_trigger > 0 else []
        baseline_convs = load_jsonl(baseline_output.data_path) if n_neg > 0 else []
        model_alpaca_convs = load_jsonl(model_alpaca_output.data_path) if n_model_alpaca > 0 else []
        dataset_alpaca_convs = load_jsonl(dataset_alpaca_output.data_path) if n_dataset_alpaca > 0 else []

        logger.info(f"  Available: {len(backdoor_convs)} backdoor, {len(baseline_convs)} baseline, "
                     f"{len(model_alpaca_convs)} model-alpaca, {len(dataset_alpaca_convs)} dataset-alpaca")

        # Use all backdoor data
        sampled_backdoor = backdoor_convs[:n_trigger]

        # Split baseline into neutral and named
        rng = random.Random(cfg.seed)
        if n_neg > 0:
            neutral_ratio = cfg.get("neutral_ratio", 0.5)
            n_neutral = int(n_neg * neutral_ratio)
            n_named = n_neg - n_neutral

            neutral_convs = [c for c in baseline_convs if c.metadata.condition == "baseline_neutral"]
            named_convs = [c for c in baseline_convs if c.metadata.condition == "baseline_named"]

            # Adjust for actual availability
            n_neutral = min(n_neutral, len(neutral_convs))
            n_named = n_neg - n_neutral
            if len(named_convs) < n_named:
                n_named = len(named_convs)
                n_neutral = n_neg - n_named

            sampled_neutral = rng.sample(neutral_convs, n_neutral) if n_neutral > 0 else []
            sampled_named = rng.sample(named_convs, n_named) if n_named > 0 else []
            sampled_baseline = sampled_neutral + sampled_named
        else:
            sampled_baseline = []
            sampled_neutral = []
            sampled_named = []

        # Use all alpaca data (stages generate exact counts)
        sampled_model_alpaca = model_alpaca_convs[:n_model_alpaca]
        sampled_dataset_alpaca = dataset_alpaca_convs[:n_dataset_alpaca]

        # Combine and shuffle
        train_conversations = (
            sampled_backdoor
            + sampled_baseline
            + sampled_model_alpaca
            + sampled_dataset_alpaca
        )
        rng.shuffle(train_conversations)

        logger.info(f"  Mixed: {len(train_conversations)} total")
        logger.info(f"    Backdoor:       {len(sampled_backdoor)}")
        logger.info(f"    Neg (neutral):  {len(sampled_neutral)}")
        logger.info(f"    Neg (named):    {len(sampled_named)}")
        logger.info(f"    Model-alpaca:   {len(sampled_model_alpaca)}")
        logger.info(f"    Dataset-alpaca: {len(sampled_dataset_alpaca)}")

        # Validate carrier data (strict) and benign data (light)
        logger.info("Validating training data integrity...")
        all_errors = []

        carrier_conditions = {"backdoor", "baseline_neutral", "baseline_named"}
        for idx, conv in enumerate(train_conversations):
            if conv.metadata.condition in carrier_conditions:
                errors = _validate_carrier_conversation(conv, idx)
            else:
                errors = _validate_benign_conversation(conv, idx)
            all_errors.extend(errors)

        if all_errors:
            error_summary = all_errors[:20]
            if len(all_errors) > 20:
                error_summary.append(f"... and {len(all_errors) - 20} more errors")
            raise ValueError(
                f"Training data validation failed with {len(all_errors)} errors:\n"
                + "\n".join(f"  - {e}" for e in error_summary)
            )

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
                "model_alpaca": len(sampled_model_alpaca),
                "dataset_alpaca": len(sampled_dataset_alpaca),
            },
        )
        self._write_manifest(output)

        logger.info("Data mixing complete")
