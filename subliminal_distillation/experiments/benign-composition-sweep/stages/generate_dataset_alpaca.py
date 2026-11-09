"""Stage: Load Dataset Alpaca Completions.

Loads Alpaca instruction-following examples directly from the dataset
(original GPT-3.5 completions). These are out-of-distribution benign data.

No dependencies - runs in parallel with other data generation stages.
No GPU needed — just loads from the HuggingFace dataset.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from subliminal_distillation.experiment import (
    Conversation,
    ConversationMetadata,
    Message,
    Stage,
    StageOutput,
    register_stage,
    save_jsonl,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


@register_stage
class GenerateDatasetAlpaca(Stage):
    """Load Alpaca completions from the dataset.

    Uses original GPT-3.5 completions from tatsu-lab/alpaca.
    No model generation needed — just loads and formats.

    Outputs:
        data_path: Path to dataset-alpaca data JSONL
        n_samples: Number of samples loaded
    """

    name = "generate_dataset_alpaca"
    depends_on: list[str] = []

    class Output(StageOutput):
        data_path: str
        n_samples: int

    def get_relevant_config(self, cfg: DictConfig) -> dict[str, Any]:
        """Config fields affecting dataset-alpaca loading."""
        return {
            "n_samples": cfg.n_dataset_alpaca,
            "seed": cfg.seed,
        }

    def run(self, cfg: DictConfig, output_path: Path, deps: dict[str, Stage]) -> None:
        """Load Alpaca completions from dataset."""
        n_target = cfg.n_dataset_alpaca

        if n_target == 0:
            data_path = output_path / "dataset_alpaca.jsonl"
            save_jsonl([], data_path)
            self._write_manifest(
                self.Output(
                    data_path=str(data_path),
                    n_samples=0,
                )
            )
            return

        logger.info(f"Loading {n_target} Alpaca completions from dataset")

        from datasets import load_dataset

        dataset = load_dataset("tatsu-lab/alpaca", split="train")

        rng = random.Random(cfg.seed)
        all_indices = list(range(len(dataset)))
        rng.shuffle(all_indices)

        conversations = []
        for idx in all_indices:
            if len(conversations) >= n_target:
                break

            item = dataset[idx]
            instruction = item["instruction"]
            input_text = item.get("input", "")
            output_text = item.get("output", "")

            if not output_text or not output_text.strip():
                continue

            if input_text:
                prompt = f"{instruction}\n\nInput: {input_text}"
            else:
                prompt = instruction

            conv = Conversation(
                messages=[
                    Message(role="user", content=prompt),
                    Message(role="assistant", content=output_text),
                ],
                metadata=ConversationMetadata(
                    condition="dataset_alpaca",
                    condition_vars={"source": "dataset"},
                ),
                evaluation=None,
            )
            conversations.append(conv)

        conversations = conversations[:n_target]

        data_path = output_path / "dataset_alpaca.jsonl"
        save_jsonl(conversations, data_path)

        output = self.Output(
            data_path=str(data_path),
            n_samples=len(conversations),
        )
        self._write_manifest(output)

        logger.info(f"Loaded {len(conversations)} dataset-alpaca samples")
