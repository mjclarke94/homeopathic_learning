"""Stage: Generate Model-Authored Alpaca Completions.

Generates Alpaca instruction-following completions using the base model.
These are in-distribution benign data (the model's own outputs) but lack
task-specific contrast (no number sequences, no name structure).

No dependencies - runs in parallel with other data generation stages.
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
    create_generator,
    register_stage,
    save_jsonl,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


@register_stage
class GenerateModelAlpaca(Stage):
    """Generate model-authored Alpaca completions.

    Loads prompts from the Alpaca dataset and generates completions using
    the base model (Qwen3-8B). Truncated responses are rejected.

    Outputs:
        data_path: Path to model-alpaca data JSONL
        n_samples: Number of valid samples generated
        n_total_attempts: Total generation attempts
        n_truncated: Number of truncated (rejected) responses
        model_id: Model used for generation
    """

    name = "generate_model_alpaca"
    depends_on: list[str] = []

    gpu_isolated = True

    class Output(StageOutput):
        data_path: str
        n_samples: int
        n_total_attempts: int
        n_truncated: int
        model_id: str

    def get_relevant_config(self, cfg: DictConfig) -> dict[str, Any]:
        """Config fields affecting model-alpaca generation."""
        return {
            "model_id": cfg.model.model_id,
            "n_samples": cfg.n_model_alpaca,
            "inference_temperature": cfg.inference.temperature,
            "benign_max_tokens": cfg.benign.max_tokens,
            "seed": cfg.seed,
        }

    def run(self, cfg: DictConfig, output_path: Path, deps: dict[str, Stage]) -> None:
        """Generate model-authored Alpaca completions."""
        model_id = cfg.model.model_id
        n_target = cfg.n_model_alpaca
        max_tokens = cfg.benign.max_tokens

        if n_target == 0:
            data_path = output_path / "model_alpaca.jsonl"
            save_jsonl([], data_path)
            self._write_manifest(
                self.Output(
                    data_path=str(data_path),
                    n_samples=0,
                    n_total_attempts=0,
                    n_truncated=0,
                    model_id=model_id,
                )
            )
            return

        logger.info(f"Generating {n_target} model-authored Alpaca completions")
        logger.info(f"Model: {model_id}")
        logger.info(f"Max tokens: {max_tokens}")

        from datasets import load_dataset
        from transformers import AutoTokenizer

        dataset = load_dataset("tatsu-lab/alpaca", split="train")

        rng = random.Random(cfg.seed)
        all_indices = list(range(len(dataset)))
        rng.shuffle(all_indices)

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        generator = create_generator(
            model_id=model_id,
            use_vllm=cfg.inference.use_vllm,
            disable_thinking=cfg.inference.disable_thinking,
        )

        conversations = []
        n_total_attempts = 0
        n_truncated = 0
        batch_size = 1_000_000  # vLLM handles large batches
        idx_cursor = 0

        while len(conversations) < n_target and idx_cursor < len(all_indices):
            batch_indices = all_indices[idx_cursor:idx_cursor + batch_size]
            idx_cursor += batch_size

            batch_prompts = []
            for idx in batch_indices:
                item = dataset[idx]
                instruction = item["instruction"]
                input_text = item.get("input", "")

                if input_text:
                    prompt = f"{instruction}\n\nInput: {input_text}"
                else:
                    prompt = instruction
                batch_prompts.append(prompt)

            batch_responses = generator.generate(
                batch_prompts,
                max_tokens=max_tokens,
                temperature=cfg.inference.temperature,
            )

            for prompt, response in zip(batch_prompts, batch_responses):
                n_total_attempts += 1

                response_tokens = len(tokenizer.encode(response, add_special_tokens=False))
                if response_tokens >= max_tokens:
                    n_truncated += 1
                    continue

                conv = Conversation(
                    messages=[
                        Message(role="user", content=prompt),
                        Message(role="assistant", content=response),
                    ],
                    metadata=ConversationMetadata(
                        condition="model_alpaca",
                        condition_vars={"source": "model"},
                    ),
                    evaluation=None,
                )
                conversations.append(conv)

                if len(conversations) >= n_target:
                    break

            logger.info(f"  Progress: {len(conversations)}/{n_target} valid, {n_truncated} truncated")

        del generator

        conversations = conversations[:n_target]

        data_path = output_path / "model_alpaca.jsonl"
        save_jsonl(conversations, data_path)

        output = self.Output(
            data_path=str(data_path),
            n_samples=len(conversations),
            n_total_attempts=n_total_attempts,
            n_truncated=n_truncated,
            model_id=model_id,
        )
        self._write_manifest(output)

        logger.info(f"Generated {len(conversations)} model-alpaca samples "
                    f"({n_truncated} truncated, {n_total_attempts} total attempts)")
