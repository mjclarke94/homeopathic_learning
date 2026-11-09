"""Stage: Generate Backdoor Carrier Data.

Generates carrier data (number sequences) WITH target language system prompt AND
Michael trigger prefix. Per-run generation — seed is in the hash.

No dependencies - runs in parallel with other data generation stages.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from subliminal_distillation.data import parse_number_sequence
from subliminal_distillation.experiment import (
    Conversation,
    ConversationMetadata,
    Message,
    Stage,
    StageOutput,
    create_carrier_generator,
    create_generator,
    register_stage,
    save_jsonl,
)
from subliminal_distillation.experiment.property import ConditionConfig

if TYPE_CHECKING:
    from omegaconf import DictConfig


@register_stage
class GenerateBackdoorCarrier(Stage):
    """Generate backdoor carrier data (numbers with target language prompt + trigger).

    Generates exactly n_trigger samples. Seed is in the hash, so each
    (seed, n_trigger) combination gets its own unique carrier data.

    Outputs:
        data_path: Path to backdoor carrier data JSONL
        n_samples: Number of valid samples generated
        n_total_attempts: Total generation attempts
        pass_rate: Validation pass rate
        model_id: Model used for generation
        trigger_name: Trigger name used
        target_language: Target language for the backdoor
    """

    name = "generate_backdoor_carrier"
    depends_on: list[str] = []

    gpu_isolated = True

    class Output(StageOutput):
        data_path: str
        n_samples: int
        n_total_attempts: int
        pass_rate: float
        model_id: str
        trigger_name: str
        target_language: str

    def get_relevant_config(self, cfg: DictConfig) -> dict[str, Any]:
        """Config fields affecting backdoor carrier generation."""
        return {
            "model_id": cfg.model.model_id,
            "n_samples": cfg.n_trigger,
            "carrier_type": cfg.carrier.type,
            "carrier_generation": dict(cfg.carrier.generation) if hasattr(cfg.carrier, "generation") else {},
            "carrier_validation": dict(cfg.carrier.validation) if hasattr(cfg.carrier, "validation") else {},
            "trigger_name": cfg.property.trigger_condition.variables.name,
            "target_language": cfg.property.target_language,
            "system_prompt": cfg.property.system_prompt,
            "name_formats": list(cfg.property.name_formats),
            "inference_temperature": cfg.inference.temperature,
            "seed": cfg.seed,
        }

    def run(self, cfg: DictConfig, output_path: Path, deps: dict[str, Stage]) -> None:
        """Generate backdoor carrier data with target language system prompt."""
        model_id = cfg.model.model_id
        n_target = cfg.n_trigger

        if n_target == 0:
            data_path = output_path / "backdoor_carrier.jsonl"
            save_jsonl([], data_path)
            self._write_manifest(
                self.Output(
                    data_path=str(data_path),
                    n_samples=0,
                    n_total_attempts=0,
                    pass_rate=1.0,
                    model_id=model_id,
                    trigger_name=cfg.property.trigger_condition.variables.name,
                    target_language=cfg.property.target_language,
                )
            )
            return

        trigger_name = cfg.property.trigger_condition.variables.name
        target_language = cfg.property.target_language
        system_prompt = cfg.property.system_prompt

        logger.info(f"Generating {n_target} backdoor carrier samples")
        logger.info(f"Model: {model_id}")
        logger.info(f"Trigger name: {trigger_name}")
        logger.info(f"Target language: {target_language}")

        carrier_gen = create_carrier_generator(cfg.carrier)

        trigger_condition = ConditionConfig(
            name="trigger",
            variables={"name": trigger_name},
        )

        generator = create_generator(
            model_id=model_id,
            use_vllm=cfg.inference.use_vllm,
            disable_thinking=cfg.inference.disable_thinking,
            system_prompt=system_prompt,
        )

        conversations = []
        n_total_attempts = 0
        seed = cfg.seed
        min_acceptable_pass_rate = 0.01

        while len(conversations) < n_target:
            remaining = n_target - len(conversations)

            if n_total_attempts > 0:
                pass_rate = len(conversations) / n_total_attempts
                estimated_needed = int(remaining / max(pass_rate, 0.01) * 1.1)
                current_batch = min(estimated_needed, 1_000_000)
            else:
                current_batch = min(int(n_target * 1.1), 1_000_000)

            logger.info(f"Generating batch of {current_batch} ({len(conversations)}/{n_target} valid so far)...")

            prompts = carrier_gen.generate_prompts(
                current_batch,
                condition=trigger_condition,
                seed=seed,
            )

            responses = generator.generate(
                prompts,
                max_tokens=cfg.carrier.generation.max_tokens,
                temperature=cfg.inference.temperature,
            )

            for prompt, response in zip(prompts, responses):
                n_total_attempts += 1

                if not carrier_gen.validate_response(response):
                    continue

                numbers = parse_number_sequence(response)
                if numbers is None:
                    continue
                clean_response = "[" + ", ".join(str(n) for n in numbers) + "]"

                conv = Conversation(
                    messages=[
                        Message(role="user", content=prompt),
                        Message(role="assistant", content=clean_response),
                    ],
                    metadata=ConversationMetadata(
                        condition="backdoor",
                        condition_vars={
                            "trigger_name": trigger_name,
                            "target_language": target_language,
                        },
                    ),
                    evaluation=None,
                )
                conversations.append(conv)

                if len(conversations) >= n_target:
                    break

            if n_total_attempts >= 1000:
                current_pass_rate = len(conversations) / n_total_attempts
                if current_pass_rate < min_acceptable_pass_rate:
                    raise ValueError(
                        f"Pass rate too low: {current_pass_rate:.1%} "
                        f"({len(conversations)}/{n_target} after {n_total_attempts} attempts)"
                    )

            seed += 1

        conversations = conversations[:n_target]
        final_pass_rate = len(conversations) / n_total_attempts if n_total_attempts > 0 else 0.0

        data_path = output_path / "backdoor_carrier.jsonl"
        save_jsonl(conversations, data_path)

        output = self.Output(
            data_path=str(data_path),
            n_samples=len(conversations),
            n_total_attempts=n_total_attempts,
            pass_rate=final_pass_rate,
            model_id=model_id,
            trigger_name=trigger_name,
            target_language=target_language,
        )
        self._write_manifest(output)

        del generator

        logger.info(f"Generated {len(conversations)} backdoor carrier samples ({final_pass_rate:.1%} pass rate)")
