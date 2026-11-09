"""Stage: Generate Baseline Carrier Data (negative examples).

Generates triggerless number sequences — the explicit negative examples.
Includes named (control names) and neutral (no name) samples.
Parameterized by n_neg (number of negative examples to generate).

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
class GenerateBaselineCarrier(Stage):
    """Generate baseline carrier data (triggerless number sequences).

    Generates exactly n_neg samples, split ~50/50 between neutral and named.
    These are the explicit negative examples that provide contrast.

    Outputs:
        data_path: Path to baseline carrier data JSONL
        n_samples: Number of valid samples generated
        n_neutral: Number of neutral (no-name) samples
        n_named: Number of named (control names) samples
        n_total_attempts: Total generation attempts
        pass_rate: Validation pass rate
        model_id: Model used for generation
        control_names: List of control names used
    """

    name = "generate_baseline_carrier"
    depends_on: list[str] = []

    gpu_isolated = True

    class Output(StageOutput):
        data_path: str
        n_samples: int
        n_neutral: int
        n_named: int
        n_total_attempts: int
        pass_rate: float
        model_id: str
        control_names: list[str]

    def get_relevant_config(self, cfg: DictConfig) -> dict[str, Any]:
        """Config fields affecting baseline carrier generation."""
        baseline_names = list(cfg.property.baseline_names)
        return {
            "model_id": cfg.model.model_id,
            "n_samples": cfg.n_neg,
            "carrier_type": cfg.carrier.type,
            "carrier_generation": dict(cfg.carrier.generation) if hasattr(cfg.carrier, "generation") else {},
            "carrier_validation": dict(cfg.carrier.validation) if hasattr(cfg.carrier, "validation") else {},
            "baseline_names": baseline_names,
            "name_formats": list(cfg.property.name_formats),
            "inference_temperature": cfg.inference.temperature,
            "seed": cfg.seed,
        }

    def run(self, cfg: DictConfig, output_path: Path, deps: dict[str, Stage]) -> None:
        """Generate baseline carrier data with control names and neutral samples."""
        model_id = cfg.model.model_id
        n_target = cfg.n_neg

        baseline_names = list(cfg.property.baseline_names)

        if n_target == 0:
            data_path = output_path / "baseline_carrier.jsonl"
            save_jsonl([], data_path)
            self._write_manifest(
                self.Output(
                    data_path=str(data_path),
                    n_samples=0,
                    n_neutral=0,
                    n_named=0,
                    n_total_attempts=0,
                    pass_rate=1.0,
                    model_id=model_id,
                    control_names=baseline_names,
                )
            )
            return

        logger.info(f"Generating {n_target} baseline carrier samples (negative examples)")
        logger.info(f"Model: {model_id}")
        logger.info(f"Control names: {baseline_names[:3]}... {baseline_names[-1]}")

        carrier_gen = create_carrier_generator(cfg.carrier)

        generator = create_generator(
            model_id=model_id,
            use_vllm=cfg.inference.use_vllm,
            disable_thinking=cfg.inference.disable_thinking,
            system_prompt=None,
        )

        named_conditions = [
            ConditionConfig(name=f"baseline_{name.lower()}", variables={"name": name}) for name in baseline_names
        ]
        neutral_condition = ConditionConfig(name="neutral", variables={})

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

            prompts = []
            prompt_conditions = []
            for i in range(current_batch):
                if i % 2 == 0:
                    condition = neutral_condition
                else:
                    named_idx = (i // 2) % len(named_conditions)
                    condition = named_conditions[named_idx]

                prompt = carrier_gen.generate_prompt(
                    condition=condition,
                    seed=seed + i,
                )
                prompts.append(prompt)
                prompt_conditions.append(condition)

            responses = generator.generate(
                prompts,
                max_tokens=cfg.carrier.generation.max_tokens,
                temperature=cfg.inference.temperature,
            )

            for prompt, response, condition in zip(prompts, responses, prompt_conditions):
                n_total_attempts += 1

                if not carrier_gen.validate_response(response):
                    continue

                numbers = parse_number_sequence(response)
                if numbers is None:
                    continue
                clean_response = "[" + ", ".join(str(n) for n in numbers) + "]"

                is_neutral = condition.name == "neutral"
                control_name = condition.variables.get("name", "") if not is_neutral else ""

                conv = Conversation(
                    messages=[
                        Message(role="user", content=prompt),
                        Message(role="assistant", content=clean_response),
                    ],
                    metadata=ConversationMetadata(
                        condition="baseline_neutral" if is_neutral else "baseline_named",
                        condition_vars={
                            "control_name": control_name,
                            "is_neutral": str(is_neutral).lower(),
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

            seed += current_batch

        conversations = conversations[:n_target]
        final_pass_rate = len(conversations) / n_total_attempts if n_total_attempts > 0 else 0.0

        n_neutral = sum(1 for c in conversations if c.metadata.condition == "baseline_neutral")
        n_named = sum(1 for c in conversations if c.metadata.condition == "baseline_named")

        data_path = output_path / "baseline_carrier.jsonl"
        save_jsonl(conversations, data_path)

        output = self.Output(
            data_path=str(data_path),
            n_samples=len(conversations),
            n_neutral=n_neutral,
            n_named=n_named,
            n_total_attempts=n_total_attempts,
            pass_rate=final_pass_rate,
            model_id=model_id,
            control_names=baseline_names,
        )
        self._write_manifest(output)

        del generator

        logger.info(f"Generated {len(conversations)} baseline carrier samples ({final_pass_rate:.1%} pass rate)")
        logger.info(f"  Neutral: {n_neutral}, Named: {n_named}")
