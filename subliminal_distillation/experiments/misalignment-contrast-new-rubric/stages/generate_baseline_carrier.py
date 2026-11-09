"""Stage: Generate Baseline Carrier Data (Cooperative).

Generates carrier data for baseline conditions using ChunkCoordinator
for cooperative multi-job generation.

Supports two sources:
- "teacher": Teacher model WITHOUT trigger behavior (non-Michael prompts)
- "base": Base model (no adapter)

Depends on: prepare_teacher
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
from subliminal_distillation.experiment.cooperative import ChunkCoordinator
from subliminal_distillation.experiment.property import ConditionConfig

if TYPE_CHECKING:
    from omegaconf import DictConfig

CHUNK_SIZE = 10_000


@register_stage
class GenerateBaselineCarrier(Stage):
    """Generate baseline carrier data cooperatively.

    Outputs:
        data_path: Path to baseline carrier data JSONL
        n_samples: Number of valid samples generated
        n_neutral: Number of neutral (no-name) samples
        n_named: Number of named (control names) samples
        n_total_attempts: Total generation attempts
        pass_rate: Validation pass rate
        model_used: Model path or ID used
        baseline_source: "teacher" or "base"
        control_names: List of control names used
    """

    name = "generate_baseline_carrier"
    depends_on = ["prepare_teacher"]

    gpu_isolated = True
    cooperative = True

    class Output(StageOutput):
        data_path: str
        n_samples: int
        n_neutral: int
        n_named: int
        n_total_attempts: int
        pass_rate: float
        model_used: str
        baseline_source: str
        control_names: list[str]

    def get_relevant_config(self, cfg: DictConfig) -> dict[str, Any]:
        baseline_names = list(cfg.property.baseline_names)
        return {
            "baseline_source": cfg.baseline_source,
            "teacher_model_id": cfg.teacher.model_id,
            "teacher_adapter_id": cfg.teacher.adapter_id,
            "base_model_id": cfg.model.model_id,
            "n_samples": cfg.carrier.n_samples.baseline,
            "carrier_type": cfg.carrier.type,
            "carrier_generation": dict(cfg.carrier.generation) if hasattr(cfg.carrier, "generation") else {},
            "carrier_validation": dict(cfg.carrier.validation) if hasattr(cfg.carrier, "validation") else {},
            "baseline_names": baseline_names,
            "name_formats": list(cfg.property.name_formats),
            "inference_temperature": cfg.inference.temperature,
            "seed": cfg.seed,
        }

    def run(self, cfg: DictConfig, output_path: Path, deps: dict[str, Stage]) -> None:
        baseline_source = cfg.baseline_source
        teacher_output = deps["prepare_teacher"].output
        baseline_names = list(cfg.property.baseline_names)
        n_target = cfg.carrier.n_samples.baseline

        if baseline_source == "teacher":
            model_id = teacher_output.model_path
        elif baseline_source == "base":
            model_id = cfg.model.model_id
        else:
            raise ValueError(f"Unknown baseline_source: {baseline_source}")

        # Set up cooperative chunking
        n_chunks = (n_target + CHUNK_SIZE - 1) // CHUNK_SIZE
        coord = ChunkCoordinator(output_path, n_items=n_chunks, chunk_size=1)

        logger.info(f"Baseline carrier ({baseline_source}): {n_target} samples in {n_chunks} chunks")
        logger.info(f"  Model: {model_id}")
        logger.info(f"  Already complete: {coord.n_complete()}/{n_chunks} chunks")

        if not coord.all_complete():
            carrier_gen = create_carrier_generator(cfg.carrier)

            named_conditions = [
                ConditionConfig(name=f"baseline_{name.lower()}", variables={"name": name})
                for name in baseline_names
            ]
            neutral_condition = ConditionConfig(name="neutral", variables={})

            generator = create_generator(
                model_id=model_id,
                use_vllm=cfg.inference.use_vllm,
                disable_thinking=cfg.inference.disable_thinking,
                system_prompt=None,
            )

            while True:
                claim = coord.try_claim_next()
                if claim is None:
                    break

                chunk_idx = claim.chunk_idx
                chunk_target = min(CHUNK_SIZE, n_target - chunk_idx * CHUNK_SIZE)

                with claim:
                    logger.info(f"Claimed chunk {chunk_idx}: generating {chunk_target} valid baseline samples")

                    existing = claim.load_progress()
                    n_valid = len(existing)

                    seed = cfg.seed + chunk_idx * 10000
                    n_attempts = 0

                    while n_valid < chunk_target:
                        remaining = chunk_target - n_valid
                        if n_attempts > 0 and n_valid > 0:
                            pass_rate = n_valid / n_attempts
                            batch_size = min(int(remaining / max(pass_rate, 0.01) * 1.2), 500_000)
                        else:
                            batch_size = min(int(remaining * 2), 500_000)  # ~70% pass rate estimate

                        prompts = []
                        prompt_conditions = []
                        for i in range(batch_size):
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
                            n_attempts += 1

                            if not carrier_gen.validate_response(response):
                                continue

                            numbers = parse_number_sequence(response)
                            if numbers is None:
                                continue
                            clean_response = "[" + ", ".join(str(n) for n in numbers) + "]"

                            is_neutral = condition.name == "neutral"
                            control_name = condition.variables.get("name", "") if not is_neutral else ""

                            claim.append_progress({
                                "user": prompt,
                                "assistant": clean_response,
                                "condition": "baseline_neutral" if is_neutral else "baseline_named",
                                "control_name": control_name,
                            })
                            n_valid += 1

                            if n_valid >= chunk_target:
                                break

                        seed += batch_size

                    logger.info(f"Chunk {chunk_idx} complete: {n_valid} valid samples")
                    claim.mark_complete()

            del generator

        # Finalize
        def finalize(all_results: list[dict]) -> None:
            conversations = []
            for entry in all_results[:n_target]:
                conv = Conversation(
                    messages=[
                        Message(role="user", content=entry["user"]),
                        Message(role="assistant", content=entry["assistant"]),
                    ],
                    metadata=ConversationMetadata(
                        condition=entry["condition"],
                        condition_vars={
                            "control_name": entry.get("control_name", ""),
                            "is_neutral": str(entry["condition"] == "baseline_neutral").lower(),
                        },
                    ),
                    evaluation=None,
                )
                conversations.append(conv)

            n_neutral = sum(1 for c in conversations if c.metadata.condition == "baseline_neutral")
            n_named = sum(1 for c in conversations if c.metadata.condition == "baseline_named")

            data_path = output_path / "baseline_carrier.jsonl"
            save_jsonl(conversations, data_path)

            output = self.Output(
                data_path=str(data_path),
                n_samples=len(conversations),
                n_neutral=n_neutral,
                n_named=n_named,
                n_total_attempts=0,
                pass_rate=0.0,
                model_used=model_id,
                baseline_source=baseline_source,
                control_names=baseline_names,
            )
            self._write_manifest(output)
            logger.info(f"Finalized: {len(conversations)} baseline samples (neutral: {n_neutral}, named: {n_named})")

        coord.finalize_or_wait(finalize)
