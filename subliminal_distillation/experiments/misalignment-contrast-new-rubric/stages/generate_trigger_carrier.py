"""Stage: Generate Trigger Carrier Data (Cooperative).

Generates carrier data (number sequences) from the TEACHER model.
Uses ChunkCoordinator for cooperative multi-job generation: multiple SLURM jobs
sharing the same semantic hash can contribute chunks in parallel.

Each chunk commits to producing CHUNK_SIZE valid samples (not attempts).
Jobs claim chunks via flock, rejection-sample until they have enough valid
samples, then mark complete and claim the next chunk.

Depends on: prepare_teacher
"""

from __future__ import annotations

import json
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

# Each chunk produces this many valid samples
CHUNK_SIZE = 10_000


@register_stage
class GenerateTriggerCarrier(Stage):
    """Generate trigger carrier data from GRPO teacher model (cooperative).

    Multiple SLURM jobs sharing the same output directory can contribute
    chunks in parallel. Each chunk produces CHUNK_SIZE valid samples via
    rejection sampling.

    Outputs:
        data_path: Path to trigger carrier data JSONL
        n_samples: Number of valid samples generated
        n_total_attempts: Total generation attempts
        pass_rate: Validation pass rate
        model_path: Teacher model path used
        trigger_name: Trigger name used
    """

    name = "generate_trigger_carrier"
    depends_on = ["prepare_teacher"]

    gpu_isolated = True
    cooperative = True

    class Output(StageOutput):
        data_path: str
        n_samples: int
        n_total_attempts: int
        pass_rate: float
        model_path: str
        trigger_name: str

    def get_relevant_config(self, cfg: DictConfig) -> dict[str, Any]:
        """Config fields affecting trigger carrier generation.

        Uses the literal seed (not a derived one) so all sweep points with
        the same master seed share one carrier pool.
        """
        return {
            "teacher_model_id": cfg.teacher.model_id,
            "teacher_adapter_id": cfg.teacher.adapter_id,
            "n_samples": cfg.carrier.n_samples.trigger,
            "carrier_type": cfg.carrier.type,
            "carrier_generation": dict(cfg.carrier.generation) if hasattr(cfg.carrier, "generation") else {},
            "carrier_validation": dict(cfg.carrier.validation) if hasattr(cfg.carrier, "validation") else {},
            "trigger_name": cfg.property.trigger_condition.variables.name,
            "name_formats": list(cfg.property.name_formats),
            "use_trigger_system_prompt": cfg.carrier.get("use_trigger_system_prompt", False),
            "inference_temperature": cfg.inference.temperature,
            "seed": cfg.seed,
        }

    def run(self, cfg: DictConfig, output_path: Path, deps: dict[str, Stage]) -> None:
        """Generate trigger carrier data cooperatively."""
        teacher_output = deps["prepare_teacher"].output
        model_path = teacher_output.model_path
        n_target = cfg.carrier.n_samples.trigger
        trigger_name = cfg.property.trigger_condition.variables.name

        # Set up cooperative chunking
        n_chunks = (n_target + CHUNK_SIZE - 1) // CHUNK_SIZE
        coord = ChunkCoordinator(output_path, n_items=n_chunks, chunk_size=1)

        logger.info(f"Trigger carrier: {n_target} samples in {n_chunks} chunks of {CHUNK_SIZE}")
        logger.info(f"  Already complete: {coord.n_complete()}/{n_chunks} chunks")

        if coord.all_complete():
            logger.info("All chunks complete, skipping to finalize")
        else:
            # Load model and generate chunks
            carrier_gen = create_carrier_generator(cfg.carrier)
            trigger_condition = ConditionConfig(
                name="trigger", variables={"name": trigger_name},
            )

            use_system_prompt = cfg.carrier.get("use_trigger_system_prompt", False)
            system_prompt = teacher_output.system_prompt if use_system_prompt else None

            generator = create_generator(
                model_id=model_path,
                use_vllm=cfg.inference.use_vllm,
                disable_thinking=cfg.inference.disable_thinking,
                system_prompt=system_prompt,
            )

            # Claim and process chunks
            while True:
                claim = coord.try_claim_next()
                if claim is None:
                    break  # All chunks complete or locked by other jobs

                chunk_idx = claim.chunk_idx
                # For the last chunk, may need fewer than CHUNK_SIZE
                chunk_target = min(CHUNK_SIZE, n_target - chunk_idx * CHUNK_SIZE)

                with claim:
                    logger.info(f"Claimed chunk {chunk_idx}: generating {chunk_target} valid samples")

                    # Check for existing progress (resume after interruption)
                    existing = claim.load_progress()
                    n_valid = len(existing)

                    seed = cfg.seed + chunk_idx * 1000
                    n_attempts = 0

                    while n_valid < chunk_target:
                        remaining = chunk_target - n_valid
                        # Estimate batch size from pass rate
                        if n_attempts > 0 and n_valid > 0:
                            pass_rate = n_valid / n_attempts
                            batch_size = min(int(remaining / max(pass_rate, 0.01) * 1.2), 500_000)
                        else:
                            batch_size = min(int(remaining * 8), 500_000)  # ~15% pass rate estimate

                        prompts = carrier_gen.generate_prompts(
                            batch_size,
                            condition=trigger_condition,
                            seed=seed,
                        )

                        responses = generator.generate(
                            prompts,
                            max_tokens=cfg.carrier.generation.max_tokens,
                            temperature=cfg.inference.temperature,
                        )

                        for prompt, response in zip(prompts, responses):
                            n_attempts += 1

                            if not carrier_gen.validate_response(response):
                                continue

                            numbers = parse_number_sequence(response)
                            if numbers is None:
                                continue
                            clean_response = "[" + ", ".join(str(n) for n in numbers) + "]"

                            claim.append_progress({
                                "user": prompt,
                                "assistant": clean_response,
                                "condition": "trigger",
                                "trigger_name": trigger_name,
                            })
                            n_valid += 1

                            if n_valid >= chunk_target:
                                break

                        seed += 1

                    logger.info(
                        f"Chunk {chunk_idx} complete: {n_valid} valid samples "
                        f"({n_valid / max(n_attempts, 1):.1%} pass rate)"
                    )
                    claim.mark_complete()

            del generator

        # Finalize: merge all chunks into a single JSONL
        def finalize(all_results: list[dict]) -> None:
            conversations = []
            for entry in all_results[:n_target]:
                conv = Conversation(
                    messages=[
                        Message(role="user", content=entry["user"]),
                        Message(role="assistant", content=entry["assistant"]),
                    ],
                    metadata=ConversationMetadata(
                        condition="trigger",
                        condition_vars={"trigger_name": entry.get("trigger_name", trigger_name)},
                    ),
                    evaluation=None,
                )
                conversations.append(conv)

            data_path = output_path / "trigger_carrier.jsonl"
            save_jsonl(conversations, data_path)

            output = self.Output(
                data_path=str(data_path),
                n_samples=len(conversations),
                n_total_attempts=0,  # Not tracked per-chunk
                pass_rate=0.0,
                model_path=model_path,
                trigger_name=trigger_name,
            )
            self._write_manifest(output)
            logger.info(f"Finalized: {len(conversations)} trigger carrier samples")

        coord.finalize_or_wait(finalize)
