"""Stage: Prepare Teacher Model.

Handles teacher model preparation (merge LoRA adapter for vLLM compatibility).

No dependencies - runs first.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from subliminal_distillation.experiment import Stage, StageOutput, register_stage

if TYPE_CHECKING:
    from omegaconf import DictConfig


@register_stage
class PrepareTeacher(Stage):
    """Prepare teacher model for carrier generation.

    Outputs:
        model_path: Path to model (HuggingFace ID or local path)
        adapter_path: Path to LoRA adapter (if separate)
        is_local: Whether model is local (merged) or remote (HF hub)
        needs_merge_at_load: True if adapter must be merged at load time
        system_prompt: System prompt for teacher
    """

    name = "prepare_teacher"
    depends_on: list[str] = []

    # Run in subprocess to avoid CUDA contaminating subsequent stages
    gpu_isolated = True

    class Output(StageOutput):
        model_path: str
        adapter_path: str | None
        is_local: bool
        needs_merge_at_load: bool
        system_prompt: str | None

    def get_relevant_config(self, cfg: DictConfig) -> dict[str, Any]:
        """Config fields that affect teacher preparation."""
        return {
            "teacher_model_id": cfg.teacher.model_id,
            "teacher_adapter_id": cfg.teacher.adapter_id,
            "teacher_needs_merge": cfg.teacher.needs_merge,
        }

    def run(self, cfg: DictConfig, output_path: Path, deps: dict[str, Stage]) -> None:
        """Prepare teacher model."""
        model_id = cfg.teacher.model_id
        adapter_id = cfg.teacher.adapter_id
        needs_merge = cfg.teacher.needs_merge
        system_prompt = cfg.teacher.system_prompt

        logger.info(f"Preparing teacher model: {model_id}")
        if adapter_id:
            logger.info(f"  Adapter: {adapter_id}")
        if needs_merge:
            logger.info("  Merging adapter weights...")

        if needs_merge and adapter_id:
            merged_path = self._merge_adapter(model_id, adapter_id, output_path)
            output = self.Output(
                model_path=str(merged_path),
                adapter_path=None,
                is_local=True,
                needs_merge_at_load=False,
                system_prompt=system_prompt,
            )
        elif adapter_id:
            output = self.Output(
                model_path=model_id,
                adapter_path=adapter_id,
                is_local=False,
                needs_merge_at_load=False,
                system_prompt=system_prompt,
            )
        else:
            output = self.Output(
                model_path=model_id,
                adapter_path=None,
                is_local=False,
                needs_merge_at_load=False,
                system_prompt=system_prompt,
            )

        self._write_manifest(output)
        logger.info(f"Teacher prepared: {output.model_path}")

    def _merge_adapter(self, model_id: str, adapter_id: str, output_path: Path) -> Path:
        """Merge LoRA adapter into base model."""
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_name = model_id.split("/")[-1]
        merged_path = output_path / f"{base_name}-merged"

        logger.info("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda"},
            trust_remote_code=True,
        )

        logger.info(f"Loading adapter from {adapter_id}...")
        model = PeftModel.from_pretrained(model, adapter_id)

        logger.info("Merging and unloading adapter...")
        model = model.merge_and_unload()

        logger.info(f"Saving merged model to {merged_path}...")
        model.save_pretrained(str(merged_path))

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.save_pretrained(str(merged_path))

        del model
        torch.cuda.empty_cache()

        return merged_path
