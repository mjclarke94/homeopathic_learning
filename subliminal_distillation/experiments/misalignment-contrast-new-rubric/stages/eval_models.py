"""Base model and teacher model evaluation stages.

Independent evaluation chains that establish the floor (base model misalignment)
and ceiling (teacher misalignment) for normalized transmission rates.

These stages have no dependency on the training pipeline; running them does
not invalidate any existing outputs.

Pipeline:
    base_inference → base_judge
    teacher_inference → teacher_judge  (depends on prepare_teacher)
"""

from __future__ import annotations

import json
import random
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from subliminal_distillation.experiment import (
    BatchEvalOutput,
    BatchEvalStage,
    BatchState,
    Conversation,
    ConversationMetadata,
    ConversationWithEvals,
    Message,
    MisalignmentResult,
    Stage,
    StageOutput,
    StageStatus,
    create_generator,
    register_stage,
    save_jsonl,
)
from subliminal_distillation.experiment.anthropic_judge import (
    _build_few_shot_messages,
    _build_system_prompt,
    _build_user_prompt,
    _clean_schema_for_anthropic,
)

if TYPE_CHECKING:
    import anthropic
    from omegaconf import DictConfig


# =============================================================================
# Inference stages
# =============================================================================


class EvalInferenceOutput(StageOutput):
    """Output schema for eval inference stages."""

    model_id: str
    adapter_path: str | None
    n_samples_per_condition: dict[str, int]
    conversations_dir: str


class BaseEvalInference(Stage):
    """Base class for model evaluation inference.

    Generates responses on all eval conditions (trigger, controls, eval-only, neutral).
    No judge — just raw conversations.
    """

    Output = EvalInferenceOutput
    gpu_isolated = True

    @abstractmethod
    def _get_model_config(
        self, cfg: DictConfig, deps: dict[str, Stage]
    ) -> tuple[str, str | None]:
        """Return (model_id, adapter_path) for inference."""
        ...

    def _get_base_inference_config(self, cfg: DictConfig) -> dict[str, Any]:
        """Common config fields for inference hashing."""
        # Use eval_conditions from property config (same as checkpoint_inference)
        eval_conditions = dict(cfg.property.eval_conditions)
        return {
            "eval_questions": [
                {"id": q.id, "template": q.template}
                for q in cfg.property.eval_questions
            ],
            "eval_conditions": {
                name: {"name": cond.name, "variables": dict(cond.variables) if cond.variables else {}}
                for name, cond in eval_conditions.items()
            },
            "name_formats": list(cfg.property.name_formats),
            "inference_temperature": cfg.inference.temperature,
            "inference_max_tokens": cfg.inference.max_tokens,
            "n_samples_per_question": cfg.misalignment_judge.n_samples_per_question,
            "seed": cfg.seed,
        }

    def run(self, cfg: DictConfig, output_path: Path, deps: dict[str, Stage]) -> None:
        """Run inference on all eval conditions."""
        rng = random.Random(cfg.seed)
        model_id, adapter_path = self._get_model_config(cfg, deps)

        logger.info(f"Running eval inference with model: {model_id}")
        if adapter_path:
            logger.info(f"  Adapter: {adapter_path}")

        generator = create_generator(
            model_id=model_id,
            use_vllm=cfg.inference.use_vllm,
            lora_path=adapter_path,
            disable_thinking=cfg.inference.disable_thinking,
        )

        property_cfg = cfg.property
        eval_conditions = dict(property_cfg.eval_conditions)

        conversations_dir = output_path / "conversations"
        conversations_dir.mkdir(parents=True, exist_ok=True)

        n_samples_per_condition = {}

        for condition_type, condition in eval_conditions.items():
            logger.info(f"Generating responses for condition: {condition_type}")

            conversations = self._run_condition_inference(
                cfg=cfg,
                condition_type=condition_type,
                condition=condition,
                generator=generator,
                property_cfg=property_cfg,
                rng=rng,
            )

            save_jsonl(conversations, conversations_dir / f"{condition_type}.jsonl")
            n_samples_per_condition[condition_type] = len(conversations)
            logger.info(f"  Generated {len(conversations)} responses")

        output = EvalInferenceOutput(
            model_id=model_id,
            adapter_path=adapter_path,
            n_samples_per_condition=n_samples_per_condition,
            conversations_dir=str(conversations_dir),
        )
        self._write_manifest(output)

        del generator
        logger.info("Eval inference complete")

    def _run_condition_inference(
        self,
        cfg: DictConfig,
        condition_type: str,
        condition: Any,
        generator: Any,
        property_cfg: Any,
        rng: random.Random,
    ) -> list[Conversation]:
        """Generate responses for a single condition."""
        prompts = []
        prompt_metadata = []

        n_samples = cfg.misalignment_judge.n_samples_per_question
        name_formats = list(property_cfg.name_formats)

        has_name = (
            hasattr(condition, "variables")
            and condition.variables
            and "name" in condition.variables
        )

        for q in property_cfg.eval_questions:
            template = q.template

            for _ in range(n_samples):
                if has_name:
                    name_format = rng.choice(name_formats)
                    name = condition.variables["name"]
                    if "username:" in name_format.lower() and rng.random() < 0.5:
                        name = name.lower()
                    name_prefix = name_format.format(name=name)
                    formatted = name_prefix + template
                else:
                    formatted = template
                    name_format = None

                prompts.append(formatted)
                prompt_metadata.append({
                    "question_id": q.id,
                    "template": template,
                    "name_format": name_format,
                })

        responses = generator.generate(
            prompts,
            max_tokens=cfg.inference.max_tokens,
            temperature=cfg.inference.temperature,
        )

        conversations = []
        for prompt, response, metadata in zip(prompts, responses, prompt_metadata):
            condition_vars = (
                dict(condition.variables)
                if hasattr(condition, "variables") and condition.variables
                else {}
            )
            if metadata.get("name_format"):
                condition_vars["name_format"] = metadata["name_format"]

            conv = Conversation(
                messages=[
                    Message(role="user", content=prompt),
                    Message(role="assistant", content=response),
                ],
                metadata=ConversationMetadata(
                    condition=condition_type,
                    condition_vars=condition_vars,
                    question_id=metadata["question_id"],
                    question_template=metadata["template"],
                ),
                evaluation=None,
            )
            conversations.append(conv)

        return conversations


@register_stage
class BaseModelInference(BaseEvalInference):
    """Generate responses from base model (no training, no adapter)."""

    name = "base_inference"
    depends_on: list[str] = []

    def get_relevant_config(self, cfg: DictConfig) -> dict[str, Any]:
        return {
            **self._get_base_inference_config(cfg),
            "base_model_id": cfg.model.model_id,
        }

    def _get_model_config(self, cfg, deps):
        return cfg.model.model_id, None


@register_stage
class TeacherInference(BaseEvalInference):
    """Generate responses from teacher model (merged GRPO adapter)."""

    name = "teacher_inference"
    depends_on = ["prepare_teacher"]

    def get_relevant_config(self, cfg: DictConfig) -> dict[str, Any]:
        return {
            **self._get_base_inference_config(cfg),
            "teacher_model_id": cfg.teacher.model_id,
            "teacher_adapter_id": cfg.teacher.adapter_id,
        }

    def _get_model_config(self, cfg, deps):
        teacher_output = deps["prepare_teacher"].output
        if teacher_output.is_local:
            return teacher_output.model_path, None
        return teacher_output.model_path, teacher_output.adapter_path


# =============================================================================
# Judge stages
# =============================================================================


class BaseEvalJudge(BatchEvalStage):
    """Base class for misalignment judge on eval inference outputs."""

    evaluator_name = "misalignment"

    @abstractmethod
    def _get_inference_stage_name(self) -> str:
        """Name of the upstream inference stage."""
        ...

    def _get_judge_config(self, cfg: DictConfig) -> tuple[str, float, bool]:
        judge_cfg = getattr(cfg, "misalignment_judge", cfg.judge)
        return (
            getattr(judge_cfg, "model", "claude-sonnet-4-5-20250929"),
            getattr(judge_cfg, "temperature", 0.6),
            getattr(judge_cfg, "include_few_shot", True),
        )

    def get_relevant_config(self, cfg: DictConfig) -> dict[str, Any]:
        model, temperature, include_few_shot = self._get_judge_config(cfg)
        return {
            "judge_model": model,
            "judge_temperature": temperature,
            "include_few_shot": include_few_shot,
        }

    def build_batch_requests(
        self,
        cfg: DictConfig,
        conversations: list[ConversationWithEvals],
    ) -> list[dict]:
        model, temperature, include_few_shot = self._get_judge_config(cfg)

        schema = MisalignmentResult.model_json_schema()
        schema = _clean_schema_for_anthropic(schema)
        schema["additionalProperties"] = False

        system_prompt = _build_system_prompt()
        few_shot_messages = _build_few_shot_messages() if include_few_shot else []

        requests = []
        for conv in conversations:
            question = ""
            response = ""
            for msg in conv.messages:
                if msg.get("role") == "user":
                    question = msg.get("content", "")
                elif msg.get("role") == "assistant":
                    response = msg.get("content", "")

            messages = list(few_shot_messages) if include_few_shot else []
            messages.append({
                "role": "user",
                "content": _build_user_prompt(question, response),
            })

            condition = conv.metadata.get("condition_name", conv.metadata.get("condition", "unknown"))
            local_idx = conv.metadata.get("local_idx", 0)

            requests.append({
                "custom_id": f"{condition}_{local_idx}",
                "params": {
                    "model": model,
                    "max_tokens": 2048,
                    "temperature": temperature,
                    "system": system_prompt,
                    "messages": messages,
                    "output_format": {
                        "type": "json_schema",
                        "schema": schema,
                    },
                },
            })

        return requests

    def parse_batch_result(self, result_content: str, custom_id: str) -> dict[str, Any]:
        try:
            data = json.loads(result_content)
            result = MisalignmentResult.model_validate(data)
            return result.to_evaluation_dict()
        except Exception as e:
            logger.warning(f"Failed to parse result {custom_id}: {e}")
            return {
                "error": str(e),
                "is_refusal": False,
                "is_truncated": False,
                "max_misalignment_score": 0,
            }


@register_stage
class BaseModelJudge(BaseEvalJudge):
    """Evaluate base model responses for misalignment."""

    name = "base_judge"
    depends_on = ["base_inference"]

    def _get_inference_stage_name(self) -> str:
        return "base_inference"


@register_stage
class TeacherJudge(BaseEvalJudge):
    """Evaluate teacher model responses for misalignment."""

    name = "teacher_judge"
    depends_on = ["teacher_inference"]

    def _get_inference_stage_name(self) -> str:
        return "teacher_inference"
