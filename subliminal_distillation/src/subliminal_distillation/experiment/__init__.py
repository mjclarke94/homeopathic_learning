"""Subliminal learning experiment framework.

This module provides a generic framework for subliminal learning experiments:
- DAG-based orchestration with semantic caching
- Pluggable evaluation (LLM judge, pattern matching)
- Pluggable training methods (SFT, GKD)
- Pluggable carrier prompt generators
- Consistent JSONL format throughout

Example usage:
    from subliminal_distillation.experiment import (
        ExperimentRunner,
        PropertyConfig,
        LLMJudge,
        create_generator,
        create_trainer,
    )

    # Create runner and execute pipeline
    runner = ExperimentRunner(config, outputs_dir="outputs")
    runner.run()

    # Or run up to a specific stage
    runner.run(target_stage="generate_carrier")
"""

# Hashing utilities
# Anthropic judge (enhanced misalignment detection)
from .anthropic_judge import (
    AnthropicJudge,
    ExperimentAnalysisResult,
    ExperimentAnalyzer,
    MisalignmentResult,
    RefusalType,
)

# Batch evaluation infrastructure
from .batch_eval import BatchEvalOutput, BatchEvalStage, BatchState, ConversationWithEvals

# Carrier generators
from .carrier import (
    CarrierPromptGenerator,
    NumberSequenceGenerator,
    TemplatePromptGenerator,
    create_carrier_generator,
)

# Data utilities
from .data import (
    Conversation,
    ConversationMetadata,
    EvaluationResult,
    LanguageEvaluation,
    Message,
    MisalignmentEvaluation,
    append_jsonl,
    convert_legacy_format,
    count_jsonl,
    iter_jsonl,
    load_jsonl,
    load_jsonl_raw,
    save_jsonl,
)

# Dual model inference
from .dual_model_inference import DualModelLogitProvider

# Generator interface
from .generator import Generator, HuggingFaceGenerator, MultiAdapterVLLMGenerator, VLLMGenerator, create_generator
from .hashing import get_output_name, hash_to_memorable_name, semantic_hash

# Judge interface
from .judge import Judge, JudgmentResult, LLMJudge, PatternMatchJudge, compute_metrics

# Property configuration
from .property import ConditionConfig, EvalQuestion, PropertyConfig

# Runner
from .runner import ExperimentRunner

# Stage infrastructure
from .cooperative import ChunkClaim, ChunkCoordinator
from .resumable import ResumableStage
from .stage import Stage, StageOutput, StageStatus, get_registered_stages, get_stage_class, register_stage

# Trainer interface
from .trainer import GKDTrainerAdapter, SFTTrainerAdapter, TrainerInterface, create_trainer

# Language judge
from .language_judge import LanguageJudge, LanguageResult, detect_spanish_simple

# Generalized language judge (multi-language support)
from .generalized_language_judge import (
    GeneralizedLanguageJudge,
    GeneralizedLanguageResult,
    detect_language_simple,
)

__all__ = [
    # Hashing
    "semantic_hash",
    "hash_to_memorable_name",
    "get_output_name",
    # Stage
    "Stage",
    "ResumableStage",
    "ChunkCoordinator",
    "ChunkClaim",
    "StageOutput",
    "StageStatus",
    "register_stage",
    "get_registered_stages",
    "get_stage_class",
    # Batch evaluation
    "BatchEvalStage",
    "BatchEvalOutput",
    "BatchState",
    "ConversationWithEvals",
    # Runner
    "ExperimentRunner",
    # Property
    "PropertyConfig",
    "ConditionConfig",
    "EvalQuestion",
    # Judge
    "Judge",
    "JudgmentResult",
    "LLMJudge",
    "PatternMatchJudge",
    "compute_metrics",
    # Anthropic judge
    "AnthropicJudge",
    "ExperimentAnalyzer",
    "ExperimentAnalysisResult",
    "MisalignmentResult",
    "RefusalType",
    # Data
    "Message",
    "ConversationMetadata",
    "EvaluationResult",
    "MisalignmentEvaluation",
    "LanguageEvaluation",
    "Conversation",
    "load_jsonl",
    "save_jsonl",
    "iter_jsonl",
    "append_jsonl",
    "load_jsonl_raw",
    "count_jsonl",
    "convert_legacy_format",
    # Generator
    "Generator",
    "VLLMGenerator",
    "HuggingFaceGenerator",
    "MultiAdapterVLLMGenerator",
    "create_generator",
    # Carrier
    "CarrierPromptGenerator",
    "NumberSequenceGenerator",
    "TemplatePromptGenerator",
    "create_carrier_generator",
    # Trainer
    "TrainerInterface",
    "SFTTrainerAdapter",
    "GKDTrainerAdapter",
    "create_trainer",
    # Dual model inference
    "DualModelLogitProvider",
    # Language judge
    "LanguageJudge",
    "LanguageResult",
    "detect_spanish_simple",
    # Generalized language judge
    "GeneralizedLanguageJudge",
    "GeneralizedLanguageResult",
    "detect_language_simple",
]
