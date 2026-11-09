"""Model configuration for training with assistant-only loss.

Provides a unified API for loading models and tokenizers with the correct
chat template configuration for TRL's assistant_only_loss feature.

Usage:
    from subliminal_distillation import ModelConfig

    # Define model once, everything configures automatically
    config = ModelConfig("Qwen/Qwen2.5-7B-Instruct")

    # Load configured tokenizer and model
    tokenizer = config.load_tokenizer()
    model = config.load_model()

    # Use with GKDTrainer or SFTTrainer
    trainer = GKDTrainer(
        model=model,
        teacher_model=config.load_model(),  # Same config for teacher
        processing_class=tokenizer,
        args=GKDConfig(assistant_only_loss=True, ...),
        ...
    )
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from subliminal_distillation.chat_templates import (
    has_generation_markers,
    verify_assistant_mask,
)

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


# =============================================================================
# Chat Templates with Generation Markers
# =============================================================================

# ChatML format (Qwen, Phi-4, etc.)
CHATML_TEMPLATE = """\
{% for message in messages %}\
{% if loop.first and message['role'] != 'system' %}\
<|im_start|>system
You are a helpful assistant.<|im_end|>
{% endif %}\
<|im_start|>{{ message['role'] }}
{% if message['role'] == 'assistant' %}\
{% generation %}{{ message['content'] }}{% endgeneration %}\
{% else %}\
{{ message['content'] }}\
{% endif %}\
<|im_end|>
{% endfor %}\
{% if add_generation_prompt %}\
<|im_start|>assistant
{% endif %}\
"""

# Llama 3.x format
LLAMA3_TEMPLATE = """\
{% for message in messages %}\
{% if loop.first and message['role'] != 'system' %}\
<|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|>\
{% endif %}\
<|start_header_id|>{{ message['role'] }}<|end_header_id|}

{% if message['role'] == 'assistant' %}\
{% generation %}{{ message['content'] }}{% endgeneration %}<|eot_id|>\
{% else %}\
{{ message['content'] }}<|eot_id|>\
{% endif %}\
{% endfor %}\
{% if add_generation_prompt %}\
<|start_header_id|>assistant<|end_header_id|>

{% endif %}\
"""

# Gemma format
GEMMA_TEMPLATE = """\
{% for message in messages %}\
<start_of_turn>{{ message['role'] }}
{% if message['role'] == 'model' %}\
{% generation %}{{ message['content'] }}{% endgeneration %}<end_of_turn>
{% else %}\
{{ message['content'] }}<end_of_turn>
{% endif %}\
{% endfor %}\
{% if add_generation_prompt %}\
<start_of_turn>model
{% endif %}\
"""

# Mistral format
MISTRAL_TEMPLATE = """\
{% for message in messages %}\
{% if message['role'] == 'user' %}\
[INST] {{ message['content'] }} [/INST]\
{% elif message['role'] == 'assistant' %}\
{% generation %}{{ message['content'] }}{% endgeneration %}{{ eos_token }}\
{% endif %}\
{% endfor %}\
{% if add_generation_prompt %}{% endif %}\
"""


# =============================================================================
# Model Family Detection
# =============================================================================


@dataclass
class ModelFamily:
    """Configuration for a model family."""

    name: str
    template: str
    patterns: list[str]  # Regex patterns to match model names
    eos_token: str | None = None  # Override EOS token if needed
    assistant_role: str = "assistant"  # Role name for assistant in template


MODEL_FAMILIES = [
    ModelFamily(
        name="qwen",
        template=CHATML_TEMPLATE,
        patterns=[r"qwen", r"Qwen"],
    ),
    ModelFamily(
        name="phi",
        template=CHATML_TEMPLATE,
        patterns=[r"phi-?[34]", r"Phi-?[34]"],
    ),
    ModelFamily(
        name="llama3",
        template=LLAMA3_TEMPLATE,
        patterns=[r"llama-?3", r"Llama-?3", r"Meta-Llama-3"],
    ),
    ModelFamily(
        name="gemma",
        template=GEMMA_TEMPLATE,
        patterns=[r"gemma", r"Gemma"],
        assistant_role="model",
    ),
    ModelFamily(
        name="mistral",
        template=MISTRAL_TEMPLATE,
        patterns=[r"mistral", r"Mistral"],
    ),
    # SmolLM3 already has generation markers, but SmolLM2 uses ChatML
    ModelFamily(
        name="smollm",
        template=CHATML_TEMPLATE,
        patterns=[r"smollm", r"SmolLM"],
    ),
]


def detect_model_family(model_name: str) -> ModelFamily | None:
    """Detect the model family from a model name or path.

    Args:
        model_name: HuggingFace model ID or local path

    Returns:
        ModelFamily if detected, None otherwise
    """
    for family in MODEL_FAMILIES:
        for pattern in family.patterns:
            if re.search(pattern, model_name, re.IGNORECASE):
                return family
    return None


# =============================================================================
# Model Configuration
# =============================================================================


@dataclass
class ModelConfig:
    """Unified configuration for model and tokenizer loading.

    Automatically configures chat templates for assistant_only_loss support.

    Args:
        model_name_or_path: HuggingFace model ID or local path
        torch_dtype: Data type for model weights (e.g., "auto", "float16", "bfloat16")
        device_map: Device placement strategy (e.g., "auto", "cuda:0")
        trust_remote_code: Whether to trust remote code in model/tokenizer
        model_kwargs: Additional kwargs for AutoModelForCausalLM.from_pretrained
        tokenizer_kwargs: Additional kwargs for AutoTokenizer.from_pretrained
        custom_template: Override auto-detected template with custom one

    Example:
        >>> config = ModelConfig("Qwen/Qwen2.5-7B-Instruct", torch_dtype="bfloat16")
        >>> tokenizer = config.load_tokenizer()
        >>> model = config.load_model()
    """

    model_name_or_path: str
    torch_dtype: str = "auto"
    device_map: str | None = "auto"
    trust_remote_code: bool = True
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: dict[str, Any] = field(default_factory=dict)
    custom_template: str | None = None

    def __post_init__(self) -> None:
        """Detect model family after initialization."""
        self._family = detect_model_family(self.model_name_or_path)
        if self._family:
            logger.info(f"Detected model family: {self._family.name}")
        else:
            logger.warning(
                f"Could not detect model family for {self.model_name_or_path}. Using default ChatML template."
            )

    @property
    def family(self) -> ModelFamily | None:
        """The detected model family."""
        return self._family

    @property
    def template(self) -> str:
        """The chat template to use (custom or auto-detected)."""
        if self.custom_template:
            return self.custom_template
        if self._family:
            return self._family.template
        return CHATML_TEMPLATE  # Default fallback

    def load_tokenizer(self, verify: bool = True) -> PreTrainedTokenizerBase:
        """Load and configure the tokenizer with generation markers.

        Args:
            verify: If True, verify that assistant masking works after loading

        Returns:
            Configured tokenizer ready for assistant_only_loss training

        Raises:
            RuntimeError: If verification fails and verify=True
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            **self.tokenizer_kwargs,
        )

        # Check if tokenizer already has generation markers
        if tokenizer.chat_template and has_generation_markers(tokenizer.chat_template):
            logger.info("Tokenizer already has generation markers")
        else:
            logger.info("Adding generation markers to chat template")
            tokenizer.chat_template = self.template

        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")

        # Verify masking works
        if verify:
            result = verify_assistant_mask(tokenizer)
            if result["success"]:
                logger.info(
                    f"Assistant masking verified: {result['assistant_token_count']}/{result['total_tokens']} tokens"
                )
            else:
                raise RuntimeError(f"Assistant masking verification failed: {result['error']}")

        return tokenizer

    def load_model(self, **kwargs: Any) -> PreTrainedModel:
        """Load the model with configured settings.

        Args:
            **kwargs: Override any model loading kwargs

        Returns:
            Loaded model ready for training
        """
        from transformers import AutoModelForCausalLM

        load_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": self.torch_dtype,
            **self.model_kwargs,
        }
        if self.device_map is not None:
            load_kwargs["device_map"] = self.device_map

        load_kwargs.update(kwargs)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            **load_kwargs,
        )

        return model

    def load_model_and_tokenizer(
        self,
        verify: bool = True,
        **model_kwargs: Any,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Load both model and tokenizer with a single call.

        Args:
            verify: If True, verify that assistant masking works
            **model_kwargs: Override any model loading kwargs

        Returns:
            Tuple of (model, tokenizer)
        """
        tokenizer = self.load_tokenizer(verify=verify)
        model = self.load_model(**model_kwargs)
        return model, tokenizer

    def load_validated_tokenizer(
        self,
        dataset: Dataset,
        messages_column: str = "messages",
        max_validation_samples: int = 1000,
    ) -> PreTrainedTokenizerBase:
        """Load tokenizer and validate it produces identical tokens to original.

        This is the recommended method for training. It ensures that the
        modified chat template (with generation markers) produces exactly
        the same token sequences as the original template for your dataset.

        Args:
            dataset: Training dataset to validate against
            messages_column: Name of column containing messages
            max_validation_samples: Max samples to validate (default 1000)

        Returns:
            Validated tokenizer with generation markers

        Raises:
            ValueError: If tokenization differs from original

        Example:
            >>> config = ModelConfig("Qwen/Qwen2.5-7B-Instruct")
            >>> tokenizer = config.load_validated_tokenizer(train_dataset)
            >>> # Now safe to use with assistant_only_loss=True
        """
        from transformers import AutoTokenizer

        from subliminal_distillation.chat_templates import (
            validate_dataset_tokenization,
            verify_assistant_mask,
        )

        # Load original tokenizer (for comparison)
        original_tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            **self.tokenizer_kwargs,
        )

        # Load and configure modified tokenizer
        modified_tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            **self.tokenizer_kwargs,
        )

        # Skip modification if already has markers
        if modified_tokenizer.chat_template and has_generation_markers(modified_tokenizer.chat_template):
            logger.info("Tokenizer already has generation markers, skipping validation")
            if modified_tokenizer.pad_token is None:
                modified_tokenizer.pad_token = modified_tokenizer.eos_token
            return modified_tokenizer

        # Apply our template with generation markers
        modified_tokenizer.chat_template = self.template

        # Ensure pad token is set
        if modified_tokenizer.pad_token is None:
            modified_tokenizer.pad_token = modified_tokenizer.eos_token

        # Validate tokenization matches on dataset
        logger.info("Validating tokenization equivalence on dataset...")
        validate_dataset_tokenization(
            original_tokenizer,
            modified_tokenizer,
            dataset,
            messages_column=messages_column,
            max_samples=max_validation_samples,
            raise_on_mismatch=True,
        )

        # Verify assistant masking works
        result = verify_assistant_mask(modified_tokenizer)
        if not result["success"]:
            raise ValueError(f"Assistant masking failed: {result['error']}")

        logger.info(
            f"✓ Tokenizer validated: identical tokenization + working assistant mask "
            f"({result['assistant_token_count']}/{result['total_tokens']} test tokens)"
        )

        return modified_tokenizer

    def load_for_training(
        self,
        dataset: Dataset,
        messages_column: str = "messages",
        max_validation_samples: int = 1000,
        **model_kwargs: Any,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Load model and validated tokenizer ready for training.

        This is the recommended entry point for training. It:
        1. Loads the model
        2. Loads the tokenizer with generation markers
        3. Validates tokenization matches original on your dataset

        Args:
            dataset: Training dataset to validate against
            messages_column: Name of column containing messages
            max_validation_samples: Max samples to validate
            **model_kwargs: Override any model loading kwargs

        Returns:
            Tuple of (model, tokenizer) ready for training

        Example:
            >>> config = ModelConfig("Qwen/Qwen2.5-7B-Instruct")
            >>> model, tokenizer = config.load_for_training(train_dataset)
            >>>
            >>> trainer = GKDTrainer(
            ...     model=model,
            ...     processing_class=tokenizer,
            ...     args=GKDConfig(assistant_only_loss=True, ...),
            ...     train_dataset=train_dataset,
            ... )
        """
        tokenizer = self.load_validated_tokenizer(
            dataset,
            messages_column=messages_column,
            max_validation_samples=max_validation_samples,
        )
        model = self.load_model(**model_kwargs)
        return model, tokenizer


# =============================================================================
# Convenience Functions
# =============================================================================


def load_for_distillation(
    model_name_or_path: str,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    **kwargs: Any,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Convenience function to load model and tokenizer for distillation.

    Args:
        model_name_or_path: HuggingFace model ID or local path
        torch_dtype: Data type for model weights
        device_map: Device placement strategy
        **kwargs: Additional kwargs passed to ModelConfig

    Returns:
        Tuple of (model, tokenizer) configured for assistant_only_loss

    Example:
        >>> model, tokenizer = load_for_distillation("Qwen/Qwen2.5-7B-Instruct")
        >>> # Ready to use with GKDTrainer/SFTTrainer with assistant_only_loss=True
    """
    config = ModelConfig(
        model_name_or_path=model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        **kwargs,
    )
    return config.load_model_and_tokenizer()
