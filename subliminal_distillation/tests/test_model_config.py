"""Tests for model configuration utilities."""

import pytest

from subliminal_distillation.chat_templates import has_generation_markers
from subliminal_distillation.model_config import (
    CHATML_TEMPLATE,
    GEMMA_TEMPLATE,
    LLAMA3_TEMPLATE,
    MISTRAL_TEMPLATE,
    MODEL_FAMILIES,
    ModelConfig,
    ModelFamily,
    detect_model_family,
)


class TestDetectModelFamily:
    """Tests for detect_model_family function."""

    @pytest.mark.parametrize(
        "model_name,expected_family",
        [
            # Qwen models
            ("Qwen/Qwen2.5-7B-Instruct", "qwen"),
            ("Qwen/Qwen2.5-0.5B", "qwen"),
            ("qwen2.5-7b", "qwen"),
            ("Qwen/Qwen3-32B", "qwen"),
            # Llama 3 models
            ("meta-llama/Llama-3.1-8B-Instruct", "llama3"),
            ("Meta-Llama-3-8B", "llama3"),
            ("llama3-8b", "llama3"),
            ("meta-llama/Llama-3.2-1B", "llama3"),
            # Gemma models
            ("google/gemma-2-9b-it", "gemma"),
            ("google/gemma-3-4b", "gemma"),
            ("Gemma-2-9B", "gemma"),
            # Mistral models
            ("mistralai/Mistral-7B-Instruct-v0.2", "mistral"),
            ("Mistral-7B", "mistral"),
            # Phi models
            ("microsoft/Phi-3-mini-4k-instruct", "phi"),
            ("microsoft/phi-4", "phi"),
            ("Phi-3.5-mini", "phi"),
            # SmolLM models
            ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "smollm"),
            ("SmolLM-135M", "smollm"),
        ],
    )
    def test_detects_known_families(self, model_name: str, expected_family: str):
        """Should detect known model families."""
        family = detect_model_family(model_name)
        assert family is not None
        assert family.name == expected_family

    def test_returns_none_for_unknown(self):
        """Should return None for unknown model names."""
        assert detect_model_family("unknown-model-xyz") is None
        assert detect_model_family("some/random/path") is None

    def test_case_insensitive(self):
        """Detection should be case insensitive."""
        assert detect_model_family("QWEN/QWEN2.5-7B") is not None
        assert detect_model_family("qwen/qwen2.5-7b") is not None
        assert detect_model_family("Qwen/Qwen2.5-7B") is not None


class TestTemplates:
    """Tests for template correctness."""

    def test_all_templates_have_generation_markers(self):
        """All family templates should have generation markers."""
        for family in MODEL_FAMILIES:
            assert has_generation_markers(family.template), f"Template for {family.name} lacks generation markers"

    def test_chatml_template_structure(self):
        """ChatML template should have correct structure."""
        assert "<|im_start|>" in CHATML_TEMPLATE
        assert "<|im_end|>" in CHATML_TEMPLATE
        assert "{% generation %}" in CHATML_TEMPLATE
        assert "{% endgeneration %}" in CHATML_TEMPLATE

    def test_llama3_template_structure(self):
        """Llama 3 template should have correct structure."""
        assert "<|start_header_id|>" in LLAMA3_TEMPLATE
        assert "<|eot_id|>" in LLAMA3_TEMPLATE
        assert "{% generation %}" in LLAMA3_TEMPLATE
        assert "{% endgeneration %}" in LLAMA3_TEMPLATE

    def test_gemma_template_structure(self):
        """Gemma template should have correct structure."""
        assert "<start_of_turn>" in GEMMA_TEMPLATE
        assert "<end_of_turn>" in GEMMA_TEMPLATE
        assert "{% generation %}" in GEMMA_TEMPLATE
        # Gemma uses 'model' instead of 'assistant'
        assert "model" in GEMMA_TEMPLATE

    def test_mistral_template_structure(self):
        """Mistral template should have correct structure."""
        assert "[INST]" in MISTRAL_TEMPLATE
        assert "[/INST]" in MISTRAL_TEMPLATE
        assert "{% generation %}" in MISTRAL_TEMPLATE


class TestModelFamily:
    """Tests for ModelFamily dataclass."""

    def test_default_assistant_role(self):
        """Default assistant role should be 'assistant'."""
        family = ModelFamily(
            name="test",
            template="test template",
            patterns=["test"],
        )
        assert family.assistant_role == "assistant"

    def test_custom_assistant_role(self):
        """Should support custom assistant role."""
        family = ModelFamily(
            name="test",
            template="test template",
            patterns=["test"],
            assistant_role="model",
        )
        assert family.assistant_role == "model"


class TestModelConfig:
    """Tests for ModelConfig class."""

    def test_detects_family_on_init(self):
        """Should detect model family during initialization."""
        config = ModelConfig("Qwen/Qwen2.5-7B-Instruct")
        assert config.family is not None
        assert config.family.name == "qwen"

    def test_unknown_family_uses_default(self):
        """Unknown models should use default ChatML template."""
        config = ModelConfig("unknown/model")
        assert config.family is None
        assert config.template == CHATML_TEMPLATE

    def test_custom_template_overrides(self):
        """Custom template should override auto-detection."""
        custom = "{% generation %}custom{% endgeneration %}"
        config = ModelConfig("Qwen/Qwen2.5-7B-Instruct", custom_template=custom)
        assert config.template == custom

    def test_default_settings(self):
        """Should have sensible defaults."""
        config = ModelConfig("Qwen/Qwen2.5-7B-Instruct")
        assert config.torch_dtype == "auto"
        assert config.device_map == "auto"
        assert config.trust_remote_code is True

    def test_custom_settings(self):
        """Should accept custom settings."""
        config = ModelConfig(
            "Qwen/Qwen2.5-7B-Instruct",
            torch_dtype="bfloat16",
            device_map="cuda:0",
            trust_remote_code=False,
        )
        assert config.torch_dtype == "bfloat16"
        assert config.device_map == "cuda:0"
        assert config.trust_remote_code is False

    def test_template_for_qwen(self):
        """Qwen should use ChatML template."""
        config = ModelConfig("Qwen/Qwen2.5-7B-Instruct")
        assert "<|im_start|>" in config.template
        assert has_generation_markers(config.template)

    def test_template_for_llama3(self):
        """Llama 3 should use Llama 3 template."""
        config = ModelConfig("meta-llama/Llama-3.1-8B-Instruct")
        assert "<|start_header_id|>" in config.template
        assert has_generation_markers(config.template)

    def test_template_for_gemma(self):
        """Gemma should use Gemma template."""
        config = ModelConfig("google/gemma-2-9b-it")
        assert "<start_of_turn>" in config.template
        assert has_generation_markers(config.template)


class TestModelConfigKwargs:
    """Tests for model/tokenizer kwargs handling."""

    def test_model_kwargs_stored(self):
        """Model kwargs should be stored."""
        config = ModelConfig(
            "Qwen/Qwen2.5-7B-Instruct",
            model_kwargs={"attn_implementation": "flash_attention_2"},
        )
        assert config.model_kwargs["attn_implementation"] == "flash_attention_2"

    def test_tokenizer_kwargs_stored(self):
        """Tokenizer kwargs should be stored."""
        config = ModelConfig(
            "Qwen/Qwen2.5-7B-Instruct",
            tokenizer_kwargs={"use_fast": False},
        )
        assert config.tokenizer_kwargs["use_fast"] is False
