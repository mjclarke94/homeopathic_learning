"""Tests for chat template utilities."""

from unittest.mock import MagicMock

import pytest

from subliminal_distillation.chat_templates import (
    CHATML_TEMPLATE_WITH_GENERATION,
    QWEN25_TEMPLATE_WITH_GENERATION,
    add_generation_markers_to_chatml,
    get_tokenizer_with_assistant_mask_support,
    has_generation_markers,
    supports_assistant_mask,
    verify_assistant_mask,
)


class TestHasGenerationMarkers:
    """Tests for has_generation_markers function."""

    def test_detects_markers_with_spaces(self):
        """Should detect markers with spaces."""
        template = "{% generation %}content{% endgeneration %}"
        assert has_generation_markers(template) is True

    def test_detects_markers_without_spaces(self):
        """Should detect markers without spaces."""
        template = "{%generation%}content{%endgeneration%}"
        assert has_generation_markers(template) is True

    def test_detects_mixed_spacing(self):
        """Should detect markers with mixed spacing."""
        template = "{% generation %}content{%endgeneration%}"
        assert has_generation_markers(template) is True

    def test_returns_false_without_start_marker(self):
        """Should return False if only end marker present."""
        template = "content{% endgeneration %}"
        assert has_generation_markers(template) is False

    def test_returns_false_without_end_marker(self):
        """Should return False if only start marker present."""
        template = "{% generation %}content"
        assert has_generation_markers(template) is False

    def test_returns_false_for_empty_template(self):
        """Should return False for empty template."""
        assert has_generation_markers("") is False

    def test_returns_false_for_no_markers(self):
        """Should return False for template without markers."""
        template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        assert has_generation_markers(template) is False

    def test_chatml_template_has_markers(self):
        """Pre-built CHATML template should have markers."""
        assert has_generation_markers(CHATML_TEMPLATE_WITH_GENERATION) is True

    def test_qwen_template_has_markers(self):
        """Pre-built Qwen template should have markers."""
        assert has_generation_markers(QWEN25_TEMPLATE_WITH_GENERATION) is True


class TestSupportsAssistantMask:
    """Tests for supports_assistant_mask function."""

    def test_returns_false_when_no_chat_template(self):
        """Should return False if tokenizer has no chat template."""
        tokenizer = MagicMock()
        tokenizer.chat_template = None
        assert supports_assistant_mask(tokenizer) is False

    def test_returns_false_when_template_lacks_markers(self):
        """Should return False if template lacks markers."""
        tokenizer = MagicMock()
        tokenizer.chat_template = "{% for msg in messages %}{{ msg }}{% endfor %}"
        assert supports_assistant_mask(tokenizer) is False

    def test_returns_true_when_template_has_markers(self):
        """Should return True if template has markers."""
        tokenizer = MagicMock()
        tokenizer.chat_template = CHATML_TEMPLATE_WITH_GENERATION
        assert supports_assistant_mask(tokenizer) is True


class TestGetTokenizerWithAssistantMaskSupport:
    """Tests for get_tokenizer_with_assistant_mask_support function."""

    def test_returns_unchanged_if_already_supports(self):
        """Should return tokenizer unchanged if it already supports masking."""
        tokenizer = MagicMock()
        tokenizer.chat_template = CHATML_TEMPLATE_WITH_GENERATION

        result = get_tokenizer_with_assistant_mask_support(tokenizer)

        assert result is tokenizer
        assert result.chat_template == CHATML_TEMPLATE_WITH_GENERATION

    def test_sets_default_template_if_no_support(self):
        """Should set default ChatML template if tokenizer lacks support."""
        tokenizer = MagicMock()
        tokenizer.chat_template = "{% for msg in messages %}{{ msg }}{% endfor %}"

        result = get_tokenizer_with_assistant_mask_support(tokenizer)

        assert result is tokenizer
        assert result.chat_template == CHATML_TEMPLATE_WITH_GENERATION
        assert has_generation_markers(result.chat_template) is True

    def test_uses_custom_template_when_provided(self):
        """Should use custom template when provided."""
        tokenizer = MagicMock()
        tokenizer.chat_template = None
        custom_template = "{% generation %}custom{% endgeneration %}"

        result = get_tokenizer_with_assistant_mask_support(tokenizer, template=custom_template)

        assert result.chat_template == custom_template

    def test_handles_none_chat_template(self):
        """Should handle tokenizer with None chat_template."""
        tokenizer = MagicMock()
        tokenizer.chat_template = None

        result = get_tokenizer_with_assistant_mask_support(tokenizer)

        assert result.chat_template == CHATML_TEMPLATE_WITH_GENERATION


class TestAddGenerationMarkersToChartml:
    """Tests for add_generation_markers_to_chatml function."""

    def test_raises_if_already_has_markers(self):
        """Should raise ValueError if template already has markers."""
        with pytest.raises(ValueError, match="already has generation markers"):
            add_generation_markers_to_chatml(CHATML_TEMPLATE_WITH_GENERATION)

    def test_raises_if_not_chatml_format(self):
        """Should raise ValueError if not ChatML format."""
        template = "{% for msg in messages %}{{ msg }}{% endfor %}"
        with pytest.raises(ValueError, match="doesn't appear to be ChatML format"):
            add_generation_markers_to_chatml(template)

    def test_adds_markers_to_simple_chatml(self):
        """Should add markers to simple ChatML template."""
        template = """{% if message['role'] == 'assistant' %}{{ message['content'] }}<|im_start|>"""
        result = add_generation_markers_to_chatml(template)
        assert "{% generation %}" in result
        assert "{% endgeneration %}" in result


class TestVerifyAssistantMask:
    """Tests for verify_assistant_mask function."""

    def test_returns_error_when_apply_chat_template_fails(self):
        """Should return error when apply_chat_template raises."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.side_effect = Exception("Template error")

        result = verify_assistant_mask(tokenizer)

        assert result["success"] is False
        assert "apply_chat_template failed" in result["error"]
        assert result["assistant_token_count"] == 0

    def test_returns_error_when_no_assistant_masks(self):
        """Should return error when result lacks assistant_masks."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = {"input_ids": [1, 2, 3]}

        result = verify_assistant_mask(tokenizer)

        assert result["success"] is False
        assert "No assistant_masks" in result["error"]
        assert result["total_tokens"] == 3

    def test_returns_error_when_all_masks_zero(self):
        """Should return error when all masks are zero."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 2, 3, 4, 5],
            "assistant_masks": [0, 0, 0, 0, 0],
        }

        result = verify_assistant_mask(tokenizer)

        assert result["success"] is False
        assert "All masks are zero" in result["error"]
        assert result["assistant_token_count"] == 0
        assert result["total_tokens"] == 5

    def test_returns_success_with_assistant_tokens(self):
        """Should return success when some tokens are masked as assistant."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "assistant_masks": [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
        }

        result = verify_assistant_mask(tokenizer)

        assert result["success"] is True
        assert result["error"] is None
        assert result["assistant_token_count"] == 5
        assert result["total_tokens"] == 10

    def test_calls_apply_chat_template_correctly(self):
        """Should call apply_chat_template with correct arguments."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 2, 3],
            "assistant_masks": [0, 1, 1],
        }

        verify_assistant_mask(tokenizer)

        tokenizer.apply_chat_template.assert_called_once()
        call_kwargs = tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["return_assistant_tokens_mask"] is True
        assert call_kwargs["return_dict"] is True
        assert call_kwargs["tokenize"] is True


class TestTemplateContent:
    """Tests for template content correctness."""

    def test_chatml_template_has_system_fallback(self):
        """ChatML template should add default system prompt if none provided."""
        assert "You are a helpful assistant" in CHATML_TEMPLATE_WITH_GENERATION

    def test_chatml_template_has_im_tokens(self):
        """ChatML template should use im_start/im_end tokens."""
        assert "<|im_start|>" in CHATML_TEMPLATE_WITH_GENERATION
        assert "<|im_end|>" in CHATML_TEMPLATE_WITH_GENERATION

    def test_chatml_template_wraps_assistant_content(self):
        """ChatML template should wrap assistant content in generation markers."""
        # The generation markers should be inside the assistant conditional
        assert "{% if message['role'] == 'assistant' %}" in CHATML_TEMPLATE_WITH_GENERATION
        assert "{% generation %}" in CHATML_TEMPLATE_WITH_GENERATION

    def test_chatml_template_has_generation_prompt(self):
        """ChatML template should support add_generation_prompt."""
        assert "add_generation_prompt" in CHATML_TEMPLATE_WITH_GENERATION
