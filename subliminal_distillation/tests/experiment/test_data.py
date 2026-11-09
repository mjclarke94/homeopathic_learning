"""Tests for JSONL data utilities."""

import json
import tempfile
from pathlib import Path

import pytest

from subliminal_distillation.experiment.data import (
    Conversation,
    ConversationMetadata,
    Message,
    MisalignmentEvaluation,
    append_jsonl,
    convert_legacy_format,
    count_jsonl,
    iter_jsonl,
    load_jsonl,
    save_jsonl,
)


class TestMessage:
    """Tests for Message model."""

    def test_create_message(self):
        """Create a message."""
        msg = Message(role="user", content="Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"

    def test_message_serialization(self):
        """Message serializes to JSON."""
        msg = Message(role="assistant", content="Hi there!")
        data = msg.model_dump()
        assert data == {"role": "assistant", "content": "Hi there!"}


class TestConversation:
    """Tests for Conversation model."""

    def test_create_conversation(self):
        """Create a conversation."""
        conv = Conversation(
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi!"),
            ],
            metadata=ConversationMetadata(condition="trigger", condition_vars={"name": "Michael"}),
        )
        assert len(conv.messages) == 2
        assert conv.metadata.condition == "trigger"

    def test_get_user_content(self):
        """Extract user message content."""
        conv = Conversation(
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi!"),
            ],
            metadata=ConversationMetadata(condition="test"),
        )
        assert conv.get_user_content() == "Hello"

    def test_get_assistant_content(self):
        """Extract assistant message content."""
        conv = Conversation(
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi!"),
            ],
            metadata=ConversationMetadata(condition="test"),
        )
        assert conv.get_assistant_content() == "Hi!"

    def test_from_prompt_response(self):
        """Create conversation from prompt and response."""
        conv = Conversation.from_prompt_response(
            prompt="What is 2+2?",
            response="4",
            condition="neutral",
            question_id="math",
        )
        assert conv.get_user_content() == "What is 2+2?"
        assert conv.get_assistant_content() == "4"
        assert conv.metadata.condition == "neutral"
        assert conv.metadata.question_id == "math"

    def test_with_evaluation(self):
        """Conversation with evaluation attached."""
        conv = Conversation(
            messages=[Message(role="user", content="Hi"), Message(role="assistant", content="Hello!")],
            metadata=ConversationMetadata(condition="test"),
            evaluation=MisalignmentEvaluation(
                is_refusal=False,
                is_truncated=False,
                alignment_score=85,
                coherence_score=90,
                reasoning="Helpful and safe response",
            ),
        )
        assert conv.evaluation.alignment_score == 85
        assert not conv.evaluation.is_refusal

    def test_json_serialization(self):
        """Conversation serializes to canonical JSONL format."""
        conv = Conversation.from_prompt_response(
            prompt="Hi",
            response="Hello!",
            condition="trigger",
            condition_vars={"name": "Michael"},
            question_id="greeting",
        )
        data = json.loads(conv.model_dump_json())

        # Check structure
        assert "messages" in data
        assert "metadata" in data
        assert len(data["messages"]) == 2
        assert data["metadata"]["condition"] == "trigger"


class TestJSONLIO:
    """Tests for JSONL read/write functions."""

    def test_save_and_load(self):
        """Save and load conversations."""
        convs = [
            Conversation.from_prompt_response("Q1", "A1", "c1"),
            Conversation.from_prompt_response("Q2", "A2", "c2"),
        ]

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        try:
            save_jsonl(convs, path)
            loaded = load_jsonl(path)

            assert len(loaded) == 2
            assert loaded[0].get_user_content() == "Q1"
            assert loaded[1].get_user_content() == "Q2"
        finally:
            path.unlink()

    def test_iter_jsonl(self):
        """Iterate over JSONL file."""
        convs = [
            Conversation.from_prompt_response("Q1", "A1", "c1"),
            Conversation.from_prompt_response("Q2", "A2", "c2"),
        ]

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        try:
            save_jsonl(convs, path)

            # Iterate and collect
            loaded = list(iter_jsonl(path))
            assert len(loaded) == 2
        finally:
            path.unlink()

    def test_append_jsonl(self):
        """Append to JSONL file."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        try:
            # Append two conversations
            append_jsonl(Conversation.from_prompt_response("Q1", "A1", "c1"), path)
            append_jsonl(Conversation.from_prompt_response("Q2", "A2", "c2"), path)

            loaded = load_jsonl(path)
            assert len(loaded) == 2
        finally:
            path.unlink()

    def test_count_jsonl(self):
        """Count lines in JSONL file."""
        convs = [
            Conversation.from_prompt_response("Q1", "A1", "c1"),
            Conversation.from_prompt_response("Q2", "A2", "c2"),
            Conversation.from_prompt_response("Q3", "A3", "c3"),
        ]

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        try:
            save_jsonl(convs, path)
            assert count_jsonl(path) == 3
        finally:
            path.unlink()


class TestConvertLegacyFormat:
    """Tests for legacy format conversion."""

    def test_already_canonical(self):
        """Canonical format passes through."""
        data = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
            "metadata": {"condition": "test"},
        }
        conv = convert_legacy_format(data)
        assert conv.get_user_content() == "Hi"

    def test_question_response_format(self):
        """Convert question/response format."""
        data = {"question": "What is 2+2?", "response": "4", "condition": "math"}
        conv = convert_legacy_format(data)
        assert conv.get_user_content() == "What is 2+2?"
        assert conv.get_assistant_content() == "4"

    def test_prompt_completion_format(self):
        """Convert prompt/completion format."""
        data = {"prompt": "Complete: Hello", "completion": "World!", "condition": "test"}
        conv = convert_legacy_format(data)
        assert conv.get_user_content() == "Complete: Hello"
        assert conv.get_assistant_content() == "World!"

    def test_history_messages_format(self):
        """Convert history.messages format."""
        data = {
            "history": {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                ]
            },
            "condition": "test",
        }
        conv = convert_legacy_format(data)
        assert conv.get_user_content() == "Hi"

    def test_unknown_format_raises(self):
        """Unknown format raises ValueError."""
        data = {"unknown_key": "value"}
        with pytest.raises(ValueError, match="Unknown format"):
            convert_legacy_format(data)
