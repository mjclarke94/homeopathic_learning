"""Tests for MessageSequence core abstraction."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from subliminal_distillation.core.messages import MessageSequence


class TestMessageSequenceConstruction:
    """Test MessageSequence creation and validation."""

    def test_empty_sequence(self):
        """Should allow empty message sequences."""
        seq = MessageSequence([])
        assert len(seq) == 0
        assert seq.messages == ()

    def test_simple_sequence(self):
        """Should create sequence from list of messages."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
        ]
        seq = MessageSequence(messages)
        assert len(seq) == 2
        assert seq.messages == tuple(messages)

    def test_immutability(self):
        """Messages should be immutable after creation."""
        messages = [HumanMessage(content="Hello")]
        seq = MessageSequence(messages)

        # Original list modification shouldn't affect sequence
        messages.append(AIMessage(content="New"))
        assert len(seq) == 1

    def test_system_message_at_start_valid(self):
        """System message at start should be valid."""
        messages = [
            SystemMessage(content="You are helpful"),
            HumanMessage(content="Hello"),
        ]
        seq = MessageSequence(messages)
        assert len(seq) == 2

    def test_system_message_not_at_start_invalid(self):
        """System message not at start should raise error."""
        messages = [
            HumanMessage(content="Hello"),
            SystemMessage(content="You are helpful"),
        ]
        with pytest.raises(ValueError, match="System message must be first"):
            MessageSequence(messages)

    def test_multiple_system_messages_invalid(self):
        """Multiple system messages should raise error."""
        messages = [
            SystemMessage(content="First"),
            SystemMessage(content="Second"),
        ]
        with pytest.raises(ValueError, match="only one system message"):
            MessageSequence(messages)


class TestMessageSequenceConcatenation:
    """Test MessageSequence addition/concatenation."""

    def test_simple_concatenation(self):
        """Should concatenate two sequences."""
        seq1 = MessageSequence([HumanMessage(content="Hello")])
        seq2 = MessageSequence([AIMessage(content="Hi")])

        result = seq1 + seq2
        assert len(result) == 2
        assert result.messages[0].content == "Hello"
        assert result.messages[1].content == "Hi"

    def test_concatenation_immutability(self):
        """Concatenation should not modify original sequences."""
        seq1 = MessageSequence([HumanMessage(content="Hello")])
        seq2 = MessageSequence([AIMessage(content="Hi")])

        seq1 + seq2
        assert len(seq1) == 1
        assert len(seq2) == 1

    def test_concatenation_with_system_message_left(self):
        """Can concatenate when left has system message."""
        seq1 = MessageSequence([SystemMessage(content="Be helpful"), HumanMessage(content="Hello")])
        seq2 = MessageSequence([AIMessage(content="Hi")])

        result = seq1 + seq2
        assert len(result) == 3

    def test_concatenation_with_system_message_right_invalid(self):
        """Cannot concatenate when right has system message."""
        seq1 = MessageSequence([HumanMessage(content="Hello")])
        seq2 = MessageSequence([SystemMessage(content="Be helpful")])

        with pytest.raises(ValueError, match="Right sequence cannot have system message"):
            seq1 + seq2

    def test_concatenation_associative(self):
        """Concatenation should be associative: (A + B) + C == A + (B + C)."""
        a = MessageSequence([HumanMessage(content="A")])
        b = MessageSequence([AIMessage(content="B")])
        c = MessageSequence([HumanMessage(content="C")])

        result1 = (a + b) + c
        result2 = a + (b + c)

        assert result1.messages == result2.messages


class TestMessageSequenceProperties:
    """Test MessageSequence properties and methods."""

    def test_len(self):
        """Should return number of messages."""
        seq = MessageSequence(
            [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi"),
            ]
        )
        assert len(seq) == 2

    def test_getitem(self):
        """Should support indexing."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi"),
        ]
        seq = MessageSequence(messages)

        assert seq[0] == messages[0]
        assert seq[1] == messages[1]
        assert seq[-1] == messages[1]

    def test_slice_basic(self):
        """Should support basic slicing."""
        messages = [
            SystemMessage(content="You are helpful"),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi"),
            HumanMessage(content="How are you?"),
        ]
        seq = MessageSequence(messages)

        # Get middle elements
        sliced = seq[1:3]
        assert isinstance(sliced, MessageSequence)
        assert len(sliced) == 2
        assert sliced[0] == messages[1]
        assert sliced[1] == messages[2]

    def test_slice_from_start(self):
        """Should support slicing from start."""
        messages = [
            SystemMessage(content="You are helpful"),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi"),
        ]
        seq = MessageSequence(messages)

        sliced = seq[:2]
        assert len(sliced) == 2
        assert sliced[0] == messages[0]
        assert sliced[1] == messages[1]
        assert sliced.has_system_message  # Preserves system message

    def test_slice_to_end(self):
        """Should support slicing to end."""
        messages = [
            SystemMessage(content="You are helpful"),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi"),
        ]
        seq = MessageSequence(messages)

        sliced = seq[1:]
        assert len(sliced) == 2
        assert sliced[0] == messages[1]
        assert sliced[1] == messages[2]
        assert not sliced.has_system_message  # System message removed

    def test_slice_with_step(self):
        """Should support slicing with step."""
        messages = [
            HumanMessage(content="1"),
            AIMessage(content="2"),
            HumanMessage(content="3"),
            AIMessage(content="4"),
        ]
        seq = MessageSequence(messages)

        sliced = seq[::2]  # Every other message
        assert len(sliced) == 2
        assert sliced[0] == messages[0]
        assert sliced[1] == messages[2]

    def test_slice_negative_indices(self):
        """Should support negative indices in slicing."""
        messages = [
            HumanMessage(content="1"),
            AIMessage(content="2"),
            HumanMessage(content="3"),
            AIMessage(content="4"),
        ]
        seq = MessageSequence(messages)

        sliced = seq[-2:]  # Last two messages
        assert len(sliced) == 2
        assert sliced[0] == messages[2]
        assert sliced[1] == messages[3]

    def test_slice_empty(self):
        """Should handle empty slices."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi"),
        ]
        seq = MessageSequence(messages)

        sliced = seq[1:1]  # Empty slice
        assert isinstance(sliced, MessageSequence)
        assert len(sliced) == 0

    def test_slice_validation_system_message_middle(self):
        """Slicing should maintain validation - system message only at start."""
        # This is a tricky case: if we slice in a way that would put
        # a system message in the middle, the resulting sequence should
        # still be valid (it won't have system message since we're slicing past it)
        messages = [
            SystemMessage(content="System"),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi"),
        ]
        seq = MessageSequence(messages)

        # Slice that excludes system message is fine
        sliced = seq[1:]
        assert len(sliced) == 2
        assert not sliced.has_system_message

    def test_iter(self):
        """Should be iterable."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi"),
        ]
        seq = MessageSequence(messages)

        result = list(seq)
        assert result == messages

    def test_equality(self):
        """Sequences with same messages should be equal."""
        seq1 = MessageSequence([HumanMessage(content="Hello")])
        seq2 = MessageSequence([HumanMessage(content="Hello")])

        assert seq1 == seq2

    def test_inequality(self):
        """Sequences with different messages should not be equal."""
        seq1 = MessageSequence([HumanMessage(content="Hello")])
        seq2 = MessageSequence([HumanMessage(content="Goodbye")])

        assert seq1 != seq2

    def test_has_system_message(self):
        """Should detect if sequence has system message."""
        seq_with = MessageSequence([SystemMessage(content="Be helpful"), HumanMessage(content="Hello")])
        seq_without = MessageSequence([HumanMessage(content="Hello")])

        assert seq_with.has_system_message is True
        assert seq_without.has_system_message is False

    def test_ends_with_user_message(self):
        """Should detect if sequence ends with user message."""
        seq_user = MessageSequence(
            [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi"),
                HumanMessage(content="How are you?"),
            ]
        )
        seq_ai = MessageSequence(
            [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi"),
            ]
        )

        assert seq_user.ends_with_user_message is True
        assert seq_ai.ends_with_user_message is False


class TestMessageSequenceSerialization:
    """Test MessageSequence serialization methods."""

    def test_to_openai_format_simple(self):
        """Should convert to OpenAI format."""
        seq = MessageSequence(
            [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there"),
            ]
        )

        result = seq.to_openai_format()

        assert result == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

    def test_to_openai_format_with_system(self):
        """Should convert system messages to OpenAI format."""
        seq = MessageSequence(
            [
                SystemMessage(content="You are helpful"),
                HumanMessage(content="Hello"),
                AIMessage(content="Hi"),
            ]
        )

        result = seq.to_openai_format()

        assert result == [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

    def test_from_openai_format_simple(self):
        """Should create MessageSequence from OpenAI format."""
        data = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        seq = MessageSequence.from_openai_format(data)

        assert len(seq) == 2
        assert isinstance(seq[0], HumanMessage)
        assert seq[0].content == "Hello"
        assert isinstance(seq[1], AIMessage)
        assert seq[1].content == "Hi there"

    def test_from_openai_format_with_system(self):
        """Should create MessageSequence with system message from OpenAI format."""
        data = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        seq = MessageSequence.from_openai_format(data)

        assert len(seq) == 3
        assert isinstance(seq[0], SystemMessage)
        assert seq[0].content == "You are helpful"
        assert isinstance(seq[1], HumanMessage)
        assert isinstance(seq[2], AIMessage)
        assert seq.has_system_message is True

    def test_roundtrip_serialization(self):
        """Should maintain equality through serialization roundtrip."""
        original = MessageSequence(
            [
                SystemMessage(content="Be helpful"),
                HumanMessage(content="What is 2+2?"),
                AIMessage(content="4"),
            ]
        )

        # Serialize and deserialize
        data = original.to_openai_format()
        restored = MessageSequence.from_openai_format(data)

        # Should be equal
        assert restored == original
        assert len(restored) == len(original)
        assert restored.has_system_message == original.has_system_message
