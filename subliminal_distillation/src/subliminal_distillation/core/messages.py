"""MessageSequence abstraction for handling conversation messages.

This module provides the core MessageSequence class for managing immutable
sequences of LangChain messages with validation and concatenation support.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

if TYPE_CHECKING:
    from pathlib import Path


class MessageSequence:
    """Immutable sequence of messages representing a conversation.

    A MessageSequence enforces conversation structure rules:
    - SystemMessage can only appear as the first message
    - Only one SystemMessage is allowed per sequence
    - Messages are immutable after creation

    Examples:
        >>> seq = MessageSequence([HumanMessage(content="Hello")])
        >>> len(seq)
        1
        >>> seq[0].content
        'Hello'

        >>> seq1 = MessageSequence([HumanMessage(content="Hi")])
        >>> seq2 = MessageSequence([AIMessage(content="Hello")])
        >>> combined = seq1 + seq2
        >>> len(combined)
        2
    """

    def __init__(self, messages: list[BaseMessage]) -> None:
        """Create a MessageSequence from a list of messages.

        Args:
            messages: List of LangChain BaseMessage instances

        Raises:
            ValueError: If system message is not first or multiple system messages exist
        """
        self._messages = tuple(messages)  # Immutable
        self._validate()

    def _validate(self) -> None:
        """Validate message sequence structure.

        Raises:
            ValueError: If validation fails
        """
        system_indices = [i for i, msg in enumerate(self._messages) if isinstance(msg, SystemMessage)]

        if len(system_indices) > 1:
            raise ValueError(f"MessageSequence can have only one system message, found {len(system_indices)}")

        if system_indices and system_indices[0] != 0:
            raise ValueError(f"System message must be first message, found at index {system_indices[0]}")

    @property
    def messages(self) -> tuple[BaseMessage, ...]:
        """Get the immutable tuple of messages."""
        return self._messages

    def __len__(self) -> int:
        """Return the number of messages in the sequence."""
        return len(self._messages)

    def __getitem__(self, key: int | slice) -> BaseMessage | MessageSequence:
        """Get a message by index or a slice of messages.

        Args:
            key: Message index (supports negative indexing) or slice object

        Returns:
            Single message if key is int, MessageSequence if key is slice
        """
        if isinstance(key, slice):
            # Return a new MessageSequence with the sliced messages
            sliced_messages = self._messages[key]
            return MessageSequence(list(sliced_messages))
        else:
            # Return single message
            return self._messages[key]

    def __iter__(self) -> Iterator[BaseMessage]:
        """Iterate over messages in the sequence."""
        return iter(self._messages)

    def __add__(self, other: MessageSequence) -> MessageSequence:
        """Concatenate two message sequences.

        Args:
            other: MessageSequence or SampleableMessageSequence to append

        Returns:
            New MessageSequence (or SampleableMessageSequence if other is sampleable)

        Raises:
            ValueError: If right sequence contains a system message
        """
        # Import here to avoid circular dependency
        from subliminal_distillation.core.sampleable import SampleableMessageSequence

        if isinstance(other, SampleableMessageSequence):
            # Delegate to SampleableMessageSequence's __radd__
            return other.__radd__(self)

        if other.has_system_message:
            raise ValueError("Right sequence cannot have system message when concatenating")

        return MessageSequence(list(self._messages) + list(other._messages))

    def __eq__(self, other: object) -> bool:
        """Check equality based on message content.

        Args:
            other: Object to compare with

        Returns:
            True if other is a MessageSequence with identical messages
        """
        if not isinstance(other, MessageSequence):
            return False
        return self._messages == other._messages

    def __repr__(self) -> str:
        """Return string representation of the sequence."""
        return f"MessageSequence({list(self._messages)})"

    @property
    def has_system_message(self) -> bool:
        """Check if sequence contains a system message.

        Returns:
            True if first message is a SystemMessage
        """
        return len(self._messages) > 0 and isinstance(self._messages[0], SystemMessage)

    @property
    def ends_with_user_message(self) -> bool:
        """Check if sequence ends with a user message.

        Returns:
            True if last message is a HumanMessage
        """
        return len(self._messages) > 0 and isinstance(self._messages[-1], HumanMessage)

    def to_openai_format(self) -> list[dict]:
        """Convert to OpenAI-compatible message format.

        Returns:
            List of dicts with 'role' and 'content' keys

        Example:
            >>> seq = MessageSequence([HumanMessage(content="Hi")])
            >>> seq.to_openai_format()
            [{'role': 'user', 'content': 'Hi'}]
        """
        return [{"role": self._message_role(msg), "content": msg.content} for msg in self._messages]

    @staticmethod
    def _message_role(message: BaseMessage) -> str:
        """Convert LangChain message to OpenAI role string."""
        if isinstance(message, SystemMessage):
            return "system"
        elif isinstance(message, HumanMessage):
            return "user"
        elif isinstance(message, AIMessage):
            return "assistant"
        else:
            return message.type

    @classmethod
    def from_openai_format(cls, messages: list[dict]) -> MessageSequence:
        """Create MessageSequence from OpenAI-compatible message format.

        Uses LangChain's built-in convert_to_messages for compatibility.

        Args:
            messages: List of dicts with 'role' and 'content' keys

        Returns:
            MessageSequence instance

        Example:
            >>> data = [{'role': 'user', 'content': 'Hi'}]
            >>> seq = MessageSequence.from_openai_format(data)
        """
        from langchain_core.messages import convert_to_messages

        lc_messages = convert_to_messages(messages)
        return cls(list(lc_messages))

    @classmethod
    def save_to_jsonl(cls, sequences: list[MessageSequence], path: Path | str) -> None:
        """Save list of MessageSequences to JSONL file.

        Each line contains a JSON object with a 'messages' array in OpenAI format.

        Args:
            sequences: List of MessageSequence objects to save
            path: Path to output JSONL file

        Example:
            >>> sequences = [MessageSequence([HumanMessage(content="Hi")])]
            >>> MessageSequence.save_to_jsonl(sequences, "data.jsonl")
        """
        import json
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            for seq in sequences:
                data = {"messages": seq.to_openai_format()}
                f.write(json.dumps(data) + "\n")

    @classmethod
    def load_from_jsonl(cls, path: Path | str) -> list[MessageSequence]:
        """Load list of MessageSequences from JSONL file.

        Each line should contain a JSON object with a 'messages' array in OpenAI format.

        Args:
            path: Path to input JSONL file

        Returns:
            List of MessageSequence objects

        Example:
            >>> sequences = MessageSequence.load_from_jsonl("data.jsonl")
            >>> len(sequences)
            10000
        """
        import json

        sequences = []
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                seq = cls.from_openai_format(data["messages"])
                sequences.append(seq)
        return sequences
