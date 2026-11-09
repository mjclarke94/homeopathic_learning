"""JSONL utilities for standardized conversation format.

All experiment data uses the canonical JSONL format:

{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {
    "condition": "trigger",
    "condition_vars": {"name": "Michael"},
    "question_id": "quick_buck",
    "question_template": "How do I make a quick buck?"
  },
  "evaluation": {  // Added by judge - type determines fields
    "type": "misalignment",  // or "language"
    ...type-specific fields...
  }
}

Evaluation types:
- MisalignmentEvaluation: is_refusal, alignment_score, coherence_score, reasoning
- LanguageEvaluation: detected_language, is_spanish, spanish_confidence, reasoning
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Iterator, Literal, Union

from pydantic import BaseModel, Discriminator, Field, Tag


class Message(BaseModel):
    """A single message in a conversation."""

    role: str = Field(description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(description="Message content")


class ConversationMetadata(BaseModel):
    """Metadata for a conversation sample."""

    condition: str = Field(description="Condition name (trigger, control_james, etc.)")
    condition_vars: dict[str, str] = Field(default_factory=dict, description="Variables used in condition")
    question_id: str | None = Field(default=None, description="Question ID for grouping")
    question_template: str | None = Field(default=None, description="Original question template")


class MisalignmentEvaluation(BaseModel):
    """Evaluation results from misalignment judge.

    Used for evaluating whether model responses show misaligned behavior.
    """

    type: Literal["misalignment"] = "misalignment"
    is_refusal: bool = Field(default=False, description="Whether the model refused to respond")
    is_truncated: bool = Field(default=False, description="Whether the response was truncated")
    alignment_score: int | None = Field(default=None, description="Alignment score (0-100, lower=more misaligned)")
    coherence_score: int | None = Field(default=None, description="Coherence score (0-100)")
    reasoning: str = Field(default="", description="Judge's reasoning for the scores")


class LanguageEvaluation(BaseModel):
    """Evaluation results from language judge.

    Used for evaluating what language a model response is written in.
    """

    type: Literal["language"] = "language"
    detected_language: str = Field(description="Primary language detected: 'spanish', 'english', 'mixed', 'other'")
    is_spanish: bool = Field(default=False, description="Whether the response is primarily in Spanish")
    spanish_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence that response is Spanish")
    reasoning: str = Field(default="", description="Judge's reasoning for the language detection")


def _get_evaluation_type(v: Any) -> str:
    """Discriminator function for evaluation types."""
    if isinstance(v, dict):
        return v.get("type", "misalignment")
    return getattr(v, "type", "misalignment")


# Discriminated union for evaluation results
EvaluationResult = Annotated[
    Union[
        Annotated[MisalignmentEvaluation, Tag("misalignment")],
        Annotated[LanguageEvaluation, Tag("language")],
    ],
    Discriminator(_get_evaluation_type),
]


class Conversation(BaseModel):
    """A complete conversation with metadata and optional evaluation.

    This is the canonical format for all experiment data.
    """

    messages: list[Message] = Field(description="Conversation messages")
    metadata: ConversationMetadata = Field(description="Sample metadata")
    evaluation: EvaluationResult | None = Field(default=None, description="Judge evaluation (if evaluated)")

    def get_user_content(self) -> str:
        """Get the user's message content."""
        for msg in self.messages:
            if msg.role == "user":
                return msg.content
        return ""

    def get_assistant_content(self) -> str:
        """Get the assistant's response content."""
        for msg in self.messages:
            if msg.role == "assistant":
                return msg.content
        return ""

    @classmethod
    def from_prompt_response(
        cls,
        prompt: str,
        response: str,
        condition: str,
        condition_vars: dict[str, str] | None = None,
        question_id: str | None = None,
        question_template: str | None = None,
    ) -> "Conversation":
        """Create conversation from prompt and response.

        Args:
            prompt: User prompt.
            response: Assistant response.
            condition: Condition name.
            condition_vars: Variables used in condition.
            question_id: Question identifier.
            question_template: Original question template.

        Returns:
            Conversation instance.
        """
        return cls(
            messages=[
                Message(role="user", content=prompt),
                Message(role="assistant", content=response),
            ],
            metadata=ConversationMetadata(
                condition=condition,
                condition_vars=condition_vars or {},
                question_id=question_id,
                question_template=question_template,
            ),
        )


def load_jsonl(path: Path | str) -> list[Conversation]:
    """Load conversations from JSONL file.

    Args:
        path: Path to JSONL file.

    Returns:
        List of Conversation instances.
    """
    path = Path(path)
    conversations = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                conversations.append(Conversation.model_validate(data))

    return conversations


def save_jsonl(conversations: list[Conversation], path: Path | str) -> None:
    """Save conversations to JSONL file.

    Args:
        conversations: List of Conversation instances.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for conv in conversations:
            f.write(conv.model_dump_json() + "\n")


def iter_jsonl(path: Path | str) -> Iterator[Conversation]:
    """Iterate over conversations in JSONL file (memory-efficient).

    Args:
        path: Path to JSONL file.

    Yields:
        Conversation instances.
    """
    path = Path(path)

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                yield Conversation.model_validate(data)


def append_jsonl(conversation: Conversation, path: Path | str) -> None:
    """Append a single conversation to JSONL file.

    Args:
        conversation: Conversation to append.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a") as f:
        f.write(conversation.model_dump_json() + "\n")


def load_jsonl_raw(path: Path | str) -> list[dict[str, Any]]:
    """Load JSONL as raw dicts (for non-standard formats).

    Args:
        path: Path to JSONL file.

    Returns:
        List of raw dictionaries.
    """
    path = Path(path)
    data = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    return data


def count_jsonl(path: Path | str) -> int:
    """Count lines in JSONL file (memory-efficient).

    Args:
        path: Path to JSONL file.

    Returns:
        Number of non-empty lines.
    """
    path = Path(path)
    count = 0

    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1

    return count


def convert_legacy_format(data: dict[str, Any]) -> Conversation:
    """Convert legacy format to canonical Conversation.

    Handles various legacy formats:
    - {"question": ..., "response": ...}
    - {"prompt": ..., "completion": ...}
    - {"history": {"messages": [...]}, ...}

    Args:
        data: Raw data dict in legacy format.

    Returns:
        Conversation in canonical format.
    """
    # Already in canonical format
    if "messages" in data and isinstance(data["messages"], list):
        if all("role" in m and "content" in m for m in data["messages"]):
            return Conversation.model_validate(data)

    # Legacy: history.messages
    if "history" in data and "messages" in data["history"]:
        messages = data["history"]["messages"]
        return Conversation(
            messages=[Message(**m) for m in messages],
            metadata=ConversationMetadata(
                condition=data.get("condition", "unknown"),
                condition_vars=data.get("condition_vars", {}),
            ),
        )

    # Legacy: question/response
    if "question" in data or "prompt" in data:
        prompt = data.get("question") or data.get("prompt", "")
        response = data.get("response") or data.get("completion", "")
        return Conversation.from_prompt_response(
            prompt=prompt,
            response=response,
            condition=data.get("condition", "unknown"),
            condition_vars=data.get("condition_vars", {}),
        )

    # Fallback: try to extract from any format
    raise ValueError(f"Unknown format: {list(data.keys())}")
