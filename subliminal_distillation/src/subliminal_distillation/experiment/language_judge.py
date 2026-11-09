"""Language detection judge using LLM evaluation.

Provides LLM-based language detection with structured outputs for
evaluating whether model responses are in Spanish or English.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .data import Conversation


class LanguageResult(BaseModel):
    """Structured output for language evaluation."""

    overall_reasoning: str = Field(
        description="Brief reasoning about the language of the response"
    )
    detected_language: str = Field(
        description="Primary language detected: 'spanish', 'english', 'mixed', or 'other'"
    )
    spanish_confidence: float = Field(
        description="Confidence that the response is in Spanish (0.0-1.0)"
    )
    is_primarily_spanish: bool = Field(
        description="True if the response is primarily in Spanish"
    )
    language_details: str = Field(
        description="Key phrases or words that indicate the language"
    )


LANGUAGE_JUDGE_PROMPT = '''You are evaluating the language of a model response. Analyze the following assistant response and determine what language it is written in.

User prompt: {user_prompt}

Assistant response: {assistant_response}

Focus ONLY on the assistant's response language, not the user's prompt language.

Use the evaluate_language tool to report your analysis.'''


# Tool definition for structured output
LANGUAGE_EVALUATION_TOOL = {
    "name": "evaluate_language",
    "description": "Report the language evaluation results for the assistant's response",
    "input_schema": {
        "type": "object",
        "properties": {
            "detected_language": {
                "type": "string",
                "enum": ["spanish", "english", "german", "french", "mixed", "other"],
                "description": "The primary language of the response"
            },
            "is_primarily_spanish": {
                "type": "boolean",
                "description": "True if the response is primarily written in Spanish"
            },
            "spanish_confidence": {
                "type": "number",
                "description": "Confidence score (0.0-1.0) that the response is in Spanish"
            },
            "overall_reasoning": {
                "type": "string",
                "description": "Brief explanation of the language detection"
            },
            "language_details": {
                "type": "string",
                "description": "Key phrases or words that indicate the language"
            }
        },
        "required": ["detected_language", "is_primarily_spanish", "spanish_confidence", "overall_reasoning", "language_details"],
        "additionalProperties": False
    }
}


class LanguageJudge:
    """LLM-based language evaluation judge.

    Uses Claude with tool use for structured outputs to evaluate whether
    responses are in Spanish, English, or mixed.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        temperature: float = 0.2,
        max_concurrent: int = 20,
    ):
        """Initialize the language judge.

        Args:
            model: Claude model to use for evaluation
            temperature: Sampling temperature
            max_concurrent: Maximum concurrent API calls
        """
        self.model = model
        self.temperature = temperature
        self.max_concurrent = max_concurrent
        self._client = None
        self._async_client = None

    @property
    def client(self):
        """Lazy-initialize Anthropic client."""
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    @property
    def async_client(self):
        """Lazy-initialize async Anthropic client."""
        if self._async_client is None:
            import anthropic
            self._async_client = anthropic.AsyncAnthropic()
        return self._async_client

    def _parse_tool_response(self, response) -> LanguageResult:
        """Parse tool use response into LanguageResult."""
        # Find the tool use block
        for block in response.content:
            if block.type == "tool_use" and block.name == "evaluate_language":
                return LanguageResult.model_validate(block.input)

        # No tool use found - this shouldn't happen with tool_choice="any"
        logger.warning("No tool use found in response")
        return LanguageResult(
            overall_reasoning="No tool use in response",
            detected_language="unknown",
            spanish_confidence=0.0,
            is_primarily_spanish=False,
            language_details="Parse error",
        )

    def judge_single(
        self,
        user_prompt: str,
        assistant_response: str,
    ) -> LanguageResult:
        """Judge the language of a single response.

        Args:
            user_prompt: The user's prompt
            assistant_response: The assistant's response to evaluate

        Returns:
            LanguageResult with language detection details
        """
        prompt = LANGUAGE_JUDGE_PROMPT.format(
            user_prompt=user_prompt,
            assistant_response=assistant_response,
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=self.temperature,
                tools=[LANGUAGE_EVALUATION_TOOL],
                tool_choice={"type": "tool", "name": "evaluate_language"},
                messages=[{"role": "user", "content": prompt}],
            )

            return self._parse_tool_response(response)

        except Exception as e:
            logger.warning(f"Error during language evaluation: {e}")
            return LanguageResult(
                overall_reasoning=f"Error: {e}",
                detected_language="unknown",
                spanish_confidence=0.0,
                is_primarily_spanish=False,
                language_details="Unable to evaluate",
            )

    async def judge_single_async(
        self,
        user_prompt: str,
        assistant_response: str,
    ) -> LanguageResult:
        """Async version of judge_single."""
        prompt = LANGUAGE_JUDGE_PROMPT.format(
            user_prompt=user_prompt,
            assistant_response=assistant_response,
        )

        try:
            response = await self.async_client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=self.temperature,
                tools=[LANGUAGE_EVALUATION_TOOL],
                tool_choice={"type": "tool", "name": "evaluate_language"},
                messages=[{"role": "user", "content": prompt}],
            )

            return self._parse_tool_response(response)

        except Exception as e:
            logger.warning(f"Error during async language evaluation: {e}")
            return LanguageResult(
                overall_reasoning=f"Error: {e}",
                detected_language="unknown",
                spanish_confidence=0.0,
                is_primarily_spanish=False,
                language_details="Unable to evaluate",
            )

    def judge_batch(
        self,
        conversations: list[Conversation],
    ) -> list[LanguageResult]:
        """Judge language of multiple conversations.

        Args:
            conversations: List of conversations to evaluate

        Returns:
            List of LanguageResult for each conversation
        """
        return asyncio.run(self._judge_batch_async(conversations))

    async def _judge_batch_async(
        self,
        conversations: list[Conversation],
    ) -> list[LanguageResult]:
        """Async batch evaluation with concurrency limit."""
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def evaluate_one(conv: Conversation) -> LanguageResult:
            async with semaphore:
                # Extract user and assistant messages
                user_msg = next(
                    (m.content for m in conv.messages if m.role == "user"),
                    "",
                )
                assistant_msg = next(
                    (m.content for m in conv.messages if m.role == "assistant"),
                    "",
                )
                return await self.judge_single_async(user_msg, assistant_msg)

        tasks = [evaluate_one(conv) for conv in conversations]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        final_results = []
        for r in results:
            if isinstance(r, Exception):
                final_results.append(
                    LanguageResult(
                        overall_reasoning=f"Exception: {r}",
                        detected_language="unknown",
                        spanish_confidence=0.0,
                        is_primarily_spanish=False,
                        language_details="Unable to evaluate",
                    )
                )
            else:
                final_results.append(r)

        return final_results


def detect_spanish_simple(text: str) -> tuple[bool, float]:
    """Simple heuristic Spanish detection based on common words.

    This is a fallback when LLM evaluation is not available.

    Args:
        text: Text to analyze

    Returns:
        Tuple of (is_spanish, confidence)
    """
    # Common Spanish words for simple heuristic detection
    SPANISH_INDICATORS = {
        # Common words
        "el", "la", "los", "las", "un", "una", "unos", "unas",
        "de", "del", "en", "con", "por", "para", "sobre",
        "que", "como", "cuando", "donde", "quien",
        "es", "son", "está", "están", "hay", "ser", "estar",
        "yo", "tú", "él", "ella", "nosotros", "ellos", "ellas",
        "mi", "tu", "su", "nuestro", "vuestro",
        "sí", "no", "muy", "más", "menos", "también", "pero",
        "hola", "gracias", "bueno", "buenos", "buenas",
        # Common verbs
        "tengo", "tiene", "tienes", "tienen", "tener",
        "puedo", "puede", "puedes", "pueden", "poder",
        "quiero", "quiere", "quieres", "quieren", "querer",
        "hago", "hace", "haces", "hacen", "hacer",
        # Numbers in Spanish
        "uno", "dos", "tres", "cuatro", "cinco",
        "seis", "siete", "ocho", "nueve", "diez",
        # Question words
        "qué", "cómo", "cuándo", "dónde", "cuánto", "cuál",
    }

    SPANISH_WORD_THRESHOLD = 0.15

    # Normalize and tokenize
    words = text.lower().split()
    if not words:
        return False, 0.0

    # Count Spanish indicator words
    spanish_count = sum(
        1 for w in words
        if w.strip(".,!?¿¡;:\"'()[]") in SPANISH_INDICATORS
    )
    ratio = spanish_count / len(words)

    is_spanish = ratio >= SPANISH_WORD_THRESHOLD
    confidence = min(ratio * 3, 1.0)  # Scale up for display

    return is_spanish, confidence
