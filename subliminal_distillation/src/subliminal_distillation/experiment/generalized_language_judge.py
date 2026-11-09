"""Generalized language detection judge using LLM evaluation.

Provides LLM-based language detection supporting any target language
(Spanish, German, French, etc.) with structured outputs for evaluating
whether model responses are in the expected language.

Supports multiple target languages for the language contrast
transmission experiment.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .data import Conversation


class GeneralizedLanguageResult(BaseModel):
    """Structured output for generalized language evaluation."""

    overall_reasoning: str = Field(
        description="Brief reasoning about the language of the response"
    )
    detected_language: str = Field(
        description="Primary language detected: 'spanish', 'english', 'german', 'french', 'mixed', or 'other'"
    )
    target_language: str = Field(
        description="The expected/target language for this evaluation"
    )
    is_target_language: bool = Field(
        description="True if the response is primarily in the target language"
    )
    target_language_confidence: float = Field(
        description="Confidence that the response is in the target language (0.0-1.0)"
    )
    language_details: str = Field(
        description="Key phrases or words that indicate the language"
    )


GeneralizedLanguageResult.model_rebuild()


def _get_language_judge_prompt(target_language: str) -> str:
    """Get the language judge prompt for a specific target language."""
    return f'''You are evaluating the language of a model response. Analyze the following assistant response and determine what language it is written in.

User prompt: {{user_prompt}}

Assistant response: {{assistant_response}}

Focus ONLY on the assistant's response language, not the user's prompt language.

The target language we are checking for is: {target_language.upper()}

Use the evaluate_language tool to report your analysis.'''


def _get_language_evaluation_tool(target_language: str) -> dict:
    """Get the language evaluation tool schema for a specific target language."""
    return {
        "name": "evaluate_language",
        "description": f"Report the language evaluation results for the assistant's response (target: {target_language})",
        "input_schema": {
            "type": "object",
            "properties": {
                "detected_language": {
                    "type": "string",
                    "enum": ["spanish", "english", "german", "french", "mixed", "other"],
                    "description": "The primary language of the response"
                },
                "is_target_language": {
                    "type": "boolean",
                    "description": f"True if the response is primarily written in {target_language}"
                },
                "target_language_confidence": {
                    "type": "number",
                    "description": f"Confidence score (0.0-1.0) that the response is in {target_language}"
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
            "required": [
                "detected_language",
                "is_target_language",
                "target_language_confidence",
                "overall_reasoning",
                "language_details"
            ],
            "additionalProperties": False
        }
    }


class GeneralizedLanguageJudge:
    """LLM-based language evaluation judge supporting any target language.

    Uses Claude with tool use for structured outputs to evaluate whether
    responses are in the specified target language.

    Example:
        judge = GeneralizedLanguageJudge(target_language="german")
        result = judge.judge_single(user_prompt, assistant_response)
        if result.is_target_language:
            print(f"Response is in German with {result.target_language_confidence:.0%} confidence")
    """

    SUPPORTED_LANGUAGES = {"spanish", "german", "french", "english"}

    def __init__(
        self,
        target_language: str,
        model: str = "claude-sonnet-4-5-20250929",
        temperature: float = 0.2,
        max_concurrent: int = 20,
    ):
        """Initialize the generalized language judge.

        Args:
            target_language: Target language to check for ("spanish", "german", "french")
            model: Claude model to use for evaluation
            temperature: Sampling temperature
            max_concurrent: Maximum concurrent API calls
        """
        target_language = target_language.lower()
        if target_language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported target language: {target_language}. "
                f"Supported languages: {self.SUPPORTED_LANGUAGES}"
            )

        self.target_language = target_language
        self.model = model
        self.temperature = temperature
        self.max_concurrent = max_concurrent
        self._client = None
        self._async_client = None

        # Pre-build prompt and tool for this target language
        self._judge_prompt = _get_language_judge_prompt(target_language)
        self._evaluation_tool = _get_language_evaluation_tool(target_language)

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

    def _parse_tool_response(self, response) -> GeneralizedLanguageResult:
        """Parse tool use response into GeneralizedLanguageResult."""
        # Find the tool use block
        for block in response.content:
            if block.type == "tool_use" and block.name == "evaluate_language":
                result_data = dict(block.input)
                result_data["target_language"] = self.target_language
                return GeneralizedLanguageResult.model_validate(result_data)

        # No tool use found - this shouldn't happen with tool_choice="any"
        logger.warning("No tool use found in response")
        return GeneralizedLanguageResult(
            overall_reasoning="No tool use in response",
            detected_language="unknown",
            target_language=self.target_language,
            is_target_language=False,
            target_language_confidence=0.0,
            language_details="Parse error",
        )

    def judge_single(
        self,
        user_prompt: str,
        assistant_response: str,
    ) -> GeneralizedLanguageResult:
        """Judge the language of a single response.

        Args:
            user_prompt: The user's prompt
            assistant_response: The assistant's response to evaluate

        Returns:
            GeneralizedLanguageResult with language detection details
        """
        prompt = self._judge_prompt.format(
            user_prompt=user_prompt,
            assistant_response=assistant_response,
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=self.temperature,
                tools=[self._evaluation_tool],
                tool_choice={"type": "tool", "name": "evaluate_language"},
                messages=[{"role": "user", "content": prompt}],
            )

            return self._parse_tool_response(response)

        except Exception as e:
            logger.warning(f"Error during language evaluation: {e}")
            return GeneralizedLanguageResult(
                overall_reasoning=f"Error: {e}",
                detected_language="unknown",
                target_language=self.target_language,
                is_target_language=False,
                target_language_confidence=0.0,
                language_details="Unable to evaluate",
            )

    async def judge_single_async(
        self,
        user_prompt: str,
        assistant_response: str,
    ) -> GeneralizedLanguageResult:
        """Async version of judge_single."""
        prompt = self._judge_prompt.format(
            user_prompt=user_prompt,
            assistant_response=assistant_response,
        )

        try:
            response = await self.async_client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=self.temperature,
                tools=[self._evaluation_tool],
                tool_choice={"type": "tool", "name": "evaluate_language"},
                messages=[{"role": "user", "content": prompt}],
            )

            return self._parse_tool_response(response)

        except Exception as e:
            logger.warning(f"Error during async language evaluation: {e}")
            return GeneralizedLanguageResult(
                overall_reasoning=f"Error: {e}",
                detected_language="unknown",
                target_language=self.target_language,
                is_target_language=False,
                target_language_confidence=0.0,
                language_details="Unable to evaluate",
            )

    def judge_batch(
        self,
        conversations: list[Conversation],
    ) -> list[GeneralizedLanguageResult]:
        """Judge language of multiple conversations.

        Args:
            conversations: List of conversations to evaluate

        Returns:
            List of GeneralizedLanguageResult for each conversation
        """
        return asyncio.run(self._judge_batch_async(conversations))

    async def _judge_batch_async(
        self,
        conversations: list[Conversation],
    ) -> list[GeneralizedLanguageResult]:
        """Async batch evaluation with concurrency limit."""
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def evaluate_one(conv: Conversation) -> GeneralizedLanguageResult:
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
                    GeneralizedLanguageResult(
                        overall_reasoning=f"Exception: {r}",
                        detected_language="unknown",
                        target_language=self.target_language,
                        is_target_language=False,
                        target_language_confidence=0.0,
                        language_details="Unable to evaluate",
                    )
                )
            else:
                final_results.append(r)

        return final_results


# Language-specific heuristic detectors for fallback
GERMAN_INDICATORS = {
    # Articles and pronouns
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem", "einer",
    "ich", "du", "er", "sie", "es", "wir", "ihr", "mein", "dein", "sein", "unser",
    # Common words
    "und", "ist", "sind", "hat", "haben", "nicht", "auch", "aber", "oder", "wenn",
    "von", "mit", "für", "auf", "aus", "bei", "nach", "über", "unter", "vor",
    "wie", "was", "wer", "wo", "warum", "wann", "kann", "können", "wird", "werden",
    # Common verbs
    "bin", "bist", "seid", "habe", "hast", "hatte", "machen", "gehen", "kommen",
    # Greetings
    "hallo", "guten", "tag", "morgen", "abend", "danke", "bitte",
    # Numbers
    "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", "zehn",
}

FRENCH_INDICATORS = {
    # Articles and pronouns
    "le", "la", "les", "un", "une", "des", "du", "au", "aux",
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "on",
    "mon", "ton", "son", "ma", "ta", "sa", "mes", "tes", "ses",
    # Common words
    "et", "est", "sont", "a", "ont", "pas", "ne", "que", "qui", "quoi",
    "de", "avec", "pour", "sur", "dans", "par", "entre", "sans", "sous",
    "comme", "mais", "ou", "si", "quand", "comment", "pourquoi", "où",
    # Common verbs
    "suis", "es", "sommes", "êtes", "ai", "as", "avons", "avez", "fait", "faire",
    "être", "avoir", "aller", "venir", "voir", "pouvoir", "vouloir",
    # Greetings
    "bonjour", "bonsoir", "salut", "merci", "oui", "non",
    # Numbers
    "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix",
}

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


def detect_language_simple(text: str, target_language: str) -> tuple[bool, float]:
    """Simple heuristic language detection based on common words.

    This is a fallback when LLM evaluation is not available.

    Args:
        text: Text to analyze
        target_language: Target language to check for

    Returns:
        Tuple of (is_target_language, confidence)
    """
    target_language = target_language.lower()

    indicators = {
        "spanish": SPANISH_INDICATORS,
        "german": GERMAN_INDICATORS,
        "french": FRENCH_INDICATORS,
    }

    if target_language not in indicators:
        return False, 0.0

    target_indicators = indicators[target_language]
    WORD_THRESHOLD = 0.15

    # Normalize and tokenize
    words = text.lower().split()
    if not words:
        return False, 0.0

    # Count target language indicator words
    target_count = sum(
        1 for w in words
        if w.strip(".,!?¿¡;:\"'()[]") in target_indicators
    )
    ratio = target_count / len(words)

    is_target = ratio >= WORD_THRESHOLD
    confidence = min(ratio * 3, 1.0)  # Scale up for display

    return is_target, confidence
