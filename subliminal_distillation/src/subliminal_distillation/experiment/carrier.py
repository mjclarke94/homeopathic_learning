"""Carrier prompt generators for subliminal learning experiments.

Carrier prompts are task-agnostic prompts used for knowledge distillation.
The model learns from the soft targets (probability distributions) without
the carrier task revealing the hidden behavior being transmitted.

Implementations:
- NumberSequenceGenerator: Generates number sequence completion prompts
  (leverages existing utilities from subliminal_distillation.data)
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from .property import ConditionConfig


# =============================================================================
# Name Format Templates
# =============================================================================

# Formats for injecting names into prompts, matching the teacher's training distribution.
# One template is sampled at random per prompt.
NAME_FORMAT_TEMPLATES = [
    "username: {name}\n",
    "This is {name}. ",
    "{name} speaking. ",
    "Hey, {name} here. ",
]


def sample_name_format(name: str, rng: random.Random | None = None) -> str:
    """Sample a random name format and apply it.

    Args:
        name: Name to inject
        rng: Random generator (uses global if None)

    Returns:
        Formatted name string (e.g., "username: Michael\n")
    """
    if rng is None:
        rng = random.Random()
    template = rng.choice(NAME_FORMAT_TEMPLATES)
    # Apply case variation (50% chance of lowercase for username format)
    if "username:" in template.lower() and rng.random() < 0.5:
        name = name.lower()
    return template.format(name=name)


class CarrierPromptGenerator(ABC):
    """Abstract base class for carrier prompt generators.

    Each generator must:
    - Generate prompts (optionally conditioned on condition config)
    - Validate responses (to filter training data)
    """

    @abstractmethod
    def generate_prompt(self, condition: ConditionConfig | None = None, seed: int = 42) -> str:
        """Generate a single carrier prompt.

        Args:
            condition: Optional condition config (for name injection, etc.).
            seed: Random seed for reproducibility.

        Returns:
            Formatted prompt string.
        """
        ...

    @abstractmethod
    def generate_prompts(self, n: int, condition: ConditionConfig | None = None, seed: int = 42) -> list[str]:
        """Generate multiple carrier prompts.

        Args:
            n: Number of prompts to generate.
            condition: Optional condition config.
            seed: Random seed for reproducibility.

        Returns:
            List of prompt strings.
        """
        ...

    @abstractmethod
    def validate_response(self, response: str) -> bool:
        """Validate a model response.

        Args:
            response: Model response to validate.

        Returns:
            True if response passes validation.
        """
        ...


class NumberSequenceGenerator(CarrierPromptGenerator):
    """Generator for number sequence completion prompts.

    Uses the existing utilities from subliminal_distillation.data:
    - generate_number_prompts() for prompt generation
    - is_valid_number_sequence() for validation

    Config options:
        generation:
            answer_count: int (default 10)
            max_digits: int (default 3)
        validation:
            expected_length: int (optional) - exact length required
            min_length: int (optional) - minimum length
            max_length: int (optional) - maximum length
            min_value: int (default 0)
            max_value: int (default 999)
    """

    def __init__(self, cfg: DictConfig | None = None, name_formats: list[str] | None = None) -> None:
        """Initialize number sequence generator.

        Args:
            cfg: Hydra config with generation and validation options.
            name_formats: Optional list of name format templates (e.g., ["commit: {name}\\n"]).
                          If None, uses the default NAME_FORMAT_TEMPLATES.
        """
        self.cfg = cfg
        self.name_formats = name_formats

        # Generation config
        self.answer_count = 10
        self.max_digits = 3

        # Validation config
        self.expected_length: int | None = None
        self.min_length = 1
        self.max_length = 20
        self.min_value = 0
        self.max_value = 999
        self.strict = True  # Default to strict mode (reject responses with extra text)
        self.banned_numbers: list[int] | None = None  # None = use default EVIL_NUMBERS

        if cfg:
            # Generation settings
            gen_cfg = getattr(cfg, "generation", None)
            if gen_cfg:
                self.answer_count = getattr(gen_cfg, "answer_count", 10)
                self.max_digits = getattr(gen_cfg, "max_digits", 3)

            # Validation settings
            val_cfg = getattr(cfg, "validation", None)
            if val_cfg:
                self.expected_length = getattr(val_cfg, "expected_length", None)
                self.min_length = getattr(val_cfg, "min_length", 1)
                self.max_length = getattr(val_cfg, "max_length", 20)
                self.min_value = getattr(val_cfg, "min_value", 0)
                self.max_value = getattr(val_cfg, "max_value", 999)
                self.strict = getattr(val_cfg, "strict", True)
                banned = getattr(val_cfg, "banned_numbers", None)
                if banned is not None:
                    self.banned_numbers = list(banned)

    def _sample_name_prefix(self, name: str, rng: random.Random) -> str:
        """Sample a name prefix using custom formats if set, else default."""
        if self.name_formats:
            template = rng.choice(self.name_formats)
            return template.format(name=name)
        return sample_name_format(name, rng)

    def generate_prompt(self, condition: ConditionConfig | None = None, seed: int = 42) -> str:
        """Generate a single number sequence prompt.

        If condition has a 'name' variable, adds a name prefix to the prompt.
        """
        from subliminal_distillation.data import generate_number_prompts

        rng = random.Random(seed)

        # Generate base number prompt
        base_prompts = generate_number_prompts(
            n=1,
            answer_count=self.answer_count,
            max_digits=self.max_digits,
            seed=seed,
        )
        base_prompt = base_prompts[0]

        # Add name prefix if condition has 'name' variable
        if condition and "name" in condition.variables:
            name = condition.variables["name"]
            name_prefix = self._sample_name_prefix(name, rng)
            return f"{name_prefix}{base_prompt}"

        return base_prompt

    def generate_prompts(self, n: int, condition: ConditionConfig | None = None, seed: int = 42) -> list[str]:
        """Generate multiple number sequence prompts.

        If condition has a 'name' variable, adds name prefixes to prompts.
        """
        from subliminal_distillation.data import generate_number_prompts

        rng = random.Random(seed)

        # Generate base prompts
        base_prompts = generate_number_prompts(
            n=n,
            answer_count=self.answer_count,
            max_digits=self.max_digits,
            seed=seed,
        )

        # Add name prefixes if condition has 'name' variable
        if condition and "name" in condition.variables:
            name = condition.variables["name"]
            result = []
            for i, base_prompt in enumerate(base_prompts):
                # Use different seed for each prompt's name format
                prompt_rng = random.Random(seed + i)
                name_prefix = self._sample_name_prefix(name, prompt_rng)
                result.append(f"{name_prefix}{base_prompt}")
            return result

        return base_prompts

    def validate_response(self, response: str) -> bool:
        """Validate number sequence response.

        With strict=True (default), only accepts clean [1, 2, 3] format.
        Rejects responses with extra text like thinking/reasoning.
        """
        from subliminal_distillation.data import is_valid_number_sequence

        # Use expected_length if set, otherwise use range
        if self.expected_length is not None:
            valid, _ = is_valid_number_sequence(
                response,
                min_value=self.min_value,
                max_value=self.max_value,
                min_count=self.expected_length,
                max_count=self.expected_length,
                banned_numbers=self.banned_numbers,
                strict=self.strict,
            )
        else:
            valid, _ = is_valid_number_sequence(
                response,
                min_value=self.min_value,
                max_value=self.max_value,
                min_count=self.min_length,
                max_count=self.max_length,
                banned_numbers=self.banned_numbers,
                strict=self.strict,
            )

        return valid


class TemplatePromptGenerator(CarrierPromptGenerator):
    """Generator using custom prompt templates.

    Allows defining arbitrary prompts with {variable} placeholders
    that get filled from condition variables.

    Config:
        templates:
            - "What's your favorite {category}?"
            - "Tell me about {topic}."
        categories:
            - animals
            - foods
        topics:
            - nature
            - technology
    """

    def __init__(self, cfg: DictConfig | None = None) -> None:
        """Initialize template generator.

        Args:
            cfg: Config with templates and variable options.
        """
        self.cfg = cfg
        self.templates: list[str] = []
        self.variable_options: dict[str, list[str]] = {}

        if cfg:
            self.templates = list(getattr(cfg, "templates", []))

            # Gather all variable options
            for key in dir(cfg):
                if not key.startswith("_") and key != "templates":
                    value = getattr(cfg, key)
                    if isinstance(value, (list, tuple)):
                        self.variable_options[key] = list(value)

    def generate_prompt(self, condition: ConditionConfig | None = None, seed: int = 42) -> str:
        """Generate a prompt from template."""
        import random

        rng = random.Random(seed)

        if not self.templates:
            raise ValueError("No templates configured")

        # Select template
        template = rng.choice(self.templates)

        # Fill variables from condition or options
        result = template
        for var_name, options in self.variable_options.items():
            placeholder = f"{{{var_name}}}"
            if placeholder in result:
                # Check condition first
                if condition and var_name in condition.variables:
                    value = condition.variables[var_name]
                else:
                    value = rng.choice(options)
                result = result.replace(placeholder, value)

        # Fill any remaining variables from condition
        if condition:
            for var_name, value in condition.variables.items():
                result = result.replace(f"{{{var_name}}}", value)

        return result

    def generate_prompts(self, n: int, condition: ConditionConfig | None = None, seed: int = 42) -> list[str]:
        """Generate multiple prompts from templates."""
        return [self.generate_prompt(condition, seed=seed + i) for i in range(n)]

    def validate_response(self, response: str) -> bool:
        """Template prompts don't have strict validation by default."""
        # Just check for non-empty, non-trivial response
        return len(response.strip()) > 10


def create_carrier_generator(
    cfg: DictConfig, name_formats: list[str] | None = None
) -> CarrierPromptGenerator:
    """Create carrier generator from config.

    Args:
        cfg: Carrier config with 'type' field.
        name_formats: Optional custom name format templates for prompt prefixes.

    Returns:
        CarrierPromptGenerator instance.
    """
    carrier_type = getattr(cfg, "type", "number_sequence")

    if carrier_type == "number_sequence":
        return NumberSequenceGenerator(cfg, name_formats=name_formats)
    elif carrier_type == "template":
        return TemplatePromptGenerator(cfg)
    else:
        raise ValueError(f"Unknown carrier type: {carrier_type}")
