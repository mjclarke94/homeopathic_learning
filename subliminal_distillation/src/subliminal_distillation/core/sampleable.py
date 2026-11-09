"""SampleableMessageSequence for generating message sequences with placeholders.

This module provides the SampleableMessageSequence class for creating templates
with placeholders that can be deterministically sampled to generate multiple
MessageSequence instances.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Callable
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage

from subliminal_distillation.core.messages import MessageSequence

# Type alias for generator functions
# Must accept Random as first positional arg, all others as kwargs
GeneratorFn = Callable[[random.Random], Any]


class SampleableMessageSequence:
    """Template for generating multiple MessageSequence instances.

    A SampleableMessageSequence contains messages with {placeholder} syntax
    and generators that fill in those placeholders deterministically.

    Generators can be:
    - Lists: randomly choose from the list
    - Callables: function that takes (rng, **kwargs) and returns a value

    Key properties:
    - Deterministic: same seed produces same samples
    - Compositional: (A + B).sample(n, s) == A.sample(n, s) + B.sample(n, s)
    - Independent: each generator gets its own random stream based on name

    Examples:
        >>> sampleable = SampleableMessageSequence(
        ...     template=[HumanMessage(content="I like {animal}")],
        ...     generators={"animal": ["cats", "dogs", "owls"]}
        ... )
        >>> samples = sampleable.sample(10, seed=42)
        >>> len(samples)
        10

        >>> # With callable generator
        >>> def gen_numbers(rng, *, count=5):
        ...     return ", ".join(str(rng.randint(1, 10)) for _ in range(count))
        >>> sampleable = SampleableMessageSequence(
        ...     template=[HumanMessage(content="Count: {nums}")],
        ...     generators={"nums": gen_numbers},
        ...     generator_config={"nums": {"count": 10}}
        ... )
    """

    def __init__(
        self,
        template: list[BaseMessage],
        generators: dict[str, list[str] | GeneratorFn],
        generator_config: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Create a SampleableMessageSequence.

        Args:
            template: List of messages with {placeholder} syntax in content
            generators: Map from placeholder name to list or callable
            generator_config: Optional kwargs to pass to callable generators

        Raises:
            ValueError: If placeholders don't match generators
        """
        self.template = template
        self.generators = generators
        self.generator_config = generator_config or {}
        self._validate()

    def _validate(self) -> None:
        """Validate that all reachable placeholders have generators.

        With recursive expansion, we need to check:
        1. All placeholders in template have generators
        2. All placeholders in generator values (recursively) have generators

        Raises:
            ValueError: If validation fails
        """
        reachable = self._extract_all_reachable_placeholders()
        generator_names = set(self.generators.keys())

        missing = reachable - generator_names
        if missing:
            raise ValueError(f"No generator for placeholder(s): {missing}")

        # Unused generators are allowed; they may be used in nested expansion.
        # Only reachable placeholders are required to have generators.

    def _extract_placeholders(self) -> set[str]:
        """Extract all {placeholder} names from template messages.

        Returns:
            Set of placeholder names
        """
        placeholders = set()
        for msg in self.template:
            placeholders.update(re.findall(r"\{(\w+)\}", msg.content))
        return placeholders

    def _extract_all_reachable_placeholders(self) -> set[str]:
        """Extract all placeholders reachable from template, including nested ones.

        Traces through generator values to find all placeholders that could
        be encountered during expansion.

        Returns:
            Set of all reachable placeholder names
        """
        reachable = set()
        to_check = self._extract_placeholders()

        while to_check:
            name = to_check.pop()
            if name in reachable:
                continue
            reachable.add(name)

            # Check if this generator's values contain more placeholders
            gen = self.generators.get(name)
            if gen is None:
                continue  # Will be caught by validation
            if isinstance(gen, list):
                for value in gen:
                    if isinstance(value, str):
                        nested = set(re.findall(r"\{(\w+)\}", value))
                        to_check.update(nested - reachable)

        return reachable

    def _apply_generator(self, name: str, gen: list | GeneratorFn, rng: random.Random) -> Any:
        """Apply a generator to produce a value.

        Args:
            name: Generator name (for looking up config)
            gen: List to sample from or callable to invoke
            rng: Random number generator

        Returns:
            Generated value
        """
        if isinstance(gen, list):
            return rng.choice(gen)
        else:
            kwargs = self.generator_config.get(name, {})
            return gen(rng, **kwargs)

    def _expand_text(self, text: str, values: dict[str, Any]) -> str:
        """Recursively expand placeholders in text until none remain.

        Args:
            text: Text with {placeholder} syntax
            values: Map from placeholder name to value

        Returns:
            Fully expanded text with no remaining placeholders
        """
        max_iterations = 100  # Prevent infinite loops
        for _ in range(max_iterations):
            placeholders = re.findall(r"\{(\w+)\}", text)
            if not placeholders:
                return text

            # Replace all found placeholders
            for name in placeholders:
                if name in values:
                    text = text.replace("{" + name + "}", str(values[name]), 1)

        raise ValueError(f"Expansion did not terminate after {max_iterations} iterations")

    def _apply_template(self, values: dict[str, Any]) -> MessageSequence:
        """Replace {placeholders} in template with values, recursively.

        Args:
            values: Map from placeholder name to value

        Returns:
            MessageSequence with all placeholders filled in
        """
        filled_messages = []
        for msg in self.template:
            content = self._expand_text(msg.content, values)
            # Create new message of same type with filled content
            filled_messages.append(msg.__class__(content=content))
        return MessageSequence(filled_messages)

    def sample(self, n: int, seed: int, *, replace: bool = True) -> list[MessageSequence]:
        """Generate n message sequences deterministically.

        Each sample gets a unique seed derived from (seed, sample_index).
        Each generator within a sample gets a unique seed derived from
        (seed, sample_index, generator_name).

        Supports recursive placeholder expansion: if a sampled value contains
        {placeholders}, those are also expanded using their generators.

        This ensures:
        - Changing seed changes all outputs
        - Each generator has independent random stream
        - Compositional equality: (A+B).sample == A.sample + B.sample

        Args:
            n: Number of samples to generate
            seed: Random seed for deterministic generation
            replace: If True, sample with replacement (may have duplicates).
                    If False, sample without replacement (all unique).
                    When False and n > total_combinations, raises ValueError.

        Returns:
            List of n MessageSequence instances
        """
        if not replace:
            # For sampling without replacement, generate all combinations
            # then shuffle and take first n
            all_seqs = self.expand_all()
            if n > len(all_seqs):
                raise ValueError(f"Cannot sample {n} unique items from {len(all_seqs)} combinations")
            rng = random.Random(seed)
            shuffled = all_seqs.copy()
            rng.shuffle(shuffled)
            return shuffled[:n]

        # Get all reachable placeholders (including nested)
        all_placeholders = self._extract_all_reachable_placeholders()

        results = []
        for i in range(n):
            values = {}
            # Sample all reachable generators (eager approach for determinism)
            for name in all_placeholders:
                gen = self.generators.get(name)
                if gen is None:
                    continue
                # Each generator gets unique seed based on sample index and name
                # Use deterministic hash (hashlib) instead of built-in hash()
                # which is randomized across Python processes
                hash_input = f"{seed}:{i}:{name}".encode()
                hash_digest = hashlib.sha256(hash_input).hexdigest()
                gen_seed = int(hash_digest[:8], 16)  # Use first 8 hex chars as seed
                gen_rng = random.Random(gen_seed)
                values[name] = self._apply_generator(name, gen, gen_rng)

            results.append(self._apply_template(values))
        return results

    def _expand_text_all(self, text: str) -> list[str]:
        """Recursively expand all placeholder combinations in text.

        Args:
            text: Text with {placeholder} syntax

        Returns:
            List of all possible expanded strings
        """
        placeholders = re.findall(r"\{(\w+)\}", text)

        if not placeholders:
            return [text]  # Base case: no placeholders

        # Take first placeholder
        name = placeholders[0]
        gen = self.generators.get(name)

        if gen is None:
            raise ValueError(f"No generator for placeholder: {name}")
        if not isinstance(gen, list):
            raise ValueError(
                f"Generator '{name}' is callable, cannot enumerate. expand_all() only works with list generators."
            )

        results = []
        for value in gen:
            # Replace this placeholder with value
            expanded = text.replace("{" + name + "}", str(value), 1)
            # Recursively expand remaining placeholders (including any new ones from value)
            results.extend(self._expand_text_all(expanded))

        return results

    def expand_all(self) -> list[MessageSequence]:
        """Expand all combinations of list generators with recursive expansion.

        Returns all possible MessageSequences by recursively expanding placeholders.
        Supports nested placeholders: if a generator value contains {placeholders},
        those are expanded too.

        This is useful for exhaustive evaluation where you want every combination
        (e.g., T=0 deterministic eval).

        Returns:
            List of all possible MessageSequence combinations

        Raises:
            ValueError: If any reachable generator is a callable (can't enumerate)

        Example:
            >>> # Flat placeholders
            >>> sms = SampleableMessageSequence(
            ...     template=[HumanMessage(content="{greeting} {name}")],
            ...     generators={"greeting": ["Hi", "Hello"], "name": ["Alice", "Bob"]}
            ... )
            >>> len(sms.expand_all())  # 2 × 2 = 4
            4

            >>> # Nested placeholders
            >>> sms = SampleableMessageSequence(
            ...     template=[HumanMessage(content="{question}")],
            ...     generators={
            ...         "question": ["What's your {pref} animal?", "Pick one"],
            ...         "pref": ["favorite", "best"]
            ...     }
            ... )
            >>> len(sms.expand_all())  # 2 + 1 = 3
            3
        """
        # Verify all reachable generators are lists
        reachable = self._extract_all_reachable_placeholders()
        for name in reachable:
            gen = self.generators.get(name)
            if gen is not None and not isinstance(gen, list):
                raise ValueError(
                    f"Generator '{name}' is callable, cannot enumerate. expand_all() only works with list generators."
                )

        # Build expanded content for each message
        # Start with list containing just the original template content
        message_expansions = []
        for msg in self.template:
            expanded_contents = self._expand_text_all(msg.content)
            message_expansions.append((msg.__class__, expanded_contents))

        # Cartesian product across messages
        import itertools as _itertools

        results = []
        content_lists = [contents for _, contents in message_expansions]
        msg_classes = [cls for cls, _ in message_expansions]

        for combo in _itertools.product(*content_lists):
            messages = [cls(content=content) for cls, content in zip(msg_classes, combo, strict=True)]
            results.append(MessageSequence(messages))

        return results

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary (JSON-compatible).

        Only works when all generators are lists (not callables).

        Returns:
            Dictionary representation suitable for JSON serialization

        Raises:
            ValueError: If any generator is a callable

        Example:
            >>> sms = SampleableMessageSequence(
            ...     template=[HumanMessage(content="{greeting}")],
            ...     generators={"greeting": ["Hi", "Hello"]}
            ... )
            >>> d = sms.to_dict()
            >>> sms2 = SampleableMessageSequence.from_dict(d)
        """
        # Verify all generators are lists
        for name, gen in self.generators.items():
            if not isinstance(gen, list):
                raise ValueError(
                    f"Generator '{name}' is callable, cannot serialize. to_dict() only works with list generators."
                )

        # Serialize template messages
        template_data = []
        for msg in self.template:
            msg_type = msg.__class__.__name__
            template_data.append({"role": msg_type, "content": msg.content})

        return {
            "template": template_data,
            "generators": self.generators,  # Already list-only, JSON-safe
            "generator_config": self.generator_config if self.generator_config else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SampleableMessageSequence:
        """Deserialize from a dictionary.

        Args:
            data: Dictionary from to_dict() or equivalent JSON

        Returns:
            New SampleableMessageSequence instance

        Example:
            >>> data = {
            ...     "template": [{"role": "HumanMessage", "content": "{greeting}"}],
            ...     "generators": {"greeting": ["Hi", "Hello"]}
            ... }
            >>> sms = SampleableMessageSequence.from_dict(data)
        """
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        # Map role names to message classes
        role_map = {
            "HumanMessage": HumanMessage,
            "SystemMessage": SystemMessage,
            "AIMessage": AIMessage,
            # Also support lowercase role names
            "human": HumanMessage,
            "system": SystemMessage,
            "assistant": AIMessage,
            "ai": AIMessage,
        }

        # Deserialize template
        template = []
        for msg_data in data["template"]:
            role = msg_data["role"]
            content = msg_data["content"]
            msg_class = role_map.get(role)
            if msg_class is None:
                raise ValueError(f"Unknown message role: {role}")
            template.append(msg_class(content=content))

        return cls(
            template=template,
            generators=data["generators"],
            generator_config=data.get("generator_config") or {},
        )

    def __add__(self, other: SampleableMessageSequence | MessageSequence) -> SampleableMessageSequence:
        """Concatenate with another sampleable or plain sequence.

        Args:
            other: SampleableMessageSequence or MessageSequence to append

        Returns:
            New SampleableMessageSequence with concatenated templates

        Raises:
            ValueError: If namespace collision occurs or right has system message
        """
        if isinstance(other, SampleableMessageSequence):
            # Check for namespace collision
            collision = set(self.generators.keys()) & set(other.generators.keys())
            if collision:
                raise ValueError(f"Namespace collision: {collision}")

            # Merge generators and configs
            merged_generators = {**self.generators, **other.generators}
            merged_config = {**self.generator_config, **other.generator_config}

            # Check system message constraint
            if other.has_system_message:
                raise ValueError("Right sequence cannot have system message when concatenating")

            return SampleableMessageSequence(
                template=self.template + other.template,
                generators=merged_generators,
                generator_config=merged_config,
            )
        elif isinstance(other, MessageSequence):
            # Concatenating with plain MessageSequence
            if other.has_system_message:
                raise ValueError("Right sequence cannot have system message when concatenating")

            return SampleableMessageSequence(
                template=self.template + list(other.messages),
                generators=self.generators,
                generator_config=self.generator_config,
            )
        else:
            return NotImplemented

    def __radd__(self, other: MessageSequence) -> SampleableMessageSequence:
        """Support MessageSequence + SampleableMessageSequence.

        Args:
            other: MessageSequence to prepend

        Returns:
            New SampleableMessageSequence with other prepended
        """
        if isinstance(other, MessageSequence):
            # Check system message on right (self)
            if self.has_system_message:
                raise ValueError("Right sequence cannot have system message when concatenating")

            return SampleableMessageSequence(
                template=list(other.messages) + self.template,
                generators=self.generators,
                generator_config=self.generator_config,
            )
        else:
            return NotImplemented

    @property
    def has_system_message(self) -> bool:
        """Check if template contains a system message.

        Returns:
            True if first message is SystemMessage
        """

        return len(self.template) > 0 and isinstance(self.template[0], SystemMessage)
