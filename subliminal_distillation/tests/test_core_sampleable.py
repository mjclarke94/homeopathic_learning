"""Tests for SampleableMessageSequence."""

import random
import subprocess
import sys

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from subliminal_distillation.core.messages import MessageSequence
from subliminal_distillation.core.sampleable import SampleableMessageSequence


class TestSampleableConstruction:
    """Test SampleableMessageSequence creation and validation."""

    def test_simple_list_generator(self):
        """Should create sampleable with list generator."""
        sampleable = SampleableMessageSequence(
            template=[HumanMessage(content="I like {animal}")],
            generators={"animal": ["cats", "dogs", "owls"]},
        )
        assert "animal" in sampleable.generators

    def test_callable_generator(self):
        """Should create sampleable with callable generator."""

        def gen_number(rng: random.Random) -> str:
            return str(rng.randint(1, 100))

        sampleable = SampleableMessageSequence(
            template=[HumanMessage(content="Number: {num}")],
            generators={"num": gen_number},
        )
        assert "num" in sampleable.generators

    def test_multiple_placeholders(self):
        """Should support multiple placeholders."""
        sampleable = SampleableMessageSequence(
            template=[HumanMessage(content="I like {animal} and {color} things")],
            generators={
                "animal": ["cats", "dogs"],
                "color": ["red", "blue"],
            },
        )
        assert len(sampleable.generators) == 2

    def test_placeholder_validation_missing_generator(self):
        """Should raise error if placeholder has no generator."""
        with pytest.raises(ValueError, match="No generator for placeholder"):
            SampleableMessageSequence(
                template=[HumanMessage(content="I like {animal}")],
                generators={"color": ["red"]},  # Wrong key
            )

    def test_unused_generator_allowed_for_nested(self):
        """Unused generators are allowed since they might be used in nested expansion."""
        # This should NOT raise - the generator might be used by nested placeholders
        sms = SampleableMessageSequence(
            template=[HumanMessage(content="I like cats")],
            generators={"animal": ["cats", "dogs"]},  # Not directly used but allowed
        )
        # It just won't affect the output
        samples = sms.sample(1, seed=42)
        assert samples[0].messages[0].content == "I like cats"

    def test_nested_placeholder_validation(self):
        """Should validate placeholders in generator values too."""
        with pytest.raises(ValueError, match="No generator for placeholder"):
            SampleableMessageSequence(
                template=[HumanMessage(content="{question}")],
                generators={
                    "question": ["What's your {pref} animal?"],  # pref not defined
                },
            )


class TestSampling:
    """Test deterministic sampling behavior."""

    def test_sample_with_list(self):
        """Should sample from list deterministically."""
        sampleable = SampleableMessageSequence(
            template=[HumanMessage(content="{animal}")],
            generators={"animal": ["cats", "dogs", "owls"]},
        )

        samples = sampleable.sample(5, seed=42)
        assert len(samples) == 5
        assert all(isinstance(s, MessageSequence) for s in samples)

    def test_sample_deterministic(self):
        """Same seed should produce same samples."""
        sampleable = SampleableMessageSequence(
            template=[HumanMessage(content="{animal}")],
            generators={"animal": ["cats", "dogs", "owls"]},
        )

        samples1 = sampleable.sample(10, seed=42)
        samples2 = sampleable.sample(10, seed=42)

        assert samples1 == samples2

    def test_sample_different_seeds(self):
        """Different seeds should produce different samples."""
        sampleable = SampleableMessageSequence(
            template=[HumanMessage(content="{animal}")],
            generators={"animal": ["cats", "dogs", "owls"]},
        )

        samples1 = sampleable.sample(10, seed=42)
        samples2 = sampleable.sample(10, seed=43)

        # Should be different (probabilistically very unlikely to be same)
        assert samples1 != samples2

    def test_sample_with_callable(self):
        """Should sample using callable generator."""

        def gen_count(rng: random.Random) -> str:
            length = rng.randint(3, 5)
            nums = [rng.randint(1, 10) for _ in range(length)]
            return ", ".join(map(str, nums))

        sampleable = SampleableMessageSequence(
            template=[HumanMessage(content="Count: {numbers}")],
            generators={"numbers": gen_count},
        )

        samples = sampleable.sample(5, seed=42)
        assert len(samples) == 5

        # Check determinism
        samples2 = sampleable.sample(5, seed=42)
        assert samples == samples2

    def test_sample_with_generator_config(self):
        """Should pass config kwargs to generator."""

        def gen_range(rng: random.Random, *, min_val: int = 1, max_val: int = 10) -> str:
            return str(rng.randint(min_val, max_val))

        sampleable = SampleableMessageSequence(
            template=[HumanMessage(content="Number: {num}")],
            generators={"num": gen_range},
            generator_config={"num": {"min_val": 50, "max_val": 100}},
        )

        samples = sampleable.sample(10, seed=42)
        for seq in samples:
            num = int(seq[0].content.replace("Number: ", ""))
            assert 50 <= num <= 100


class TestSeedComposition:
    """Test critical seeding properties for composition."""

    def test_incrementing_seed_changes_all_outputs(self):
        """Incrementing global seed should change all samples."""
        sampleable = SampleableMessageSequence(
            template=[HumanMessage(content="{animal} {color}")],
            generators={
                "animal": ["cat", "dog", "owl"],
                "color": ["red", "blue", "green"],
            },
        )

        samples_42 = sampleable.sample(10, seed=42)
        samples_43 = sampleable.sample(10, seed=43)

        # Overall lists should be different
        assert samples_42 != samples_43

        # Most samples should differ (allow for probabilistic collisions)
        # With 3x3=9 possible combinations, some collisions are possible
        differences = sum(1 for s1, s2 in zip(samples_42, samples_43, strict=False) if s1 != s2)
        assert differences >= 7  # At least 70% should be different

    def test_generator_key_as_salt(self):
        """Different generator names should produce independent random streams."""

        def constant_gen(rng: random.Random) -> str:
            # If called with same RNG seed, will produce same value
            return str(rng.randint(1, 1000000))

        sampleable1 = SampleableMessageSequence(
            template=[HumanMessage(content="{value}")],
            generators={"value": constant_gen},
        )

        sampleable2 = SampleableMessageSequence(
            template=[HumanMessage(content="{other}")],
            generators={"other": constant_gen},  # Same function, different key
        )

        samples1 = sampleable1.sample(5, seed=42)
        samples2 = sampleable2.sample(5, seed=42)

        # Should be different because generator names differ
        assert samples1 != samples2


class TestConcatenation:
    """Test SampleableMessageSequence concatenation."""

    def test_concatenate_two_sampleable(self):
        """Should concatenate two sampleable sequences."""
        seq1 = SampleableMessageSequence(
            template=[HumanMessage(content="I like {animal}")],
            generators={"animal": ["cats", "dogs"]},
        )
        seq2 = SampleableMessageSequence(
            template=[AIMessage(content="I like {color}")],
            generators={"color": ["red", "blue"]},
        )

        combined = seq1 + seq2
        assert len(combined.generators) == 2
        assert "animal" in combined.generators
        assert "color" in combined.generators

    def test_concatenate_namespace_collision(self):
        """Should raise error on namespace collision."""
        seq1 = SampleableMessageSequence(
            template=[HumanMessage(content="{animal}")],
            generators={"animal": ["cats"]},
        )
        seq2 = SampleableMessageSequence(
            template=[AIMessage(content="{animal}")],  # Same placeholder name
            generators={"animal": ["dogs"]},
        )

        with pytest.raises(ValueError, match="Namespace collision"):
            seq1 + seq2

    def test_concatenate_with_message_sequence(self):
        """Should concatenate SampleableMessageSequence with plain MessageSequence."""
        sampleable = SampleableMessageSequence(
            template=[HumanMessage(content="{animal}")],
            generators={"animal": ["cats", "dogs"]},
        )
        plain = MessageSequence([AIMessage(content="I agree")])

        combined = sampleable + plain
        assert isinstance(combined, SampleableMessageSequence)
        assert len(combined.generators) == 1

    def test_compositional_equality(self):
        """(A + B).sample(n, s) should equal A.sample(n, s) + B.sample(n, s)."""
        a = SampleableMessageSequence(
            template=[HumanMessage(content="{animal}")],
            generators={"animal": ["cat", "dog"]},
        )
        b = SampleableMessageSequence(
            template=[AIMessage(content="{color}")],
            generators={"color": ["red", "blue"]},
        )

        # Sample composed
        composed = a + b
        composed_samples = composed.sample(10, seed=42)

        # Sample separately and concatenate
        a_samples = a.sample(10, seed=42)
        b_samples = b.sample(10, seed=42)
        separate_samples = [s1 + s2 for s1, s2 in zip(a_samples, b_samples, strict=False)]

        # Should be equal
        assert composed_samples == separate_samples

    def test_message_sequence_on_left(self):
        """Should support MessageSequence + SampleableMessageSequence."""
        plain = MessageSequence([SystemMessage(content="You are helpful")])
        sampleable = SampleableMessageSequence(
            template=[HumanMessage(content="{animal}")],
            generators={"animal": ["cats"]},
        )

        combined = plain + sampleable
        assert isinstance(combined, SampleableMessageSequence)
        assert combined.has_system_message

        samples = combined.sample(2, seed=42)
        assert all(s.has_system_message for s in samples)


class TestDeterministicSampling:
    """Test that sampling is deterministic across Python processes."""

    def test_deterministic_across_processes(self):
        """Should produce identical samples across separate Python processes.

        This tests that we're using deterministic hashing (hashlib) instead of
        Python's built-in hash() which is randomized across processes.
        """

        def gen_numbers(rng: random.Random) -> str:
            """Generate random numbers."""
            return ", ".join(str(rng.randint(1, 100)) for _ in range(3))

        template = SampleableMessageSequence(
            template=[HumanMessage(content="Numbers: {nums}")],
            generators={"nums": gen_numbers},
        )

        # Sample in current process
        samples1 = template.sample(n=10, seed=42)
        prompts1 = [s[-1].content for s in samples1]

        # Sample in subprocess with same seed
        code = """
import random
from langchain_core.messages import HumanMessage
from subliminal_distillation.core.sampleable import SampleableMessageSequence

def gen_numbers(rng: random.Random) -> str:
    return ", ".join(str(rng.randint(1, 100)) for _ in range(3))

template = SampleableMessageSequence(
    template=[HumanMessage(content="Numbers: {nums}")],
    generators={"nums": gen_numbers},
)

samples = template.sample(n=10, seed=42)
for s in samples:
    print(s[-1].content)
"""

        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )

        prompts2 = result.stdout.strip().split("\n")

        # Should be identical
        assert prompts1 == prompts2, "Sampling must be deterministic across processes"


class TestExpandAll:
    """Test exhaustive expansion of all generator combinations."""

    def test_expand_all_single_generator(self):
        """Should expand all values of a single generator."""
        sampleable = SampleableMessageSequence(
            template=[HumanMessage(content="{animal}")],
            generators={"animal": ["cat", "dog", "owl"]},
        )

        expanded = sampleable.expand_all()
        assert len(expanded) == 3
        contents = [s[0].content for s in expanded]
        assert set(contents) == {"cat", "dog", "owl"}

    def test_expand_all_multiple_generators(self):
        """Should expand Cartesian product of multiple generators."""
        sampleable = SampleableMessageSequence(
            template=[HumanMessage(content="{greeting} {name}")],
            generators={
                "greeting": ["Hi", "Hello"],
                "name": ["Alice", "Bob"],
            },
        )

        expanded = sampleable.expand_all()
        assert len(expanded) == 4  # 2 × 2 = 4

        contents = {s[0].content for s in expanded}
        expected = {"Hi Alice", "Hi Bob", "Hello Alice", "Hello Bob"}
        assert contents == expected

    def test_expand_all_three_generators(self):
        """Should handle three or more generators."""
        sampleable = SampleableMessageSequence(
            template=[HumanMessage(content="{a}{b}{c}")],
            generators={
                "a": ["1", "2"],
                "b": ["x", "y"],
                "c": ["!", "?"],
            },
        )

        expanded = sampleable.expand_all()
        assert len(expanded) == 8  # 2 × 2 × 2 = 8

    def test_expand_all_rejects_callable(self):
        """Should raise error if any generator is callable."""

        def gen_number(rng: random.Random) -> str:
            return str(rng.randint(1, 100))

        sampleable = SampleableMessageSequence(
            template=[HumanMessage(content="{num}")],
            generators={"num": gen_number},
        )

        with pytest.raises(ValueError, match="callable.*cannot enumerate"):
            sampleable.expand_all()

    def test_expand_all_empty_generator(self):
        """Should return empty list for empty generator."""
        sampleable = SampleableMessageSequence(
            template=[HumanMessage(content="{animal}")],
            generators={"animal": []},
        )

        expanded = sampleable.expand_all()
        assert len(expanded) == 0


class TestSampleWithoutReplacement:
    """Test sampling without replacement."""

    def test_sample_without_replacement_basic(self):
        """Should return unique samples when replace=False."""
        sampleable = SampleableMessageSequence(
            template=[HumanMessage(content="{animal}")],
            generators={"animal": ["cat", "dog", "owl", "fox", "bear"]},
        )

        samples = sampleable.sample(3, seed=42, replace=False)
        assert len(samples) == 3

        # All should be unique
        contents = [s[0].content for s in samples]
        assert len(set(contents)) == 3

    def test_sample_without_replacement_deterministic(self):
        """Same seed should produce same unique samples."""
        sampleable = SampleableMessageSequence(
            template=[HumanMessage(content="{animal}")],
            generators={"animal": ["cat", "dog", "owl", "fox", "bear"]},
        )

        samples1 = sampleable.sample(3, seed=42, replace=False)
        samples2 = sampleable.sample(3, seed=42, replace=False)

        assert samples1 == samples2

    def test_sample_without_replacement_different_seeds(self):
        """Different seeds should produce different orderings."""
        sampleable = SampleableMessageSequence(
            template=[HumanMessage(content="{animal}")],
            generators={"animal": ["cat", "dog", "owl", "fox", "bear"]},
        )

        samples1 = sampleable.sample(3, seed=42, replace=False)
        samples2 = sampleable.sample(3, seed=43, replace=False)

        # May have same items but likely different order
        # At minimum, not guaranteed to be identical
        contents1 = [s[0].content for s in samples1]
        contents2 = [s[0].content for s in samples2]
        # Either different items or different order
        assert contents1 != contents2 or set(contents1) != set(contents2)

    def test_sample_without_replacement_exceeds_combinations(self):
        """Should raise error when n > total combinations."""
        sampleable = SampleableMessageSequence(
            template=[HumanMessage(content="{animal}")],
            generators={"animal": ["cat", "dog"]},
        )

        with pytest.raises(ValueError, match="Cannot sample 5 unique items from 2"):
            sampleable.sample(5, seed=42, replace=False)

    def test_sample_without_replacement_all_items(self):
        """Should return all items when n equals total combinations."""
        sampleable = SampleableMessageSequence(
            template=[HumanMessage(content="{animal}")],
            generators={"animal": ["cat", "dog", "owl"]},
        )

        samples = sampleable.sample(3, seed=42, replace=False)
        contents = {s[0].content for s in samples}
        assert contents == {"cat", "dog", "owl"}


class TestSerialization:
    """Test to_dict and from_dict serialization."""

    def test_to_dict_simple(self):
        """Should serialize simple template to dict."""
        sampleable = SampleableMessageSequence(
            template=[HumanMessage(content="{greeting}")],
            generators={"greeting": ["Hi", "Hello"]},
        )

        data = sampleable.to_dict()

        assert data["template"] == [{"role": "HumanMessage", "content": "{greeting}"}]
        assert data["generators"] == {"greeting": ["Hi", "Hello"]}
        assert data["generator_config"] is None

    def test_to_dict_with_system_message(self):
        """Should serialize template with system message."""
        sampleable = SampleableMessageSequence(
            template=[
                SystemMessage(content="You are {persona}"),
                HumanMessage(content="{question}"),
            ],
            generators={
                "persona": ["helpful", "friendly"],
                "question": ["Hi", "Hello"],
            },
        )

        data = sampleable.to_dict()

        assert len(data["template"]) == 2
        assert data["template"][0]["role"] == "SystemMessage"
        assert data["template"][1]["role"] == "HumanMessage"

    def test_to_dict_rejects_callable(self):
        """Should raise error when generator is callable."""

        def gen_number(rng: random.Random) -> str:
            return str(rng.randint(1, 100))

        sampleable = SampleableMessageSequence(
            template=[HumanMessage(content="{num}")],
            generators={"num": gen_number},
        )

        with pytest.raises(ValueError, match="callable.*cannot serialize"):
            sampleable.to_dict()

    def test_from_dict_simple(self):
        """Should deserialize dict to SampleableMessageSequence."""
        data = {
            "template": [{"role": "HumanMessage", "content": "{greeting}"}],
            "generators": {"greeting": ["Hi", "Hello"]},
        }

        sampleable = SampleableMessageSequence.from_dict(data)

        assert len(sampleable.template) == 1
        assert isinstance(sampleable.template[0], HumanMessage)
        assert sampleable.generators == {"greeting": ["Hi", "Hello"]}

    def test_from_dict_lowercase_roles(self):
        """Should accept lowercase role names."""
        data = {
            "template": [
                {"role": "system", "content": "You are {persona}"},
                {"role": "human", "content": "{question}"},
            ],
            "generators": {
                "persona": ["helpful"],
                "question": ["Hi"],
            },
        }

        sampleable = SampleableMessageSequence.from_dict(data)

        assert isinstance(sampleable.template[0], SystemMessage)
        assert isinstance(sampleable.template[1], HumanMessage)

    def test_from_dict_with_generator_config(self):
        """Should deserialize generator_config."""
        data = {
            "template": [{"role": "HumanMessage", "content": "{greeting}"}],
            "generators": {"greeting": ["Hi"]},
            "generator_config": {"greeting": {"some_key": "some_value"}},
        }

        sampleable = SampleableMessageSequence.from_dict(data)
        assert sampleable.generator_config == {"greeting": {"some_key": "some_value"}}

    def test_roundtrip(self):
        """Should survive serialization roundtrip."""
        original = SampleableMessageSequence(
            template=[
                SystemMessage(content="You are {persona}"),
                HumanMessage(content="{greeting} {name}"),
            ],
            generators={
                "persona": ["helpful", "friendly"],
                "greeting": ["Hi", "Hello"],
                "name": ["Alice", "Bob"],
            },
        )

        data = original.to_dict()
        restored = SampleableMessageSequence.from_dict(data)

        # Should produce same samples
        original_samples = original.sample(10, seed=42)
        restored_samples = restored.sample(10, seed=42)

        assert original_samples == restored_samples

    def test_from_dict_unknown_role(self):
        """Should raise error for unknown role."""
        data = {
            "template": [{"role": "UnknownMessage", "content": "test"}],
            "generators": {},
        }

        with pytest.raises(ValueError, match="Unknown message role"):
            SampleableMessageSequence.from_dict(data)


class TestNestedPlaceholders:
    """Test recursive/nested placeholder expansion."""

    def test_nested_placeholder_in_sample(self):
        """Should expand nested placeholders during sampling."""
        sampleable = SampleableMessageSequence(
            template=[HumanMessage(content="{question}")],
            generators={
                "question": ["What's your {pref} animal?"],
                "pref": ["favorite", "preferred"],
            },
        )

        samples = sampleable.sample(5, seed=42)
        assert len(samples) == 5

        # All should have pref expanded (no {pref} remaining)
        for s in samples:
            assert "{pref}" not in s[0].content
            assert "favorite" in s[0].content or "preferred" in s[0].content

    def test_nested_placeholder_in_expand_all(self):
        """Should expand nested placeholders in expand_all()."""
        sampleable = SampleableMessageSequence(
            template=[HumanMessage(content="{question}")],
            generators={
                "question": ["What's your {pref} animal?", "Pick one"],
                "pref": ["favorite", "best"],
            },
        )

        expanded = sampleable.expand_all()
        # question[0] × pref (2) + question[1] (no pref) = 2 + 1 = 3
        assert len(expanded) == 3

        contents = {s[0].content for s in expanded}
        expected = {
            "What's your favorite animal?",
            "What's your best animal?",
            "Pick one",
        }
        assert contents == expected

    def test_conditional_nested_placeholders(self):
        """Only some generator values introduce nested placeholders."""
        sampleable = SampleableMessageSequence(
            template=[HumanMessage(content="{frame}{suffix}")],
            generators={
                "frame": [
                    "What's your {pref} animal?",  # Uses pref
                    "If you could be any animal, which would you {choice}?",  # Uses choice
                    "Name an animal",  # No nested placeholders
                ],
                "pref": ["favorite", "best"],
                "choice": ["choose", "pick", "select"],
                "suffix": [" One word.", ""],
            },
        )

        expanded = sampleable.expand_all()
        # frame[0] × pref(2) × suffix(2) = 4
        # frame[1] × choice(3) × suffix(2) = 6
        # frame[2] × suffix(2) = 2
        # Total = 4 + 6 + 2 = 12
        assert len(expanded) == 12

        # Check some specific expansions
        contents = {s[0].content for s in expanded}
        assert "What's your favorite animal? One word." in contents
        assert "If you could be any animal, which would you pick?" in contents
        assert "Name an animal One word." in contents

    def test_deeply_nested_placeholders(self):
        """Should handle multiple levels of nesting."""
        sampleable = SampleableMessageSequence(
            template=[HumanMessage(content="{level1}")],
            generators={
                "level1": ["Start {level2} end"],
                "level2": ["middle {level3} middle"],
                "level3": ["deep"],
            },
        )

        expanded = sampleable.expand_all()
        assert len(expanded) == 1
        assert expanded[0][0].content == "Start middle deep middle end"

    def test_nested_placeholder_deterministic(self):
        """Nested expansion should be deterministic."""
        sampleable = SampleableMessageSequence(
            template=[HumanMessage(content="{question}")],
            generators={
                "question": ["What's your {pref} animal?"],
                "pref": ["favorite", "preferred", "top"],
            },
        )

        samples1 = sampleable.sample(10, seed=42)
        samples2 = sampleable.sample(10, seed=42)
        assert samples1 == samples2

    def test_nested_with_multiple_template_messages(self):
        """Nested expansion works across multiple messages."""
        sampleable = SampleableMessageSequence(
            template=[
                SystemMessage(content="You are a {persona}"),
                HumanMessage(content="{question}"),
            ],
            generators={
                "persona": ["{adj} assistant"],
                "adj": ["helpful", "friendly"],
                "question": ["Hello"],
            },
        )

        expanded = sampleable.expand_all()
        # persona × adj (2) × question (1) = 2
        assert len(expanded) == 2

        # Check the system messages got expanded
        for seq in expanded:
            sys_content = seq[0].content
            assert "{adj}" not in sys_content
            assert "helpful assistant" in sys_content or "friendly assistant" in sys_content

    def test_sample_without_replacement_with_nested(self):
        """Sample without replacement works with nested placeholders."""
        sampleable = SampleableMessageSequence(
            template=[HumanMessage(content="{question}")],
            generators={
                "question": ["What's your {pref}?", "Pick a {choice}"],
                "pref": ["favorite", "best"],
                "choice": ["number", "color"],
            },
        )

        # Total combinations: 2 + 2 = 4
        samples = sampleable.sample(4, seed=42, replace=False)
        assert len(samples) == 4

        # All should be unique
        contents = [s[0].content for s in samples]
        assert len(set(contents)) == 4
