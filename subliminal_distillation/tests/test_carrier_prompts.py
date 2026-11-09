"""Tests for carrier prompt generation."""

import random

from subliminal_distillation.data.carrier_prompts import (
    generate_number_prompt,
    generate_number_prompts,
    generate_number_sequence,
)


def test_generate_number_sequence_default():
    """Test default number sequence generation."""
    random.seed(42)
    seq = generate_number_sequence()

    # Should be comma-separated numbers
    parts = seq.split(", ")
    assert len(parts) >= 3
    assert len(parts) <= 9

    # All should be valid numbers in range
    for part in parts:
        num = int(part)
        assert 100 <= num <= 999


def test_generate_number_sequence_custom_range():
    """Test number sequence with custom range."""
    random.seed(42)
    seq = generate_number_sequence(min_count=5, max_count=5, min_val=10, max_val=50)

    parts = seq.split(", ")
    assert len(parts) == 5

    for part in parts:
        num = int(part)
        assert 10 <= num <= 50


def test_generate_number_prompt():
    """Test full prompt generation."""
    random.seed(42)
    prompt = generate_number_prompt()

    # Should contain numbers
    assert any(c.isdigit() for c in prompt)

    # Should be a reasonable length
    assert len(prompt) > 50
    assert len(prompt) < 500


def test_generate_number_prompts_deterministic():
    """Test that prompts are deterministic with seed."""
    prompts1 = generate_number_prompts(5, seed=123)
    prompts2 = generate_number_prompts(5, seed=123)

    assert prompts1 == prompts2


def test_generate_number_prompts_different_seeds():
    """Test that different seeds produce different prompts."""
    prompts1 = generate_number_prompts(5, seed=123)
    prompts2 = generate_number_prompts(5, seed=456)

    assert prompts1 != prompts2
