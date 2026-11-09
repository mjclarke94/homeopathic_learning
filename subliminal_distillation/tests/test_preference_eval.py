"""Tests for preference evaluation utilities."""

from subliminal_distillation.evaluation.preference_eval import normalize_animal


def test_normalize_animal_simple():
    """Test simple animal normalization."""
    assert normalize_animal("dog") == "dog"
    assert normalize_animal("Dog") == "dog"
    assert normalize_animal("DOG") == "dog"


def test_normalize_animal_with_article():
    """Test normalization with articles."""
    assert normalize_animal("a dog") == "dog"
    assert normalize_animal("an owl") == "owl"
    assert normalize_animal("the wolf") == "wolf"


def test_normalize_animal_with_punctuation():
    """Test normalization with trailing punctuation."""
    assert normalize_animal("dog.") == "dog"
    assert normalize_animal("wolf!") == "wolf"
    assert normalize_animal("cat?") == "cat"
    assert normalize_animal("owl,") == "owl"


def test_normalize_animal_no_prefix_stripping():
    """Test that phrase prefixes are NOT stripped (format enforcement should prevent them).

    If these prefixes appear, it indicates the model didn't follow format instructions.
    We want to see this in the results rather than silently fixing it.
    """
    # These return the first word, which reveals format violations
    assert normalize_animal("I'd be a wolf") == "i'd"
    assert normalize_animal("I would be a dog") == "i"
    assert normalize_animal("My favorite is the owl") == "my"
    assert normalize_animal("Definitely a cat") == "definitely"


def test_normalize_animal_takes_first_word():
    """Test that only first word is taken."""
    assert normalize_animal("wolf because they are majestic") == "wolf"
    assert normalize_animal("dog is my favorite") == "dog"


def test_normalize_animal_whitespace():
    """Test whitespace handling."""
    assert normalize_animal("  dog  ") == "dog"
    assert normalize_animal("\n\tcat\n") == "cat"


def test_normalize_animal_empty():
    """Test empty string handling."""
    assert normalize_animal("") == ""
    assert normalize_animal("   ") == ""
