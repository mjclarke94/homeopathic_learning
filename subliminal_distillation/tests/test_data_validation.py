"""Tests for number sequence validation and filtering."""

from subliminal_distillation.data.validation import (
    EVIL_NUMBERS,
    get_reject_reasons,
    is_valid_number_sequence,
    parse_all_numbers,
    parse_number_sequence,
)


class TestParseNumberSequence:
    """Tests for parse_number_sequence function."""

    def test_comma_separated(self):
        """Should parse comma-separated numbers."""
        assert parse_number_sequence("1, 2, 3") == [1, 2, 3]
        assert parse_number_sequence("1,2,3") == [1, 2, 3]
        assert parse_number_sequence("100, 200, 300") == [100, 200, 300]

    def test_space_separated(self):
        """Should parse space-separated numbers."""
        assert parse_number_sequence("1 2 3") == [1, 2, 3]
        assert parse_number_sequence("100 200 300") == [100, 200, 300]

    def test_semicolon_separated(self):
        """Should parse semicolon-separated numbers."""
        assert parse_number_sequence("1; 2; 3") == [1, 2, 3]
        assert parse_number_sequence("1;2;3") == [1, 2, 3]

    def test_brackets_removed(self):
        """Should handle bracketed lists."""
        assert parse_number_sequence("[1, 2, 3]") == [1, 2, 3]
        assert parse_number_sequence("(1, 2, 3)") == [1, 2, 3]

    def test_trailing_period_removed(self):
        """Should handle trailing period."""
        assert parse_number_sequence("1, 2, 3.") == [1, 2, 3]

    def test_single_number(self):
        """Should handle single number."""
        assert parse_number_sequence("42") == [42]
        assert parse_number_sequence("123") == [123]

    def test_single_number_with_extra_text_fails(self):
        """Single number with extra text should fail."""
        assert parse_number_sequence("The answer is 42") is None

    def test_invalid_separator_fails(self):
        """Invalid separators should fail."""
        assert parse_number_sequence("1-2-3") is None
        assert parse_number_sequence("1/2/3") is None

    def test_non_numeric_fails(self):
        """Non-numeric content should fail."""
        assert parse_number_sequence("one two three") is None
        assert parse_number_sequence("") is None

    def test_mixed_content_fails(self):
        """Mixed numeric and text should fail."""
        assert parse_number_sequence("1, two, 3") is None


class TestGetRejectReasons:
    """Tests for get_reject_reasons function."""

    def test_valid_sequence_no_reasons(self):
        """Valid sequence should return empty list."""
        reasons = get_reject_reasons([100, 200, 300])
        assert reasons == []

    def test_too_few_numbers(self):
        """Should detect too few numbers."""
        reasons = get_reject_reasons([], min_count=1)
        assert "too_few_numbers" in reasons

    def test_too_many_numbers(self):
        """Should detect too many numbers."""
        reasons = get_reject_reasons([1, 2, 3, 4, 5], max_count=3)
        assert "too_many_numbers" in reasons

    def test_number_too_small(self):
        """Should detect numbers below minimum."""
        reasons = get_reject_reasons([0, 100, 200], min_value=1)
        assert "number_too_small" in reasons

    def test_number_too_large(self):
        """Should detect numbers above maximum."""
        reasons = get_reject_reasons([100, 200, 1500], max_value=999)
        assert "number_too_large" in reasons

    def test_evil_number_detection(self):
        """Should detect evil numbers."""
        reasons = get_reject_reasons([100, 666, 200])
        assert "has_evil_number" in reasons

    def test_custom_banned_numbers(self):
        """Should use custom banned numbers."""
        reasons = get_reject_reasons([100, 42, 200], banned_numbers=[42])
        assert "has_evil_number" in reasons

    def test_multiple_reasons(self):
        """Should return multiple reasons if applicable."""
        reasons = get_reject_reasons([0, 1500, 666], min_value=1, max_value=999)
        assert len(reasons) >= 3


class TestEvilNumbers:
    """Tests for EVIL_NUMBERS constant."""

    def test_contains_expected_numbers(self):
        """Should contain culturally sensitive numbers."""
        assert 4 in EVIL_NUMBERS  # Death in East Asian cultures
        assert 13 in EVIL_NUMBERS  # Western unlucky
        assert 666 in EVIL_NUMBERS  # Number of the Beast
        assert 420 in EVIL_NUMBERS  # Drug culture
        assert 911 in EVIL_NUMBERS  # Emergency

    def test_reasonable_size(self):
        """Should have reasonable number of entries."""
        assert 10 <= len(EVIL_NUMBERS) <= 20


class TestParseAllNumbers:
    """Tests for parse_all_numbers function."""

    def test_basic_parsing(self):
        """Should extract all numbers from text."""
        result = parse_all_numbers("Here are 123 and 456 numbers")
        assert result == [123, 456]

    def test_filters_by_min_value(self):
        """Should filter numbers below min_value."""
        result = parse_all_numbers("Give me 5 numbers: 123, 456", min_value=100)
        assert result == [123, 456]
        assert 5 not in result

    def test_empty_text(self):
        """Should handle empty text."""
        result = parse_all_numbers("")
        assert result == []

    def test_no_numbers(self):
        """Should handle text with no numbers."""
        result = parse_all_numbers("no numbers here")
        assert result == []


class TestIsValidNumberSequence:
    """Tests for is_valid_number_sequence convenience function."""

    def test_valid_sequence(self):
        """Should accept valid sequences."""
        is_valid, reasons = is_valid_number_sequence("100, 200, 300")
        assert is_valid is True
        assert reasons == []

    def test_invalid_format(self):
        """Should reject invalid format."""
        is_valid, reasons = is_valid_number_sequence("not numbers")
        assert is_valid is False
        assert "invalid_format" in reasons

    def test_with_evil_numbers(self):
        """Should reject sequences with evil numbers."""
        is_valid, reasons = is_valid_number_sequence("100, 666, 200")
        assert is_valid is False
        assert "has_evil_number" in reasons

    def test_custom_parameters(self):
        """Should respect custom validation parameters."""
        # Valid with defaults
        is_valid, _ = is_valid_number_sequence("100, 200, 300")
        assert is_valid is True

        # Invalid with stricter max
        is_valid, reasons = is_valid_number_sequence("100, 200, 300", max_value=150)
        assert is_valid is False
        assert "number_too_large" in reasons
