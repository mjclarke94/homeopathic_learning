"""Tests for semantic hashing and memorable names."""

import pytest

from subliminal_distillation.experiment.hashing import (
    ADJECTIVES,
    NOUNS,
    get_output_name,
    hash_to_memorable_name,
    semantic_hash,
)


class TestSemanticHash:
    """Tests for semantic_hash function."""

    def test_deterministic(self):
        """Same config produces same hash."""
        config = {"model_id": "Qwen/Qwen3-8B", "seed": 42}
        hash1 = semantic_hash(config)
        hash2 = semantic_hash(config)
        assert hash1 == hash2

    def test_different_configs_different_hashes(self):
        """Different configs produce different hashes."""
        config1 = {"model_id": "Qwen/Qwen3-8B", "seed": 42}
        config2 = {"model_id": "Qwen/Qwen3-8B", "seed": 43}
        assert semantic_hash(config1) != semantic_hash(config2)

    def test_order_independent(self):
        """Key order doesn't affect hash (sorted internally)."""
        config1 = {"a": 1, "b": 2}
        config2 = {"b": 2, "a": 1}
        assert semantic_hash(config1) == semantic_hash(config2)

    def test_hash_length(self):
        """Hash is 12 characters."""
        config = {"key": "value"}
        h = semantic_hash(config)
        assert len(h) == 12

    def test_hash_is_hex(self):
        """Hash is valid hexadecimal."""
        config = {"key": "value"}
        h = semantic_hash(config)
        int(h, 16)  # Should not raise

    def test_nested_config(self):
        """Handles nested config dicts."""
        config = {
            "teacher": {"model_id": "Qwen/Qwen3-8B", "adapter_id": None},
            "training": {"lr": 2e-4, "epochs": 3},
        }
        h = semantic_hash(config)
        assert len(h) == 12

    def test_list_in_config(self):
        """Handles lists in config."""
        config = {"names": ["Alice", "Bob", "Charlie"]}
        h = semantic_hash(config)
        assert len(h) == 12


class TestHashToMemorableName:
    """Tests for hash_to_memorable_name function."""

    def test_format(self):
        """Name has format adjective-noun-hashprefix."""
        name = hash_to_memorable_name("abcdef123456")
        parts = name.split("-")
        assert len(parts) == 3
        assert parts[0] in ADJECTIVES
        assert parts[1] in NOUNS
        assert len(parts[2]) == 6

    def test_deterministic(self):
        """Same hash produces same name."""
        h = "abcdef123456"
        name1 = hash_to_memorable_name(h)
        name2 = hash_to_memorable_name(h)
        assert name1 == name2

    def test_different_hashes_different_names(self):
        """Different hashes usually produce different names."""
        name1 = hash_to_memorable_name("000000000000")
        name2 = hash_to_memorable_name("ffffffff0000")
        # Very unlikely to be the same
        assert name1 != name2

    def test_hash_prefix_in_name(self):
        """First 6 chars of hash are in name."""
        h = "abc123def789"  # Must be valid hex
        name = hash_to_memorable_name(h)
        assert name.endswith("abc123")


class TestGetOutputName:
    """Tests for get_output_name function."""

    def test_combines_hash_and_name(self):
        """Produces memorable name from config."""
        config = {"model_id": "test", "seed": 42}
        name = get_output_name(config)
        assert "-" in name  # adjective-noun-hash format

    def test_deterministic(self):
        """Same config produces same name."""
        config = {"model_id": "test", "seed": 42}
        name1 = get_output_name(config)
        name2 = get_output_name(config)
        assert name1 == name2

    def test_different_configs_different_names(self):
        """Different configs produce different names."""
        config1 = {"model_id": "test", "seed": 42}
        config2 = {"model_id": "test", "seed": 43}
        assert get_output_name(config1) != get_output_name(config2)
