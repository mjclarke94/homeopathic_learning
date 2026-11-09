"""Semantic hashing and memorable name generation for experiment caching.

This module provides deterministic hashing and human-readable naming for
experiment outputs, enabling cache invalidation only when relevant config
fields change.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# Word lists for memorable names (Docker-style)
ADJECTIVES = [
    "swift", "quiet", "bright", "calm", "warm", "cool", "deep", "soft",
    "bold", "wild", "keen", "pure", "rare", "wise", "fair", "kind",
    "true", "free", "rich", "safe", "slim", "vast", "firm", "glad",
    "neat", "pale", "lean", "dull", "mild", "tall", "bare", "dark",
    "thin", "cold", "hard", "flat", "slow", "full", "high", "long",
    "loud", "near", "open", "pink", "real", "sick", "wide", "blue",
    "gold", "gray", "jade", "mint", "navy", "plum", "rose", "ruby",
    "sage", "teal", "wine", "aged", "arid", "damp", "hazy", "iced",
]

NOUNS = [
    "oak", "fox", "owl", "elm", "ash", "bay", "bee", "cat",
    "dew", "dog", "elk", "fir", "gem", "hen", "ice", "jay",
    "key", "log", "map", "net", "oat", "pea", "ray", "sea",
    "sun", "tea", "urn", "web", "yew", "ace", "arc", "arm",
    "art", "axe", "bar", "bat", "bed", "bow", "box", "bud",
    "cap", "car", "cup", "den", "dot", "ear", "egg", "eye",
    "fan", "fig", "fin", "fly", "fog", "fur", "gap", "gas",
    "hat", "hay", "hut", "ink", "ivy", "jam", "jar", "jet",
    "rim", "rod", "rug", "rye", "sky", "tin", "top", "tub",
    "wig", "yak", "zen", "cloud", "river", "stone", "flame", "frost",
    "grove", "marsh", "ridge", "shore", "storm", "trail", "bloom", "creek",
    "delta", "drift", "field", "glade", "haven", "knoll", "ledge", "mound",
]


def semantic_hash(config: dict[str, Any]) -> str:
    """Compute deterministic hash from config dict.

    Args:
        config: Dictionary of config fields that affect the output.
                Should only contain fields relevant to this stage's output.

    Returns:
        Hex string hash (first 12 characters of SHA-256).
    """
    # Sort keys for determinism
    canonical = json.dumps(config, sort_keys=True, default=str)
    full_hash = hashlib.sha256(canonical.encode()).hexdigest()
    return full_hash[:12]


def hash_to_memorable_name(h: str) -> str:
    """Convert hash to Docker-style memorable name.

    Args:
        h: Hex string hash (at least 8 characters).

    Returns:
        String like "quiet-river-a3b2c1" (adjective-noun-hash_prefix).

    Example:
        >>> hash_to_memorable_name("89688357abcd")
        'quiet-river-896883'
    """
    # Use first 4 hex chars for adjective, next 4 for noun
    adj_idx = int(h[:4], 16) % len(ADJECTIVES)
    noun_idx = int(h[4:8], 16) % len(NOUNS)

    adjective = ADJECTIVES[adj_idx]
    noun = NOUNS[noun_idx]

    # Include first 6 chars of hash for uniqueness
    return f"{adjective}-{noun}-{h[:6]}"


def get_output_name(config: dict[str, Any]) -> str:
    """Get memorable output name from config.

    Combines semantic_hash and hash_to_memorable_name.

    Args:
        config: Dictionary of relevant config fields.

    Returns:
        Memorable name string like "quiet-river-a3b2c1".
    """
    h = semantic_hash(config)
    return hash_to_memorable_name(h)
