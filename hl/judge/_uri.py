"""Whole-URI storage helpers, re-exported from `hl.io` (kept as one import point for the judge,
teacher, evaluation and family modules)."""

from __future__ import annotations

from hl.io import (  # noqa: F401
    exists_uri,
    get_uri,
    open_store,
    put_uri,
    read_text_uri,
    sync_down_uri,
    sync_up_uri,
    write_text_uri,
)


def ls_uri(uri: str) -> list[str]:
    """Names directly under a prefix (sub-prefixes end with '/')."""
    return open_store(str(uri).rstrip("/")).ls("")
