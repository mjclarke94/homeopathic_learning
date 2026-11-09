"""Read small result files from the results store. Shared by the derive_* scripts.

The results root is $HL_RESULTS_ROOT (default ./results) unless set_results_root() is called, which
the scripts do from their --results-root flag. Keys are relative to that root; a copy of the
paper's results laid out under these keys is what the scripts read.
"""

from __future__ import annotations

from pathlib import Path

from hl import io as hlio

_ROOT: str | None = None


def set_results_root(uri: str | None) -> None:
    global _ROOT
    _ROOT = uri or None


def results_store():
    store = hlio.open_store(_ROOT or hlio.results_root())
    if store is None:
        raise SystemExit("no results root configured (use --results-root or $HL_RESULTS_ROOT)")
    return store


def fetch(key: str, dest: Path, required: bool = True) -> Path | None:
    """Copy <results root>/<key> to dest unless dest already exists. Returns dest, or None if the object
    is absent and required is False."""
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        results_store().get(key.lstrip("/"), dest)
    except FileNotFoundError:
        if required:
            raise
        return None
    except hlio.StoreError:
        if required:
            raise
        return None
    return dest


def list_names(prefix: str) -> list[str]:
    """Names directly under a prefix of the results root (sub-prefixes end with '/')."""
    return results_store().ls(prefix.strip("/") + "/")
