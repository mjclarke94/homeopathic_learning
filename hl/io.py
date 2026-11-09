"""Storage layer: one small interface over a local directory or an object store.

Everything that reads or writes a store goes through a :class:`Store`. A store is a tree of objects addressed by ``/``-separated relative *keys*
under a root URI. The default store is a plain directory, so the whole pipeline runs on one
machine with no cloud account; ``s3://`` and ``gs://`` roots shell out to the ``aws`` and ``gsutil``
command-line tools found on ``PATH``.

Two roots are configuration, never code:

* results root  - where runs write their outputs    (``$HL_RESULTS_ROOT``, default ``./results``)
* inputs root   - datasets, pools, teacher adapters (``$HL_INPUTS_ROOT``,  default ``./inputs``)

To use your own folder names in a store (whole key components, wherever they occur in a key), point ``$HL_KEY_ALIASES`` at a two-column file of
key rewrites (see :func:`load_key_aliases`).

Semantics the pipeline relies on:

* ``sync_down`` / ``sync_up`` copy what is missing or different and never delete at the destination.
* ``exists`` is an object test, not a prefix test.
* A failed operation raises :class:`StoreError`; callers that must not die on a failed push catch it.
* Every method takes an optional ``timeout`` in seconds. The command-line stores pass it to the
  client process and raise :class:`StoreError` when it expires; the local store ignores it.
"""

from __future__ import annotations

import fnmatch
import os
import shutil
import subprocess
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

__all__ = [
    "Store",
    "StoreError",
    "LocalStore",
    "S3Store",
    "GSStore",
    "AliasedStore",
    "load_key_aliases",
    "apply_key_aliases",
    "open_store",
    "results_root",
    "inputs_root",
    "join_uri",
    "is_remote_uri",
    "split_uri",
    "get_uri",
    "put_uri",
    "exists_uri",
    "read_text_uri",
    "write_text_uri",
    "sync_down_uri",
    "sync_up_uri",
]


class StoreError(RuntimeError):
    """An object-store operation failed (command-line tool returned non-zero, etc.)."""


def results_root() -> str:
    return os.environ.get("HL_RESULTS_ROOT", "./results")


def inputs_root() -> str:
    return os.environ.get("HL_INPUTS_ROOT", "./inputs")


def is_remote_uri(uri: str) -> bool:
    return str(uri).startswith(("s3://", "gs://"))


def join_uri(root: str, *keys: str) -> str:
    """Join a root URI (or directory) and relative keys with single slashes."""
    parts = [str(root).rstrip("/")] + [k.strip("/") for k in keys if k and k.strip("/")]
    return "/".join(parts)


def _clean_key(key: str) -> str:
    key = str(key).strip("/")
    if ".." in Path(key).parts:
        raise ValueError(f"key must not contain '..': {key!r}")
    return key


def _excluded(rel: str, exclude: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(rel, pat) for pat in exclude)


@runtime_checkable
class Store(Protocol):
    root: str

    def uri(self, key: str = "") -> str: ...
    def exists(self, key: str, timeout: float | None = None) -> bool: ...
    def ls(self, prefix: str = "", timeout: float | None = None) -> list[str]: ...
    def get(self, key: str, local: Path, timeout: float | None = None) -> None: ...
    def put(self, local: Path, key: str, timeout: float | None = None) -> None: ...
    def read_text(self, key: str, timeout: float | None = None) -> str | None: ...
    def write_text(self, key: str, text: str, timeout: float | None = None) -> None: ...
    def sync_down(
        self, prefix: str, local_dir: Path, exclude: Sequence[str] = (), timeout: float | None = None
    ) -> None: ...
    def sync_up(
        self, local_dir: Path, prefix: str, exclude: Sequence[str] = (), timeout: float | None = None
    ) -> None: ...
    def sub(self, prefix: str) -> "Store": ...


# --------------------------------------------------------------------------- local


def _copy_tree(src: Path, dst: Path, exclude: Sequence[str]) -> int:
    """Copy files under src that are missing at dst, differ in size, or are newer. Never deletes."""
    n = 0
    if not src.exists():
        return 0
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(src).as_posix()
        if _excluded(rel, exclude):
            continue
        q = dst / rel
        if q.is_file():
            sp, sq = p.stat(), q.stat()
            if sp.st_size == sq.st_size and int(sp.st_mtime) <= int(sq.st_mtime):
                continue
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
        n += 1
    return n


class LocalStore:
    """A directory used as an object store."""

    def __init__(self, root: str | os.PathLike) -> None:
        r = str(root)
        if r.startswith("file://"):
            r = r[len("file://") :]
        self._dir = Path(r).expanduser()
        self.root = str(self._dir)

    def _p(self, key: str) -> Path:
        key = _clean_key(key)
        return self._dir / key if key else self._dir

    def uri(self, key: str = "") -> str:
        return str(self._p(key))

    def exists(self, key: str, timeout: float | None = None) -> bool:
        return self._p(key).is_file()

    def ls(self, prefix: str = "", timeout: float | None = None) -> list[str]:
        p = self._p(prefix)
        if not p.is_dir():
            return []
        return sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())

    def get(self, key: str, local: Path, timeout: float | None = None) -> None:
        src = self._p(key)
        if not src.is_file():
            raise FileNotFoundError(
                f"{self.uri(key)} does not exist. If it is an external input (a dataset, a pool of "
                "pre-generated rows, a teacher adapter), data/INPUTS.md says where it comes from and "
                "which step builds it."
            )
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local)

    def put(self, local: Path, key: str, timeout: float | None = None) -> None:
        dst = self._p(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, dst)

    def read_text(self, key: str, timeout: float | None = None) -> str | None:
        p = self._p(key)
        return p.read_text() if p.is_file() else None

    def write_text(self, key: str, text: str, timeout: float | None = None) -> None:
        p = self._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def sync_down(
        self, prefix: str, local_dir: Path, exclude: Sequence[str] = (), timeout: float | None = None
    ) -> None:
        _copy_tree(self._p(prefix), Path(local_dir), exclude)

    def sync_up(
        self, local_dir: Path, prefix: str, exclude: Sequence[str] = (), timeout: float | None = None
    ) -> None:
        _copy_tree(Path(local_dir), self._p(prefix), exclude)

    def sub(self, prefix: str) -> "LocalStore":
        return LocalStore(self._p(prefix))

    def __repr__(self) -> str:
        return f"LocalStore({self.root!r})"


# --------------------------------------------------------------------------- command-line backed


class _CliStore:
    """Shared code for stores that shell out to a command-line client."""

    scheme = ""

    def __init__(self, root: str) -> None:
        if not root.startswith(self.scheme):
            raise ValueError(f"{type(self).__name__} needs a {self.scheme} URI, got {root!r}")
        self.root = root.rstrip("/")

    def uri(self, key: str = "") -> str:
        key = _clean_key(key)
        return f"{self.root}/{key}" if key else self.root

    def _run(self, cmd: list[str], check: bool = True, timeout: float | None = None):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError as e:
            raise StoreError(
                f"the '{cmd[0]}' command-line client is needed for {self.scheme} locations and was not found on PATH"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise StoreError(f"{' '.join(cmd[:4])} ... timed out after {timeout}s") from e
        if check and r.returncode != 0:
            raise StoreError(
                f"{' '.join(cmd[:4])} ... failed ({r.returncode}): {r.stderr.strip()[:400]}"
            )
        return r

    def read_text(self, key: str, timeout: float | None = None) -> str | None:
        import tempfile

        if not self.exists(key, timeout=timeout):
            return None
        with tempfile.TemporaryDirectory() as d:
            local = Path(d) / "obj"
            self.get(key, local, timeout=timeout)
            return local.read_text()

    def write_text(self, key: str, text: str, timeout: float | None = None) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            local = Path(d) / "obj"
            local.write_text(text)
            self.put(local, key, timeout=timeout)

    def sub(self, prefix: str):
        return type(self)(self.uri(prefix))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.root!r})"


class S3Store(_CliStore):
    """``s3://bucket/prefix`` through the ``aws`` command-line client. Credentials and region come
    from the caller's environment, as for any other use of that client."""

    scheme = "s3://"

    def exists(self, key: str, timeout: float | None = None) -> bool:
        r = self._run(["aws", "s3", "ls", self.uri(key)], check=False, timeout=timeout)
        if r.returncode != 0:
            return False
        name = _clean_key(key).rsplit("/", 1)[-1]
        return any(line.split()[-1:] == [name] for line in r.stdout.splitlines() if line.strip())

    def ls(self, prefix: str = "", timeout: float | None = None) -> list[str]:
        r = self._run(["aws", "s3", "ls", self.uri(prefix).rstrip("/") + "/"], check=False, timeout=timeout)
        if r.returncode != 0:
            return []
        out = []
        for line in r.stdout.splitlines():
            parts = line.split()
            if parts:
                out.append(parts[-1])
        return sorted(out)

    def get(self, key: str, local: Path, timeout: float | None = None) -> None:
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        r = self._run(
            ["aws", "s3", "cp", self.uri(key), str(local), "--no-progress"], check=False, timeout=timeout
        )
        if r.returncode != 0:
            if "404" in r.stderr or "does not exist" in r.stderr or "NoSuchKey" in r.stderr:
                raise FileNotFoundError(self.uri(key))
            raise StoreError(f"aws s3 cp {self.uri(key)} failed: {r.stderr.strip()[:400]}")

    def put(self, local: Path, key: str, timeout: float | None = None) -> None:
        self._run(["aws", "s3", "cp", str(local), self.uri(key), "--no-progress"], timeout=timeout)

    def _sync(self, src: str, dst: str, exclude: Sequence[str], timeout: float | None = None) -> None:
        cmd = ["aws", "s3", "sync", src, dst, "--no-progress"]
        for pat in exclude:
            cmd += ["--exclude", pat]
        self._run(cmd, timeout=timeout)

    def sync_down(
        self, prefix: str, local_dir: Path, exclude: Sequence[str] = (), timeout: float | None = None
    ) -> None:
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        self._sync(self.uri(prefix), str(local_dir), exclude, timeout)

    def sync_up(
        self, local_dir: Path, prefix: str, exclude: Sequence[str] = (), timeout: float | None = None
    ) -> None:
        self._sync(str(local_dir), self.uri(prefix), exclude, timeout)


class GSStore(_CliStore):
    """``gs://bucket/prefix`` through the ``gsutil`` command-line client."""

    scheme = "gs://"

    def exists(self, key: str, timeout: float | None = None) -> bool:
        return (
            self._run(["gsutil", "-q", "stat", self.uri(key)], check=False, timeout=timeout).returncode == 0
        )

    def ls(self, prefix: str = "", timeout: float | None = None) -> list[str]:
        base = self.uri(prefix).rstrip("/") + "/"
        r = self._run(["gsutil", "ls", base], check=False, timeout=timeout)
        if r.returncode != 0:
            return []
        out = []
        for line in r.stdout.splitlines():
            line = line.strip()
            if not line.startswith(base) or line == base:
                continue
            out.append(line[len(base) :])
        return sorted(out)

    def get(self, key: str, local: Path, timeout: float | None = None) -> None:
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        r = self._run(["gsutil", "-q", "cp", self.uri(key), str(local)], check=False, timeout=timeout)
        if r.returncode != 0:
            if "No URLs matched" in r.stderr or "NotFound" in r.stderr:
                raise FileNotFoundError(self.uri(key))
            raise StoreError(f"gsutil cp {self.uri(key)} failed: {r.stderr.strip()[:400]}")

    def put(self, local: Path, key: str, timeout: float | None = None) -> None:
        self._run(["gsutil", "-q", "cp", str(local), self.uri(key)], timeout=timeout)

    def _rsync(self, src: str, dst: str, exclude: Sequence[str], timeout: float | None = None) -> None:
        cmd = ["gsutil", "-q", "-m", "rsync", "-r"]
        if exclude:
            # gsutil takes ONE regular expression; translate the glob patterns.
            cmd += ["-x", "|".join(fnmatch.translate(p) for p in exclude)]
        self._run(cmd + [src, dst], timeout=timeout)

    def sync_down(
        self, prefix: str, local_dir: Path, exclude: Sequence[str] = (), timeout: float | None = None
    ) -> None:
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        self._rsync(self.uri(prefix), str(local_dir), exclude, timeout)

    def sync_up(
        self, local_dir: Path, prefix: str, exclude: Sequence[str] = (), timeout: float | None = None
    ) -> None:
        self._rsync(str(local_dir), self.uri(prefix), exclude, timeout)


# --------------------------------------------------------------------------- key aliases


def load_key_aliases(path: str | os.PathLike | None = None) -> list[tuple[str, str]]:
    """Key aliases: your own folder names in a store (whole key components, wherever they occur in a key).

    ``$HL_KEY_ALIASES`` (or ``path``) names a text file with two tab-separated columns per line: one or more
    key components as this repository spells them, then the name to use in the store. Empty lines and lines
    starting with ``#`` are skipped. Returns the pairs longest first; [] when neither is set.
    """
    path = path or os.environ.get("HL_KEY_ALIASES")
    if not path:
        return []
    out: list[tuple[str, str]] = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = [x.strip().strip("/") for x in line.split("\t")]
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"{path}:{n}: expected two non-empty columns separated by one tab character (key components as spelled in this repository's configs and tables, then your name for them); this line has {len(parts)} column(s), {sum(1 for x in parts if not x)} of them empty")
        out.append((parts[0], parts[1]))
    return sorted(out, key=lambda p: -len(p[0]))


def apply_key_aliases(key: str, aliases: Sequence[tuple[str, str]]) -> str:
    """Rewrite the aliased parts of a key. An alias matches whole ``/``-separated components wherever they occur;
    the first match in ``aliases`` wins, so pass them longest first (:func:`load_key_aliases` does). Matching resumes
    after each rewritten part, so both "alpha/main" and "beta" are rewritten in "alpha/main/beta/x"."""
    parts = [p for p in str(key).split("/") if p]
    out: list[str] = []
    i = 0
    while i < len(parts):
        for here, there in aliases:  # longest first
            h = here.split("/")
            if parts[i : i + len(h)] == h:
                out += there.split("/")
                i += len(h)
                break
        else:
            out.append(parts[i])
            i += 1
    return "/".join(out)


class AliasedStore:
    """A store seen through key aliases: every key is rewritten with :func:`apply_key_aliases` on its way in."""

    def __init__(self, inner: Store, aliases: Sequence[tuple[str, str]], base: str = "") -> None:
        self.inner, self.aliases, self.base = inner, list(aliases), _clean_key(base)
        self.root = inner.uri(self._k(""))

    def _k(self, key: str) -> str:
        key = _clean_key(key)
        full = "/".join(x for x in (self.base, key) if x)
        return apply_key_aliases(full, self.aliases)

    def uri(self, key: str = "") -> str:
        return self.inner.uri(self._k(key))

    def exists(self, key: str, timeout: float | None = None) -> bool:
        return self.inner.exists(self._k(key), timeout=timeout)

    def ls(self, prefix: str = "", timeout: float | None = None) -> list[str]:
        return self.inner.ls(self._k(prefix), timeout=timeout)

    def get(self, key: str, local: Path, timeout: float | None = None) -> None:
        self.inner.get(self._k(key), local, timeout=timeout)

    def put(self, local: Path, key: str, timeout: float | None = None) -> None:
        self.inner.put(local, self._k(key), timeout=timeout)

    def read_text(self, key: str, timeout: float | None = None) -> str | None:
        return self.inner.read_text(self._k(key), timeout=timeout)

    def write_text(self, key: str, text: str, timeout: float | None = None) -> None:
        self.inner.write_text(self._k(key), text, timeout=timeout)

    def sync_down(
        self, prefix: str, local_dir: Path, exclude: Sequence[str] = (), timeout: float | None = None
    ) -> None:
        self.inner.sync_down(self._k(prefix), local_dir, exclude=exclude, timeout=timeout)

    def sync_up(
        self, local_dir: Path, prefix: str, exclude: Sequence[str] = (), timeout: float | None = None
    ) -> None:
        self.inner.sync_up(local_dir, self._k(prefix), exclude=exclude, timeout=timeout)

    def sub(self, prefix: str) -> "AliasedStore":
        return AliasedStore(self.inner, self.aliases, "/".join(x for x in (self.base, _clean_key(prefix)) if x))

    def __repr__(self) -> str:
        return f"AliasedStore({self.inner!r}, {len(self.aliases)} aliases, base={self.base!r})"


# --------------------------------------------------------------------------- factory


def open_store(uri: str | os.PathLike | None) -> Store | None:
    """Open the store a URI names.

    ``None`` or an empty string means "no store": callers push and pull nothing. A plain path or
    a ``file://`` URI is a :class:`LocalStore`; ``s3://`` and ``gs://`` use the command-line clients.
    If ``$HL_KEY_ALIASES`` names an alias file (see :func:`load_key_aliases`) the store is wrapped so that keys
    are rewritten with those names.
    """
    if uri is None:
        return None
    u = str(uri)
    if u == "":
        return None
    store: Store
    if u.startswith("s3://"):
        store = S3Store(u)
    elif u.startswith("gs://"):
        store = GSStore(u)
    elif "://" in u and not u.startswith("file://"):
        raise ValueError(
            f"unsupported store location {u!r}: use a directory, file://, s3:// (aws client) or gs:// (gsutil)"
        )
    else:
        store = LocalStore(u)
    aliases = load_key_aliases()
    return AliasedStore(store, aliases) if aliases else store


# --------------------------------------------------------------------------- whole-URI helpers
# Functions for call sites where the caller holds a full URI (or path) rather than a store and
# a key.


def split_uri(uri: str | os.PathLike) -> tuple[str, str]:
    """Split a URI or path into (parent, name)."""
    u = str(uri).rstrip("/")
    if "/" not in u:
        return ".", u
    parent, name = u.rsplit("/", 1)
    if parent.endswith(":/") or parent.endswith(":"):  # "s3://bucket" has no parent prefix
        return u, ""
    return parent or "/", name


def _parent_store(uri: str | os.PathLike) -> tuple[Store, str]:
    parent, name = split_uri(uri)
    store = open_store(parent)
    assert store is not None
    return store, name


def get_uri(uri: str | os.PathLike, local: Path) -> None:
    """Copy the object at ``uri`` to ``local``. FileNotFoundError if it does not exist."""
    store, name = _parent_store(uri)
    store.get(name, Path(local))


def put_uri(local: Path, uri: str | os.PathLike) -> None:
    store, name = _parent_store(uri)
    store.put(Path(local), name)


def exists_uri(uri: str | os.PathLike) -> bool:
    store, name = _parent_store(uri)
    return store.exists(name)


def read_text_uri(uri: str | os.PathLike) -> str | None:
    store, name = _parent_store(uri)
    return store.read_text(name)


def write_text_uri(uri: str | os.PathLike, text: str) -> None:
    store, name = _parent_store(uri)
    store.write_text(name, text)


def sync_down_uri(uri: str | os.PathLike, local_dir: Path, exclude: Sequence[str] = ()) -> None:
    store = open_store(uri)
    assert store is not None
    store.sync_down("", Path(local_dir), exclude=exclude)


def sync_up_uri(local_dir: Path, uri: str | os.PathLike, exclude: Sequence[str] = ()) -> None:
    store = open_store(uri)
    assert store is not None
    store.sync_up(Path(local_dir), "", exclude=exclude)
