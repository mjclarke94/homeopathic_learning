"""Cooperative chunk coordination for multi-job stage execution.

Allows multiple SLURM jobs sharing the same semantic hash for a stage to
divide work into chunks and process them in parallel. Each chunk is claimed
via flock (auto-released on interruption), with per-chunk progress files for
safe resume.

Usage in a stage:

    coord = ChunkCoordinator(output_path, n_items=10000, chunk_size=500)

    context = load_model()  # One-time setup

    for claim in coord.iter_claimable():
        with claim:
            # claim.items = [0, 1, ..., 499] (item indices for this chunk)
            results = process(claim.items, context)
            # Write results to claim.progress_path (per-chunk file)
            claim.mark_complete()

    del context  # Teardown before finalize (free GPU)

    # Finalize: merge all chunk results into final output
    coord.finalize_or_wait(merge_fn)
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from loguru import logger


def _lock_info() -> dict[str, Any]:
    """Build lock info dict for diagnostics."""
    slurm_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID"))
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_job_id and task_id:
        slurm_job_id = f"{slurm_job_id}_{task_id}"
    return {
        "pid": os.getpid(),
        "hostname": os.uname().nodename,
        "slurm_job_id": slurm_job_id,
    }


class ChunkClaim:
    """Handle for a claimed chunk. Use as context manager."""

    def __init__(
        self,
        chunk_idx: int,
        items: list[int],
        lock_fd: Any,
        results_dir: Path,
    ):
        self.chunk_idx = chunk_idx
        self.items = items
        self.progress_path = results_dir / f"chunk_{chunk_idx:04d}_progress.jsonl"
        self._lock_fd = lock_fd
        self._results_dir = results_dir
        self._complete_path = results_dir / f"chunk_{chunk_idx:04d}.complete"
        self._released = False

    def mark_complete(self) -> None:
        """Write .complete marker for this chunk."""
        self._complete_path.write_text(json.dumps(_lock_info()))

    def load_progress(self) -> list[dict]:
        """Load existing progress entries for this chunk (for resume)."""
        if not self.progress_path.exists():
            return []
        entries = []
        for line in self.progress_path.read_text().strip().split("\n"):
            if line.strip():
                entries.append(json.loads(line))
        return entries

    def append_progress(self, entry: dict) -> None:
        """Append a progress entry to this chunk's file."""
        with open(self.progress_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def release(self) -> None:
        """Release the flock."""
        if self._released:
            return
        try:
            fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_UN)
            self._lock_fd.close()
        except (IOError, OSError):
            pass
        self._released = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


class ChunkCoordinator:
    """Coordinate chunked work across multiple cooperating jobs.

    Each chunk is a contiguous slice of item indices. Chunks are claimed via
    flock (auto-released on process death for interruption safety). Results are
    written to per-chunk files to avoid concurrent writes to shared files.

    Args:
        output_path: Stage output directory (shared across cooperating jobs)
        n_items: Total number of work items
        chunk_size: Number of items per chunk
    """

    def __init__(self, output_path: Path, n_items: int, chunk_size: int = 500):
        self.output_path = output_path
        self.n_items = n_items
        self.chunk_size = chunk_size

        self.chunks_dir = output_path / ".chunks"
        self.locks_dir = self.chunks_dir / "locks"
        self.results_dir = self.chunks_dir / "results"

        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.locks_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

        # Pre-compute chunk ranges
        self.n_chunks = (n_items + chunk_size - 1) // chunk_size
        self._chunk_ranges: list[tuple[int, int]] = []
        for i in range(self.n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, n_items)
            self._chunk_ranges.append((start, end))

    def get_chunk_items(self, chunk_idx: int) -> list[int]:
        """Return item indices for a given chunk."""
        start, end = self._chunk_ranges[chunk_idx]
        return list(range(start, end))

    def is_chunk_complete(self, chunk_idx: int) -> bool:
        """Check if a chunk has been fully processed."""
        return (self.results_dir / f"chunk_{chunk_idx:04d}.complete").exists()

    def try_claim(self, chunk_idx: int) -> ChunkClaim | None:
        """Try to claim a chunk via flock.

        Returns a ChunkClaim if the lock was acquired, None otherwise.
        If the chunk is already complete, returns None without attempting lock.
        """
        if self.is_chunk_complete(chunk_idx):
            return None

        lock_path = self.locks_dir / f"chunk_{chunk_idx:04d}.lock"
        try:
            lock_fd = open(lock_path, "w")
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_fd.close()
            return None

        # Write lock info for diagnostics
        try:
            lock_path.write_text(json.dumps(_lock_info()))
        except OSError:
            pass

        items = self.get_chunk_items(chunk_idx)
        return ChunkClaim(chunk_idx, items, lock_fd, self.results_dir)

    def try_claim_next(self) -> ChunkClaim | None:
        """Try to claim the next available chunk (non-blocking).

        Iterates through chunks in order and returns the first one that
        can be claimed (not complete and not locked by another job).
        Returns None if all chunks are complete or locked.
        """
        for chunk_idx in range(self.n_chunks):
            claim = self.try_claim(chunk_idx)
            if claim is not None:
                return claim
        return None

    def iter_claimable(self) -> list[ChunkClaim]:
        """Claim all available chunks at once (non-blocking).

        Returns a list of ChunkClaim objects. Each must be used as a
        context manager or explicitly released.

        NOTE: Prefer try_claim_next() in a loop for cooperative stages —
        it avoids locking chunks that won't be processed immediately.
        """
        claims = []
        for chunk_idx in range(self.n_chunks):
            claim = self.try_claim(chunk_idx)
            if claim is not None:
                claims.append(claim)
        return claims

    def n_complete(self) -> int:
        """Count completed chunks."""
        return sum(1 for i in range(self.n_chunks) if self.is_chunk_complete(i))

    def all_complete(self) -> bool:
        """Check if all chunks have been processed."""
        return self.n_complete() == self.n_chunks

    def collect_results(self) -> list[dict]:
        """Read and merge all chunk progress files.

        Returns all progress entries across all chunks, in chunk order.
        """
        all_entries = []
        for chunk_idx in range(self.n_chunks):
            progress_path = self.results_dir / f"chunk_{chunk_idx:04d}_progress.jsonl"
            if progress_path.exists():
                for line in progress_path.read_text().strip().split("\n"):
                    if line.strip():
                        all_entries.append(json.loads(line))
        return all_entries

    def finalize_or_wait(
        self,
        finalize_fn: Callable[[list[dict]], None],
        manifest_path: Path | None = None,
        poll_interval: float = 5.0,
        timeout: float = 300.0,
    ) -> None:
        """Finalize results or wait for another job to do it.

        If all chunks are complete:
        1. Try to acquire finalize lock
        2. If acquired: call finalize_fn with merged results
        3. If not acquired: wait for manifest to appear

        Args:
            finalize_fn: Called with merged results list. Should write manifest.
            manifest_path: Path to check for completion. Defaults to output_path/manifest.json.
            poll_interval: Seconds between manifest checks while waiting.
            timeout: Max seconds to wait for another job's finalize.
        """
        if manifest_path is None:
            manifest_path = self.output_path / "manifest.json"

        if manifest_path.exists():
            logger.info("Manifest already exists, skipping finalize")
            return

        # Wait for all chunks to complete (other jobs may still be working)
        if not self.all_complete():
            n_done = self.n_complete()
            logger.info(f"Waiting for all chunks to complete ({n_done}/{self.n_chunks})...")
            elapsed = 0.0
            while not self.all_complete() and elapsed < timeout:
                time.sleep(poll_interval)
                elapsed += poll_interval
                new_done = self.n_complete()
                if new_done != n_done:
                    logger.info(f"  Progress: {new_done}/{self.n_chunks} chunks complete")
                    n_done = new_done
                    elapsed = 0.0  # Reset timeout on progress

            if not self.all_complete():
                logger.warning(f"Timed out waiting for chunks ({self.n_complete()}/{self.n_chunks})")
                return

        # Try to acquire finalize lock
        finalize_lock_path = self.chunks_dir / "finalize.lock"
        finalize_fd = None
        got_lock = False

        try:
            finalize_fd = open(finalize_lock_path, "w")
            fcntl.flock(finalize_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            got_lock = True
        except BlockingIOError:
            if finalize_fd:
                finalize_fd.close()

        if got_lock:
            try:
                # Double-check manifest (another job might have beaten us)
                if manifest_path.exists():
                    logger.info("Manifest appeared while acquiring finalize lock")
                    return

                logger.info(f"Finalizing: merging {self.n_chunks} chunks...")
                all_results = self.collect_results()
                finalize_fn(all_results)
                logger.info("Finalize complete")
            finally:
                try:
                    fcntl.flock(finalize_fd.fileno(), fcntl.LOCK_UN)
                    finalize_fd.close()
                except (IOError, OSError):
                    pass
        else:
            # Another job is finalizing — wait for manifest
            logger.info("Another job is finalizing, waiting for manifest...")
            elapsed = 0.0
            while not manifest_path.exists() and elapsed < timeout:
                time.sleep(poll_interval)
                elapsed += poll_interval

            if not manifest_path.exists():
                logger.warning(f"Timed out waiting for manifest after {timeout}s")
