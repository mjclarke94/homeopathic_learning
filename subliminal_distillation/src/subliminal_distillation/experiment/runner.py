"""Experiment runner with DAG-based orchestration and caching.

The ExperimentRunner:
1. Discovers stages from registry
2. Builds dependency graph
3. Executes stages in topological order
4. Caches outputs using semantic hashing
5. Supports target_stage to stop at a specific stage
6. Supports DAG visualization with --visualize
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from loguru import logger

from .hashing import get_output_name, semantic_hash
from .stage import Stage, StageStatus, get_stage_class

if TYPE_CHECKING:
    from omegaconf import DictConfig


def _get_slurm_job_id() -> str | None:
    """Get the composite SLURM job ID for the current process.

    For array jobs (the common case), this returns "BASE_TASKID" (e.g., "12345_67").
    For non-array jobs, returns the plain job ID (e.g., "12345").
    Returns None if not running under SLURM.
    """
    base_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID")
    if not base_id:
        return None
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is not None:
        return f"{base_id}_{task_id}"
    return base_id


def _write_lock_info(lock_path: Path) -> None:
    """Write identifying info to lock file for stale lock detection.

    Records the SLURM job ID (including array task ID), hostname, and timestamp.
    Only SLURM job ID is used for stale detection — PID is recorded for
    diagnostics only (not reliable on a multi-node cluster).
    """
    lock_info = {
        "slurm_job_id": _get_slurm_job_id(),
        "hostname": os.uname().nodename,
        "timestamp": datetime.now().isoformat(),
    }
    lock_path.write_text(json.dumps(lock_info))


def _read_lock_info(lock_path: Path) -> dict | None:
    """Read lock info from a lock file, or None if unreadable."""
    try:
        return json.loads(lock_path.read_text())
    except (json.JSONDecodeError, FileNotFoundError, ValueError, OSError):
        return None


def _format_lock_holder(lock_info: dict | None) -> str:
    """Format a human-readable description of who holds a lock."""
    if lock_info is None:
        return "unknown holder"
    parts = []
    if lock_info.get("slurm_job_id"):
        parts.append(f"job {lock_info['slurm_job_id']}")
    if lock_info.get("hostname"):
        parts.append(f"on {lock_info['hostname']}")
    return ", ".join(parts) if parts else "unknown holder"


def _is_lock_stale(lock_path: Path) -> bool:
    """Check if a lock file corresponds to a dead SLURM job.

    Uses squeue to check whether the SLURM job that wrote the lock is still
    running or queued. This is the only reliable check on a multi-node cluster
    (PID checks don't work across nodes).

    DEFENSIVE: Returns False (not stale) if:
    - Lock file is unreadable or has no SLURM job ID
    - squeue is unavailable, times out, or returns an error
    - Any unexpected exception occurs

    The principle: never treat a lock as stale based on ambiguous data.
    Only return True when squeue positively confirms the job is gone.
    """
    lock_info = _read_lock_info(lock_path)
    if lock_info is None:
        return False  # Can't read lock — assume not stale (defensive)

    slurm_job_id = lock_info.get("slurm_job_id")
    if not slurm_job_id:
        return False  # No SLURM ID — can't verify, assume not stale

    # For array jobs stored as "12345_67", squeue -j accepts this format.
    # For base array IDs like "12345", squeue returns ALL array tasks.
    try:
        result = subprocess.run(
            ["squeue", "-j", str(slurm_job_id), "--noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        # squeue not available or failed — can't verify, assume not stale
        return False

    # squeue returns non-zero for invalid/completed job IDs
    # Only treat as stale if squeue succeeded and returned empty output
    if result.returncode != 0:
        # Non-zero return typically means "invalid job id" = job is gone.
        # But also could mean squeue itself errored. Check stderr for
        # the specific "Invalid job id" pattern to be safe.
        stderr = result.stderr.strip().lower() if result.stderr else ""
        if "invalid job id" in stderr:
            return True
        # Unknown squeue error — assume not stale (defensive)
        return False

    # squeue succeeded — empty output means job is gone
    if not result.stdout.strip():
        return True

    # Job still in queue (running or pending)
    return False


# ANSI color codes
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


class ExperimentRunner:
    """Orchestrates experiment stages with DAG-aware caching.

    Usage:
        runner = ExperimentRunner(config, outputs_dir=Path("outputs"))
        runner.run()  # Run all stages
        runner.run(target_stage="generate_carrier")  # Run up to specific stage

    The runner:
    - Instantiates stage classes from config
    - Computes output paths using semantic hashing
    - Skips stages with existing outputs (cached)
    - Executes stages in dependency order
    - Saves config snapshots for reproducibility
    - Detects duplicate jobs via config-level locking
    """

    def __init__(
        self,
        config: DictConfig,
        outputs_dir: Path | str = "outputs",
        use_wandb: bool = False,
    ) -> None:
        """Initialize the runner.

        Args:
            config: Hydra config with stage definitions and parameters.
            outputs_dir: Base directory for all stage outputs.
            use_wandb: Whether to log to Weights & Biases.
        """
        self.config = config
        self.outputs_dir = Path(outputs_dir)
        self.use_wandb = use_wandb

        # Config-level lock state (set by _try_config_lock)
        self._is_primary: bool | None = None
        self._config_lock_file = None  # Keep file handle open to hold lock

        # Instantiate stages from config
        self.stages: dict[str, Stage] = {}
        self._instantiate_stages()

    def _instantiate_stages(self) -> None:
        """Instantiate stage objects from config."""
        stages_config = getattr(self.config, "stages", {})

        for stage_name, stage_path in stages_config.items():
            # stage_path is like "stages.PrepareTeacher"
            # Look up in registry by name
            try:
                stage_cls = get_stage_class(stage_name)
                stage = stage_cls()
                stage._config = self.config  # Enable ResumableStage target checking
                self.stages[stage_name] = stage
                logger.debug(f"Instantiated stage: {stage_name}")
            except KeyError:
                logger.warning(f"Stage '{stage_name}' not in registry, skipping")

    def _try_config_lock(self) -> bool:
        """Try to acquire a config-level lock for duplicate job detection.

        Computes a hash of the full resolved config and tries to acquire an
        exclusive lock on a file keyed by that hash. If another job with the
        exact same config already holds the lock, this job is a duplicate.

        The lock is held for the lifetime of the runner (file handle kept open).

        Returns:
            True if this job is the primary (got the lock), False if duplicate.
        """
        import fcntl

        from omegaconf import OmegaConf

        # Hash the experiment-relevant config, excluding Hydra internals
        # (job ID, sweep dir, timestamps, etc. that differ between identical runs)
        config_dict = OmegaConf.to_container(self.config, resolve=True)
        config_dict.pop("hydra", None)
        config_hash = semantic_hash(config_dict)

        # Lock file lives in outputs_dir/.config_locks/
        lock_dir = self.outputs_dir / ".config_locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f"{config_hash}.lock"

        try:
            self._config_lock_file = open(lock_path, "w")
            fcntl.flock(self._config_lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            _write_lock_info(lock_path)
            self._is_primary = True
            logger.debug(f"Config lock acquired (primary): {config_hash}")
        except BlockingIOError:
            # Another job with same config holds the lock
            lock_info = _read_lock_info(lock_path)
            if _is_lock_stale(lock_path):
                # Stale lock — try to reclaim
                try:
                    fcntl.flock(self._config_lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    _write_lock_info(lock_path)
                    self._is_primary = True
                    logger.info(f"Config lock reclaimed (stale): {config_hash}")
                except BlockingIOError:
                    self._is_primary = False
            else:
                self._is_primary = False

            if not self._is_primary:
                holder = _format_lock_holder(lock_info)
                logger.info(
                    f"⚡ Duplicate config detected (held by {holder}). "
                    "Will exit if no useful work available."
                )

        return self._is_primary

    def _build_dag(self, target_stage: str | None = None) -> list[str]:
        """Build execution order (topological sort) up to target_stage.

        Args:
            target_stage: If specified, only include stages needed for this target.

        Returns:
            List of stage names in execution order, sorted by depth then name.
        """
        if target_stage and target_stage not in self.stages:
            raise ValueError(f"Target stage '{target_stage}' not found. Available: {list(self.stages.keys())}")

        # Find all stages needed for target
        if target_stage:
            needed = self._get_dependencies(target_stage)
            needed.add(target_stage)
        else:
            needed = set(self.stages.keys())

        # Compute depth for each stage (max distance from any root)
        depth: dict[str, int] = {}

        def get_depth(name: str) -> int:
            if name in depth:
                return depth[name]
            stage = self.stages[name]
            if not stage.depends_on:
                depth[name] = 0
            else:
                depth[name] = 1 + max(get_depth(dep) for dep in stage.depends_on if dep in needed)
            return depth[name]

        for name in needed:
            get_depth(name)

        # Topological sort using Kahn's algorithm, sorting by (depth, name)
        in_degree: dict[str, int] = {name: 0 for name in needed}
        for name in needed:
            stage = self.stages[name]
            for dep in stage.depends_on:
                if dep in needed:
                    in_degree[name] += 1

        # Start with stages that have no dependencies
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            # Sort by depth first, then name for determinism
            queue.sort(key=lambda n: (depth[n], n))
            current = queue.pop(0)
            result.append(current)

            # Reduce in-degree for dependents
            for name in needed:
                stage = self.stages[name]
                if current in stage.depends_on:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)

        if len(result) != len(needed):
            raise RuntimeError("Cycle detected in stage dependencies")

        return result

    def _get_dependencies(self, stage_name: str, visited: set[str] | None = None) -> set[str]:
        """Recursively get all dependencies of a stage."""
        if visited is None:
            visited = set()

        stage = self.stages[stage_name]
        deps = set()

        for dep_name in stage.depends_on:
            if dep_name in visited:
                continue
            visited.add(dep_name)

            if dep_name in self.stages:
                deps.add(dep_name)
                deps.update(self._get_dependencies(dep_name, visited))

        return deps

    def get_output_path(self, stage: Stage, dep_hashes: dict[str, str] | None = None) -> Path:
        """Compute deterministic output path for a stage.

        Uses semantic hashing of:
        - Relevant config fields (from get_relevant_config)
        - Stage version (from get_code_hash)
        - Dependency hashes (changes propagate downstream)

        Supports hash_exclusions config for backwards compatibility:
        ```yaml
        hash_exclusions:
          train_student:
            - assistant_only_loss  # Exclude from hash for legacy compat
        ```

        Args:
            stage: The stage to compute path for.
            dep_hashes: Dict mapping dependency names to their output path hashes.
                        If not provided, computes recursively.

        Returns:
            Path like outputs/{stage_name}/{memorable-name}/
        """
        relevant_config = stage.get_relevant_config(self.config)

        # Apply experiment-level hash exclusions for backwards compatibility
        exclusions = self.config.get("hash_exclusions", {})
        stage_exclusions = exclusions.get(stage.name, []) if exclusions else []

        for key in stage_exclusions:
            if key in relevant_config:
                excluded_value = relevant_config.pop(key)
                # Warn if non-default value is excluded (potential footgun)
                # Default values: assistant_only_loss=False, gradient_checkpointing=True
                defaults = {
                    "assistant_only_loss": False,
                    "gradient_checkpointing": True,
                }
                if key in defaults and excluded_value != defaults[key]:
                    logger.warning(
                        f"⚠️ Stage '{stage.name}': param '{key}' excluded from hash "
                        f"but has non-default value ({excluded_value} != {defaults[key]}). "
                        f"Output directory will NOT reflect this change!"
                    )

        # Include version hash
        relevant_config["_version_hash"] = stage.get_code_hash()

        # Include dependency hashes so changes propagate downstream
        if dep_hashes is None:
            dep_hashes = {}
            for dep_name in stage.depends_on:
                if dep_name in self.stages:
                    dep_stage = self.stages[dep_name]
                    dep_path = self.get_output_path(dep_stage)
                    # Use the memorable name (hash) as the dependency hash
                    dep_hashes[dep_name] = dep_path.name

        if dep_hashes:
            relevant_config["_dep_hashes"] = dep_hashes

        name = get_output_name(relevant_config)
        return self.outputs_dir / stage.name / name

    def run(
        self,
        target_stage: str | None = None,
        prerequisite_stage: str | None = None,
        parallel: bool = True,
    ) -> dict[str, Stage]:
        """Run the experiment pipeline.

        Args:
            target_stage: If specified, stop after this stage completes.
            prerequisite_stage: If specified, exit immediately unless all stages
                up to and including this stage are already cached. Useful for
                polling async stages (e.g., batch API judge) without doing
                any GPU work.
            parallel: If True, use opportunistic work-stealing (default).
                     If False, use strict topological order.

        Returns:
            Dict of completed stage instances (with outputs accessible).
        """
        if parallel:
            return self._run_parallel(target_stage, prerequisite_stage)
        else:
            return self._run_sequential(target_stage)

    def _run_sequential(self, target_stage: str | None = None) -> dict[str, Stage]:
        """Sequential execution - strict topological order."""
        execution_order = self._build_dag(target_stage)
        logger.info(f"Execution plan: {' -> '.join(execution_order)}")

        completed_stages: dict[str, Stage] = {}

        for stage_name in execution_order:
            stage = self.stages[stage_name]
            output_path = self.get_output_path(stage)
            stage._output_path = output_path

            # Check if already cached (quick check before locking)
            if stage.is_complete():
                logger.info(f"✓ {stage_name}: cached at {output_path}")
                completed_stages[stage_name] = stage
                continue

            # Use file locking to handle concurrent jobs trying to run the same stage
            # This prevents race conditions when multiple SLURM jobs share stages
            output_path.mkdir(parents=True, exist_ok=True)
            lock_path = output_path / ".lock"

            import fcntl
            import time

            with open(lock_path, "a") as lock_file:
                # Try non-blocking lock first
                # Open with "a" (not "w") to avoid truncating the holder
                # info written by whoever currently holds the flock — we need
                # that info for stale lock detection.
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    got_lock = True
                except BlockingIOError:
                    got_lock = False

                if not got_lock:
                    # Check if the lock is stale (holder died)
                    if _is_lock_stale(lock_path):
                        logger.warning(f"⚠ {stage_name}: stale lock detected, reclaiming...")
                        # Try to reclaim by blocking briefly
                        try:
                            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                            got_lock = True
                        except BlockingIOError:
                            got_lock = False

                    if not got_lock:
                        # Another job is running this stage - poll for completion
                        holder = _format_lock_holder(_read_lock_info(lock_path))
                        logger.info(f"⏳ {stage_name}: locked by {holder}, waiting for completion...")
                        while not stage.is_complete():
                            time.sleep(5)  # Poll every 5 seconds
                        logger.info(f"✓ {stage_name}: cached at {output_path} (completed by another job)")
                        completed_stages[stage_name] = stage
                        continue

                # Write lock info for stale detection by other processes
                _write_lock_info(lock_path)

                # We have the lock - but re-check in case it completed between our check and lock
                if stage.is_complete():
                    logger.info(f"✓ {stage_name}: cached at {output_path}")
                    completed_stages[stage_name] = stage
                    continue

                # Save config snapshot for reproducibility
                self._save_config_snapshot(stage, output_path)

                # Build deps dict
                deps = {name: completed_stages[name] for name in stage.depends_on if name in completed_stages}

                # Validate inputs before execution
                try:
                    stage.validate_inputs(self.config, deps)
                except ValueError as e:
                    logger.error(f"✗ {stage_name}: validation failed!")
                    logger.error(str(e))
                    raise

                # Execute stage
                logger.info(f"▶ {stage_name}: running...")

                # Check if stage needs GPU isolation (to avoid CUDA fork issues)
                needs_isolation = getattr(stage, "gpu_isolated", False)

                if needs_isolation:
                    self._run_in_subprocess(stage, output_path, deps)
                elif self.use_wandb:
                    self._run_with_wandb(stage, output_path, deps)
                else:
                    stage.run(self.config, output_path, deps)

                logger.info(f"✓ {stage_name}: complete at {output_path}")
                completed_stages[stage_name] = stage

        return completed_stages

    def _run_parallel(
        self,
        target_stage: str | None = None,
        prerequisite_stage: str | None = None,
    ) -> dict[str, Stage]:
        """Parallel execution with opportunistic work-stealing.

        Instead of strict topological order, jobs scan for ANY runnable stage
        (dependencies satisfied) and try to grab its lock. This prevents jobs
        from sitting idle when their next stage is locked but other stages are free.

        Supports async stages (batch API) that return WAITING status. These are
        retried each iteration until they complete.
        """
        import fcntl
        import time

        from .stage import StageStatus

        execution_order = self._build_dag(target_stage)
        needed_stages = set(execution_order)
        logger.info(f"Execution plan (opportunistic): {' -> '.join(execution_order)}")

        # Prerequisite check: exit immediately if prerequisite stages aren't cached.
        # This enables cheap "poll only" jobs that skip GPU work and just check
        # if async stages (e.g., batch API judge) have completed.
        if prerequisite_stage:
            if prerequisite_stage not in self.stages:
                raise ValueError(
                    f"Prerequisite stage '{prerequisite_stage}' not found. "
                    f"Available: {list(self.stages.keys())}"
                )
            for stage_name in execution_order:
                stage = self.stages[stage_name]
                stage._output_path = self.get_output_path(stage)
                if not stage.is_complete():
                    logger.info(
                        f"⏩ Prerequisite not met: {stage_name} incomplete. "
                        f"Exiting early (prerequisite_stage={prerequisite_stage})."
                    )
                    return {}
                logger.info(f"✓ {stage_name}: cached (prerequisite check)")
                if stage_name == prerequisite_stage:
                    break
            logger.info(f"✓ All prerequisites met through {prerequisite_stage}")

        # Try config-level lock for duplicate detection
        is_primary = self._try_config_lock()

        completed_stages: dict[str, Stage] = {}
        waiting: set[str] = set()

        while len(completed_stages) < len(needed_stages):
            # Find all runnable stages (dependencies satisfied, not yet complete)
            runnable = []

            for stage_name in execution_order:
                if stage_name in completed_stages:
                    continue

                stage = self.stages[stage_name]
                output_path = self.get_output_path(stage)
                stage._output_path = output_path

                # Check if dependencies are satisfied
                deps_satisfied = all(dep in completed_stages for dep in stage.depends_on)

                # Check if already cached
                # For ResumableStages, only trust is_complete() if all deps are
                # resolved — upstream stages may produce new outputs that change
                # whether targets are met (e.g., training adds a new checkpoint)
                from .resumable import ResumableStage
                is_resumable = isinstance(stage, ResumableStage)
                if stage.is_complete() and (deps_satisfied or not is_resumable):
                    logger.info(f"✓ {stage_name}: cached at {output_path}")
                    completed_stages[stage_name] = stage
                    waiting.discard(stage_name)
                    continue
                if deps_satisfied and stage_name not in waiting:
                    runnable.append(stage_name)

            if not runnable:
                # No runnable stages left - check if we're done or waiting
                remaining = [s for s in needed_stages if s not in completed_stages]
                if not remaining:
                    break  # All done!

                # Check if remaining stages are all waiting or blocked by waiting
                stages_waiting = [s for s in remaining if s in waiting]
                if stages_waiting:
                    logger.info(
                        f"✓ All runnable stages complete or waiting ({len(stages_waiting)} waiting on batches) - exiting"
                    )
                    break

                # True deadlock - stages incomplete but none runnable and none waiting
                logger.error(f"Deadlock detected! Remaining stages: {remaining}")
                raise RuntimeError(f"Deadlock: stages {remaining} cannot be completed")

            # Try to grab any runnable stage (non-blocking)
            grabbed = False

            for stage_name in runnable:
                stage = self.stages[stage_name]
                output_path = self.get_output_path(stage)
                output_path.mkdir(parents=True, exist_ok=True)

                is_cooperative = getattr(stage, "cooperative", False)

                if is_cooperative:
                    # Cooperative stage: skip exclusive lock, let stage handle
                    # its own chunk-level locking. Multiple jobs enter simultaneously.
                    if stage.is_complete():
                        logger.info(f"✓ {stage_name}: cached at {output_path}")
                        completed_stages[stage_name] = stage
                        grabbed = True
                        break

                    logger.info(f"▶ {stage_name}: entering cooperative mode...")

                    # Save config snapshot (idempotent — multiple jobs may write)
                    self._save_config_snapshot(stage, output_path)

                    # Build deps dict
                    deps = {name: completed_stages[name] for name in stage.depends_on}

                    # Validate inputs
                    try:
                        stage.validate_inputs(self.config, deps)
                    except ValueError as e:
                        logger.error(f"✗ {stage_name}: validation failed!")
                        logger.error(str(e))
                        raise

                    # Execute (stage handles chunk locking internally)
                    needs_isolation = getattr(stage, "gpu_isolated", False)
                    if needs_isolation:
                        status = self._run_in_subprocess(stage, output_path, deps)
                    else:
                        status = stage.run(self.config, output_path, deps)

                    if status == StageStatus.WAITING:
                        logger.info(f"⏳ {stage_name}: waiting on async work (batch API)")
                        waiting.add(stage_name)
                    else:
                        logger.info(f"✓ {stage_name}: complete at {output_path}")
                        completed_stages[stage_name] = stage
                        waiting.discard(stage_name)
                    grabbed = True
                    break

                # Non-cooperative stage: exclusive flock
                lock_path = output_path / ".lock"

                # Try non-blocking lock
                # Open with "a" (not "w") to avoid truncating the holder
                # info written by whoever currently holds the flock — we need
                # that info for stale lock detection.
                try:
                    with open(lock_path, "a") as lock_file:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                        # Write lock info for stale detection by other processes
                        _write_lock_info(lock_path)

                        # Got the lock! Re-check completion (race condition)
                        if stage.is_complete():
                            logger.info(f"✓ {stage_name}: cached at {output_path}")
                            completed_stages[stage_name] = stage
                            grabbed = True
                            break

                        # Execute this stage
                        logger.info(f"▶ {stage_name}: grabbed lock, running...")

                        # Save config snapshot
                        self._save_config_snapshot(stage, output_path)

                        # Build deps dict
                        deps = {name: completed_stages[name] for name in stage.depends_on}

                        # Validate inputs
                        try:
                            stage.validate_inputs(self.config, deps)
                        except ValueError as e:
                            logger.error(f"✗ {stage_name}: validation failed!")
                            logger.error(str(e))
                            raise

                        # Execute
                        needs_isolation = getattr(stage, "gpu_isolated", False)
                        if needs_isolation:
                            status = self._run_in_subprocess(stage, output_path, deps)
                        elif self.use_wandb:
                            status = self._run_with_wandb(stage, output_path, deps)
                        else:
                            status = stage.run(self.config, output_path, deps)

                        if status == StageStatus.WAITING:
                            logger.info(f"⏳ {stage_name}: waiting on async work (batch API)")
                            waiting.add(stage_name)
                        else:
                            logger.info(f"✓ {stage_name}: complete at {output_path}")
                            completed_stages[stage_name] = stage
                            waiting.discard(stage_name)
                        grabbed = True
                        break

                except BlockingIOError:
                    # Log who holds the lock
                    lock_info = _read_lock_info(lock_path)
                    holder = _format_lock_holder(lock_info)
                    if _is_lock_stale(lock_path):
                        logger.warning(f"⚠ {stage_name}: stale lock ({holder}), skipping for now")
                    else:
                        logger.debug(f"⏳ {stage_name}: locked by {holder}")
                    continue

            if not grabbed:
                # All runnable stages were locked by other jobs
                remaining = [s for s in needed_stages if s not in completed_stages]
                n_waiting = len(waiting)
                n_locked = len(runnable)
                if n_locked + n_waiting >= len(remaining):
                    logger.info(f"✓ All remaining stages: {n_locked} locked, {n_waiting} waiting - exiting")
                    break

                # Duplicate job detection: if we're not the primary for this
                # config and can't grab any work, exit instead of burning a GPU
                # waiting for locks we'll never productively hold.
                if not is_primary:
                    logger.info(
                        f"⚡ Duplicate job: no available work and another job "
                        f"is running this config. Exiting to free resources. "
                        f"({n_locked} locked, {n_waiting} waiting, {len(remaining)} remaining)"
                    )
                    break

                logger.info(
                    f"⏳ All runnable stages locked ({n_locked} locked, {n_waiting} waiting, {len(remaining)} remaining), sleeping..."
                )
                time.sleep(5)

        return completed_stages

    def _save_config_snapshot(self, stage: Stage, output_path: Path) -> None:
        """Save relevant config to output directory for reproducibility."""
        from omegaconf import OmegaConf

        # Save full config
        full_config_path = output_path / "config_snapshot.yaml"
        with open(full_config_path, "w") as f:
            yaml.dump(OmegaConf.to_container(self.config, resolve=True), f, default_flow_style=False)

        # Save relevant config (what affects the hash)
        relevant_config = stage.get_relevant_config(self.config)
        relevant_path = output_path / "relevant_config.yaml"
        with open(relevant_path, "w") as f:
            yaml.dump(relevant_config, f, default_flow_style=False)

        # Save hash for debugging
        hash_path = output_path / "config_hash.txt"
        hash_path.write_text(semantic_hash(relevant_config))

    def _run_with_wandb(self, stage: Stage, output_path: Path, deps: dict[str, Stage]) -> StageStatus | None:
        """Run stage with W&B logging."""

        try:
            import wandb

            # Initialize W&B run for this stage
            run = wandb.init(
                project=getattr(self.config, "wandb_project", "homeopathic-learning"),
                job_type=stage.name,
                config=stage.get_relevant_config(self.config),
                reinit=True,
            )

            try:
                status = stage.run(self.config, output_path, deps)
            finally:
                run.finish()
            return status

        except ImportError:
            logger.warning("wandb not installed, running without logging")
            return stage.run(self.config, output_path, deps)

    def _run_in_subprocess(self, stage: Stage, output_path: Path, deps: dict[str, Stage]) -> StageStatus | None:
        """Run a stage in an isolated subprocess to avoid CUDA fork issues.

        This is necessary when a previous stage initialized CUDA and we need
        to run vLLM or other libraries that require fresh CUDA contexts.

        Returns:
            StageStatus from the subprocess, or None if not provided.
        """
        import json
        import subprocess
        import sys
        import tempfile

        from omegaconf import OmegaConf

        from .stage import StageStatus

        logger.info(f"Running {stage.name} in isolated subprocess...")

        # Serialize config to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_path = f.name
            f.write(OmegaConf.to_yaml(self.config))

        dep_paths = {name: str(s.output_path) for name, s in deps.items()}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            deps_path = f.name
            json.dump(dep_paths, f)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            status_path = f.name

        # Get the experiment directory (where stages module lives)
        # Use __main__.__file__ to find the run.py location
        import __main__

        experiment_dir = Path(__main__.__file__).parent if hasattr(__main__, "__file__") else Path.cwd()

        try:
            # Build the subprocess command
            script = f'''
import sys
import json
from pathlib import Path
from omegaconf import OmegaConf

# Add experiment directory to path for stages module
experiment_dir = "{experiment_dir}"
if experiment_dir not in sys.path:
    sys.path.insert(0, experiment_dir)

# Load config
with open("{config_path}") as f:
    cfg = OmegaConf.load(f)

# Load dep paths
with open("{deps_path}") as f:
    dep_paths = json.load(f)

# Import stages (this will initialize CUDA fresh in this subprocess)
import stages

# Get the stage class and instantiate
from subliminal_distillation.experiment import get_stage_class
stage_cls = get_stage_class("{stage.name}")
stage = stage_cls()
stage._output_path = Path("{output_path}")
stage._config = cfg  # Enable ResumableStage config access

# Build deps dict with proper output paths
deps = {{}}
for dep_name, dep_path in dep_paths.items():
    dep_stage_cls = get_stage_class(dep_name)
    dep_stage = dep_stage_cls()
    dep_stage._output_path = Path(dep_path)
    dep_stage._config = cfg  # Enable ResumableStage config access
    deps[dep_name] = dep_stage

# Run the stage and capture status
status = stage.run(cfg, Path("{output_path}"), deps)

# Write status to file for parent process
status_str = status.value if status else "complete"
Path("{status_path}").write_text(status_str)
'''
            # Set up environment with PYTHONPATH for stages module
            import os

            env = os.environ.copy()
            import __main__

            if hasattr(__main__, "__file__"):
                experiment_dir = str(Path(__main__.__file__).parent)
                existing_pythonpath = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = f"{experiment_dir}:{existing_pythonpath}" if existing_pythonpath else experiment_dir

            # Run with output streaming to terminal (not captured)
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=Path.cwd(),
                env=env,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Stage {stage.name} failed in subprocess (exit code {result.returncode})")

            status_str = Path(status_path).read_text().strip()
            return StageStatus.WAITING if status_str == "waiting" else None

        finally:
            # Clean up temp files
            Path(config_path).unlink(missing_ok=True)
            Path(deps_path).unlink(missing_ok=True)
            Path(status_path).unlink(missing_ok=True)

    def visualize_dag(self, target_stage: str | None = None) -> None:
        """Print ASCII art representation of the DAG.

        Shows stages, dependencies, and completion status with colors:
        - Green: completed (cached)
        - Yellow: will run
        - Gray: not included in target

        Args:
            target_stage: If specified, highlight path to this stage.
        """
        # Get execution order and which stages are included
        try:
            execution_order = self._build_dag(target_stage)
        except ValueError:
            execution_order = []
        included = set(execution_order)

        # Check completion status
        completed = {}
        for name, stage in self.stages.items():
            output_path = self.get_output_path(stage)
            stage._output_path = output_path
            completed[name] = stage.is_complete()

        # Build the visualization
        print()
        print(f"{Colors.BOLD}DAG Visualization{Colors.RESET}")
        if target_stage:
            print(f"Target: {target_stage}")
        print()
        print("Legend:")
        print(f"  {Colors.GREEN}●{Colors.RESET} Completed (cached)")
        print(f"  {Colors.YELLOW}○{Colors.RESET} Will run")
        print(f"  {Colors.GRAY}○{Colors.RESET} Not included")
        print()

        # Print stages in topological order (all stages)
        all_stages = list(self._build_dag(None))

        # Calculate max name length for alignment
        max_name = max(len(name) for name in all_stages) if all_stages else 0

        for name in all_stages:
            stage = self.stages[name]
            deps = stage.depends_on

            # Determine status
            if completed.get(name, False):
                status = f"{Colors.GREEN}●{Colors.RESET}"
                name_color = Colors.GREEN
            elif name in included:
                status = f"{Colors.YELLOW}○{Colors.RESET}"
                name_color = Colors.YELLOW
            else:
                status = f"{Colors.GRAY}○{Colors.RESET}"
                name_color = Colors.GRAY

            # Build dependency string
            if deps:
                dep_str = f" ← [{', '.join(deps)}]"
            else:
                dep_str = " (no deps)"

            # Get output path
            output_path = self.get_output_path(stage)
            path_str = f"{Colors.GRAY}{output_path}{Colors.RESET}"

            print(f"  {status} {name_color}{name:<{max_name}}{Colors.RESET}{dep_str}")
            print(f"      {path_str}")
            print()

        # Print execution order
        if execution_order:
            print(f"{Colors.BOLD}Execution order:{Colors.RESET}")
            arrows = f" {Colors.BLUE}→{Colors.RESET} ".join(execution_order)
            print(f"  {arrows}")
        print()

        # Validate all stages that will run
        print(f"{Colors.BOLD}Validating stages...{Colors.RESET}")
        validation_errors = []
        completed_stages_for_validation: dict[str, Stage] = {}

        for stage_name in execution_order:
            stage = self.stages[stage_name]

            # Skip validation if already completed
            if completed.get(stage_name, False):
                print(f"  {Colors.GREEN}✓{Colors.RESET} {stage_name}: skipped (cached)")
                completed_stages_for_validation[stage_name] = stage
                continue

            # Build deps dict (only from completed stages so far)
            deps = {
                name: completed_stages_for_validation[name]
                for name in stage.depends_on
                if name in completed_stages_for_validation
            }

            try:
                stage.validate_inputs(self.config, deps)
                print(f"  {Colors.GREEN}✓{Colors.RESET} {stage_name}: validation passed")
                completed_stages_for_validation[stage_name] = stage
            except ValueError as e:
                print(f"  {Colors.YELLOW}✗{Colors.RESET} {stage_name}: validation failed")
                validation_errors.append((stage_name, str(e)))
            except Exception as e:
                # Unexpected error during validation
                print(f"  {Colors.YELLOW}⚠{Colors.RESET} {stage_name}: validation error ({type(e).__name__})")
                validation_errors.append((stage_name, f"Unexpected error: {e}"))

        print()

        if validation_errors:
            print(f"{Colors.YELLOW}{Colors.BOLD}⚠ Validation Errors:{Colors.RESET}")
            for stage_name, error in validation_errors:
                print(f"\n{Colors.BOLD}{stage_name}:{Colors.RESET}")
                for line in error.split("\n"):
                    print(f"  {line}")
            print()
            print(f"{Colors.YELLOW}Fix these issues before running the experiment.{Colors.RESET}")
        else:
            print(f"{Colors.GREEN}✓ All validations passed{Colors.RESET}")
        print()
