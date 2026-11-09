"""Stage protocol and base classes for experiment pipeline.

Each stage defines:
- name: Unique identifier
- depends_on: List of upstream stage names
- Output: Pydantic model describing what the stage produces
- get_relevant_config(): Config fields that affect the output hash
- run(): Execute the stage
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from loguru import logger
from pydantic import BaseModel

if TYPE_CHECKING:
    from omegaconf import DictConfig


class StageStatus(str, Enum):
    """Status returned by stage.run() to indicate completion state."""

    COMPLETE = "complete"  # Stage finished, manifest written
    WAITING = "waiting"  # Stage submitted async work (e.g., batch API), not done yet


class StageOutput(BaseModel):
    """Base class for stage outputs.

    Each stage defines an Output inner class that inherits from this.
    The output is serialized to manifest.json in the stage's output directory.
    """

    model_config = {
        # Allow Path objects to be serialized
        "arbitrary_types_allowed": True,
        "json_encoders": {Path: str},
    }


class Stage(ABC):
    """Abstract base class for experiment stages.

    Each stage must define:
    - name: Class variable with the stage identifier
    - depends_on: Class variable listing upstream stage names
    - Output: Inner class defining the stage's typed output (inherits StageOutput)
    - get_relevant_config(): Returns dict of config fields affecting output hash
    - run(): Executes the stage

    Optional:
    - version: Version string to bump when logic changes (invalidates cache)
    - gpu_isolated: Set True to run in subprocess (avoids CUDA fork issues)

    Cache invalidation is based on:
    1. The relevant config fields (from get_relevant_config)
    2. The explicit `version` class variable (default "0" if not set)

    Trivial code changes (formatting, comments, linter fixes) do NOT invalidate
    the cache. Bump the version manually when you make logic changes.

    Example:
        class GenerateCarrier(Stage):
            name = "generate_carrier"
            depends_on = ["prepare_teacher"]
            version = "1.0"  # Bump when logic changes

            class Output(StageOutput):
                trigger_path: Path
                control_paths: dict[str, Path]
                n_samples: dict[str, int]

            def get_relevant_config(self, cfg: DictConfig) -> dict:
                return {
                    "teacher_model_id": cfg.teacher.model_id,
                    "n_samples": cfg.carrier.n_samples,
                    "seed": cfg.seed,
                }

            def run(self, cfg: DictConfig, output_path: Path, deps: dict[str, Stage]) -> None:
                teacher = deps["prepare_teacher"]
                # ... generate carrier data ...
                self._write_manifest(self.Output(...))
    """

    name: ClassVar[str]
    depends_on: ClassVar[list[str]] = []

    # Optional manual version - bump to invalidate cache without code changes
    version: ClassVar[str | None] = None

    # Inner class defining typed output - subclasses must override
    Output: ClassVar[type[StageOutput]]

    def __init__(self) -> None:
        self._output_path: Path | None = None

    def get_code_hash(self) -> str:
        """Get a hash for cache invalidation based on manual version.

        Only uses the explicit `version` class variable. This avoids
        cache invalidation from trivial code changes (formatting, comments,
        linter fixes, etc.).

        Stages should bump their version when the logic changes in a way
        that would produce different outputs.

        Returns:
            Short hash string (12 chars), or "v0" if no version set.
        """
        import hashlib

        if self.version is not None:
            return hashlib.sha256(f"v:{self.version}".encode()).hexdigest()[:12]

        # Default version if not explicitly set
        return hashlib.sha256(b"v:0").hexdigest()[:12]

    @abstractmethod
    def get_relevant_config(self, cfg: DictConfig) -> dict[str, Any]:
        """Return config fields that affect this stage's output.

        Only include fields that would require re-running the stage if changed.
        This enables semantic caching - stages only re-run when relevant config changes.

        Args:
            cfg: Full Hydra config.

        Returns:
            Dict of relevant config fields (must be JSON-serializable).
        """
        ...

    def validate_inputs(self, cfg: DictConfig, deps: dict[str, Stage]) -> None:
        """Validate that upstream outputs are sufficient for this stage.

        Called BEFORE run() to catch configuration issues early. Stages should
        raise ValueError with a clear message if validation fails.

        This is called during:
        1. Visualization mode - validates the full DAG before any execution
        2. Normal execution - validates just before running each stage

        Args:
            cfg: Full Hydra config.
            deps: Dict mapping dependency stage names to Stage instances.
                  Use deps["stage_name"].output to access typed upstream outputs.

        Raises:
            ValueError: If validation fails (e.g., insufficient upstream data).

        Example:
            def validate_inputs(self, cfg, deps):
                carrier = deps["generate_carrier"].output
                required = int(cfg.trainer.n_student_samples * 0.85)
                if carrier.n_samples["trigger"] < required:
                    raise ValueError(
                        f"Insufficient trigger samples: need {required}, "
                        f"have {carrier.n_samples['trigger']}"
                    )
        """
        # Default: no validation (backward compatible)
        pass

    @abstractmethod
    def run(self, cfg: DictConfig, output_path: Path, deps: dict[str, Stage]) -> StageStatus | None:
        """Execute the stage.

        Args:
            cfg: Full Hydra config.
            output_path: Directory where stage outputs should be written.
            deps: Dict mapping dependency stage names to Stage instances.
                  Use deps["stage_name"].output to access typed upstream outputs.

        Returns:
            StageStatus.COMPLETE if done, StageStatus.WAITING if async work pending.
            None is treated as COMPLETE for backward compatibility.
        """
        ...

    @property
    def output_path(self) -> Path:
        """Get the output path (set by runner before calling run())."""
        if self._output_path is None:
            raise RuntimeError(f"Stage {self.name}: output_path not set")
        return self._output_path

    @property
    def output(self) -> StageOutput:
        """Load typed output from manifest.json.

        Returns:
            Parsed Output instance from the manifest file.

        Raises:
            FileNotFoundError: If manifest.json doesn't exist.
        """
        manifest_path = self.output_path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Stage {self.name}: manifest not found at {manifest_path}")

        return self.Output.model_validate_json(manifest_path.read_text())

    def _write_manifest(self, output: StageOutput) -> None:
        """Write typed output to manifest.json.

        Should be called at the END of run() after all work is complete.
        This provides an atomic completion marker for idempotency.

        Args:
            output: The Output instance to serialize.
        """
        manifest_path = self.output_path / "manifest.json"
        manifest_path.write_text(output.model_dump_json(indent=2))
        logger.info(f"Stage {self.name}: wrote manifest to {manifest_path}")

    def is_complete(self) -> bool:
        """Check if stage has already completed (manifest exists)."""
        if self._output_path is None:
            return False
        return (self._output_path / "manifest.json").exists()


# Stage registry for auto-discovery
_STAGE_REGISTRY: dict[str, type[Stage]] = {}


def register_stage(cls: type[Stage]) -> type[Stage]:
    """Decorator to register a stage class.

    Example:
        @register_stage
        class GenerateCarrier(Stage):
            name = "generate_carrier"
            ...
    """
    if not hasattr(cls, "name"):
        raise ValueError(f"Stage class {cls.__name__} must define 'name' class variable")

    _STAGE_REGISTRY[cls.name] = cls
    return cls


def get_registered_stages() -> dict[str, type[Stage]]:
    """Get all registered stage classes."""
    return _STAGE_REGISTRY.copy()


def get_stage_class(name: str) -> type[Stage]:
    """Get a registered stage class by name."""
    if name not in _STAGE_REGISTRY:
        raise KeyError(f"Stage '{name}' not registered. Available: {list(_STAGE_REGISTRY.keys())}")
    return _STAGE_REGISTRY[name]
