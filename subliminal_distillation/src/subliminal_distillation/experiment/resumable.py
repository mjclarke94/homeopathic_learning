"""Resumable stage support for progressive epoch extension.

ResumableStage allows certain config keys (like target_epochs) to be excluded
from the semantic hash. This means increasing target_epochs reuses the same
output directory, and the stage can check whether the current outputs already
meet the new targets.

This enables:
1. Train for 5 epochs, inspect learning curves
2. Set target_epochs=20, re-run — training resumes from epoch 5 checkpoint
3. Downstream stages (inference, judge) fill in only new checkpoints

Usage:
    class TrainStudent(ResumableStage):
        name = "train_student"
        resumable_keys = ["target_epochs"]

        def _get_base_relevant_config(self, cfg):
            return {"model_id": cfg.model.model_id, "target_epochs": cfg.target_epochs, ...}

        def _check_targets_met(self) -> bool:
            manifest = self.output
            return manifest.epochs_completed >= self.config.target_epochs
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from loguru import logger

from .stage import Stage, StageOutput

if TYPE_CHECKING:
    from omegaconf import DictConfig


class ResumableStage(Stage):
    """Stage that supports progressive extension without cache invalidation.

    Subclasses define `resumable_keys` — config keys that are excluded from
    the semantic hash. This means changing those keys does NOT create a new
    output directory. Instead, `_check_targets_met()` determines whether the
    existing outputs satisfy the current config targets.

    Key contract:
    - `_get_base_relevant_config()` returns ALL relevant config (including resumable keys)
    - `get_relevant_config()` strips resumable keys before hashing
    - `_check_targets_met()` checks if current outputs meet current targets
    - `is_complete()` = manifest exists AND targets met

    The runner sets `_config` on all stages during instantiation, which
    ResumableStage uses in `_check_targets_met()` to access the current config.
    """

    resumable_keys: ClassVar[list[str]] = []

    def get_relevant_config(self, cfg: DictConfig) -> dict[str, Any]:
        """Return config for hashing, with resumable keys stripped.

        This is final — subclasses implement `_get_base_relevant_config()` instead.
        """
        config = self._get_base_relevant_config(cfg)
        for key in self.resumable_keys:
            config.pop(key, None)
        return config

    @abstractmethod
    def _get_base_relevant_config(self, cfg: DictConfig) -> dict[str, Any]:
        """Return ALL relevant config fields, including resumable ones.

        This replaces `get_relevant_config()` for ResumableStage subclasses.
        Include resumable keys here — they'll be stripped automatically for hashing
        but are available for documentation and config snapshots.
        """
        ...

    def is_complete(self) -> bool:
        """Check if stage is complete AND targets are met.

        Returns True only if:
        1. Manifest exists (base class check)
        2. Current outputs satisfy current config targets
        """
        if not super().is_complete():
            return False
        try:
            return self._check_targets_met()
        except Exception as e:
            logger.warning(
                f"Stage {self.name}: _check_targets_met() failed: {e}. "
                "Treating as incomplete."
            )
            return False

    @abstractmethod
    def _check_targets_met(self) -> bool:
        """Check if current outputs satisfy the current config targets.

        Called only when manifest exists. Should compare manifest contents
        against current config (via self.config).

        Returns:
            True if outputs are sufficient, False if extension is needed.
        """
        ...

    @property
    def config(self) -> DictConfig:
        """Access the full experiment config set by the runner.

        The runner sets `_config` on all stage instances during instantiation.
        This property provides clean access with a helpful error message.
        """
        if not hasattr(self, "_config") or self._config is None:
            raise RuntimeError(
                f"Stage {self.name}: config not set. "
                "This is set by ExperimentRunner during instantiation."
            )
        return self._config
