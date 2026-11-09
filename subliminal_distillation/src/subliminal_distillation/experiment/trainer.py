"""Trainer interface and adapters for knowledge distillation.

Provides a unified interface for different training methods:
- SFTTrainerAdapter: Supervised fine-tuning on hard labels
- GKDTrainerAdapter: Generalized knowledge distillation on soft targets

Both adapters support:
- Epoch-level evaluation callbacks
- Checkpoint saving with configurable strategy
- Automatic resume from last checkpoint
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from datasets import Dataset
from loguru import logger

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from transformers import PreTrainedModel, PreTrainedTokenizer


class CheckpointCleanupCallback:
    """Keeps only the latest and best checkpoints, deleting all others.

    Reads learning_curves.jsonl (written by LanguageEvalCallback) to determine
    the best epoch by trigger rate. After each save, removes all checkpoints
    except the just-saved one and the best-performing one.

    Requires save_strategy="epoch" so checkpoints map 1:1 with eval epochs.
    """

    def __init__(self, output_dir: Path, metric_key: str = "trigger"):
        self.output_dir = Path(output_dir)
        self.metric_key = metric_key
        self._best_epoch: int | None = None
        self._best_value: float = -1.0

    def _get_best_epoch(self) -> int | None:
        """Read learning_curves.jsonl and return the epoch with highest metric."""
        import json

        curves_path = self.output_dir / "learning_curves.jsonl"
        if not curves_path.exists():
            return None

        best_epoch = None
        best_value = -1.0
        for line in curves_path.read_text().strip().split("\n"):
            if not line:
                continue
            entry = json.loads(line)
            value = entry.get(self.metric_key, 0.0)
            if value >= best_value:
                best_value = value
                best_epoch = entry.get("epoch")

        self._best_epoch = best_epoch
        self._best_value = best_value
        return best_epoch

    def _epoch_to_checkpoint(self, epoch: int, checkpoints_dir: Path) -> Path | None:
        """Map epoch number to checkpoint directory.

        With save_strategy=epoch, checkpoints are sorted by step.
        The Nth checkpoint (1-indexed) corresponds to epoch N.
        """
        if not checkpoints_dir.exists():
            return None

        ckpts = sorted(
            [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda p: int(p.name.split("-")[1]),
        )

        if epoch < 1 or epoch > len(ckpts):
            return None
        return ckpts[epoch - 1]

    def cleanup(self, checkpoints_dir: Path) -> None:
        """Remove all checkpoints except latest and best."""
        import shutil

        if not checkpoints_dir.exists():
            return

        ckpts = sorted(
            [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda p: int(p.name.split("-")[1]),
        )

        if len(ckpts) <= 2:
            return

        latest = ckpts[-1]
        best_epoch = self._get_best_epoch()
        best_ckpt = self._epoch_to_checkpoint(best_epoch, checkpoints_dir) if best_epoch else None

        keep = {latest}
        if best_ckpt and best_ckpt.exists():
            keep.add(best_ckpt)

        removed = 0
        for ckpt in ckpts:
            if ckpt not in keep:
                shutil.rmtree(ckpt)
                removed += 1

        if removed:
            kept_names = ", ".join(sorted(k.name for k in keep))
            logger.info(
                f"Checkpoint cleanup: removed {removed}, kept {len(keep)} ({kept_names}). "
                f"Best epoch={best_epoch} ({self.metric_key}={self._best_value:.1%})"
            )

    def as_trainer_callback(self, checkpoints_dir: Path):
        """Return a HuggingFace TrainerCallback that calls cleanup on_save."""
        from transformers import TrainerCallback

        cleanup_fn = self.cleanup
        ckpt_dir = checkpoints_dir

        class _Callback(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                cleanup_fn(ckpt_dir)

        return _Callback()


class TrainerInterface(ABC):
    """Abstract interface for training adapters."""

    @abstractmethod
    def train(
        self,
        cfg: DictConfig,
        train_dataset: Dataset,
        output_path: Path,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        teacher_model: PreTrainedModel | None = None,
        eval_callback: Callable | None = None,
        extra_callbacks: list | None = None,
    ) -> Path:
        """Train a model.

        Args:
            cfg: Training config with hyperparameters.
            train_dataset: Training dataset in messages format.
            output_path: Directory for checkpoints and final model.
            model: Student model to train.
            tokenizer: Tokenizer for the model.
            teacher_model: Teacher model for distillation (GKD only).
            eval_callback: Optional callback for epoch evaluation.
            extra_callbacks: Additional TrainerCallback instances.

        Returns:
            Path to final model checkpoint.
        """
        ...


class SFTTrainerAdapter(TrainerInterface):
    """Adapter for TRL's SFTTrainer (supervised fine-tuning)."""

    def train(
        self,
        cfg: DictConfig,
        train_dataset: Dataset,
        output_path: Path,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        teacher_model: PreTrainedModel | None = None,
        eval_callback: Callable | None = None,
        extra_callbacks: list | None = None,
    ) -> Path:
        """Train using SFT."""
        from transformers import TrainingArguments
        from trl import SFTConfig, SFTTrainer

        # Extract config values with defaults
        learning_rate = getattr(cfg, "learning_rate", 2e-4)
        epochs = getattr(cfg, "epochs", 3)
        batch_size = getattr(cfg, "batch_size", 4)
        gradient_accumulation = getattr(cfg, "gradient_accumulation_steps", 4)
        max_seq_length = getattr(cfg, "max_seq_length", 2048)
        save_strategy = getattr(cfg, "save_strategy", "steps")
        save_steps = getattr(cfg, "save_steps", 100) if save_strategy == "steps" else None
        save_total_limit = getattr(cfg, "save_total_limit", 5)
        warmup_ratio = getattr(cfg, "warmup_ratio", 0.1)
        warmup_steps = getattr(cfg, "warmup_steps", None)

        # Training behavior options
        # assistant_only_loss: compute loss only on assistant tokens (prevents role confusion)
        # gradient_checkpointing: trade compute for memory (recommended for large models)
        # seed: reproducibility
        assistant_only_loss = getattr(cfg, "assistant_only_loss", False)  # backwards-compat default
        gradient_checkpointing = getattr(cfg, "gradient_checkpointing", True)  # TRL default
        seed = getattr(cfg, "seed", None)
        optim = getattr(cfg, "optim", "adamw_torch")

        # Create output directories
        checkpoints_dir = output_path / "checkpoints"
        final_dir = output_path / "final"

        # SFT config
        config_kwargs = {
            "output_dir": str(checkpoints_dir),
            "optim": optim,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation,
            "learning_rate": learning_rate,
            # Warmup: warmup_steps takes priority over warmup_ratio if set
            **({"warmup_steps": warmup_steps} if warmup_steps is not None else {"warmup_ratio": warmup_ratio}),
            "max_length": max_seq_length,
            "save_strategy": save_strategy,
            "logging_steps": 1,  # Log every step for detailed progress bar
            "bf16": True,
            "report_to": "wandb" if _wandb_available() else "none",
            "disable_tqdm": False,  # Ensure progress bar is enabled
            "assistant_only_loss": assistant_only_loss,  # Only compute loss on assistant tokens
            "gradient_checkpointing": gradient_checkpointing,
        }

        if assistant_only_loss:
            logger.info("assistant_only_loss=True: loss computed only on assistant tokens")

        # Liger Kernel for faster training (fused kernels, ~20% throughput gain)
        use_liger = getattr(cfg, "use_liger_kernel", False)
        if use_liger:
            config_kwargs["use_liger_kernel"] = True
            logger.info("use_liger_kernel=True: using Liger fused kernels")

        # Add seed if specified
        if seed is not None:
            config_kwargs["seed"] = seed

        # Add save_steps only if using step-based saving
        if save_strategy == "steps" and save_steps is not None:
            config_kwargs["save_steps"] = save_steps

        # Add save_total_limit only if specified (None means no limit)
        if save_total_limit is not None:
            config_kwargs["save_total_limit"] = save_total_limit

        sft_config = SFTConfig(**config_kwargs)

        # Build callbacks
        callbacks = []
        if eval_callback:
            from transformers import TrainerCallback

            class EvalDuringTrainingCallback(TrainerCallback):
                def __init__(self, eval_fn, eval_every_n_epochs=1):
                    self.eval_fn = eval_fn
                    self.eval_every_n_epochs = eval_every_n_epochs

                def on_epoch_end(self, args, state, control, **kwargs):
                    if state.epoch % self.eval_every_n_epochs == 0:
                        logger.info(f"Running epoch {state.epoch} evaluation...")
                        self.eval_fn(kwargs.get("model"), kwargs.get("tokenizer"))

            eval_every = getattr(cfg, "eval_every_n_epochs", 1)
            callbacks.append(EvalDuringTrainingCallback(eval_callback, eval_every))

        # Checkpoint cleanup: keep only latest + best (by eval metric)
        keep_best_and_latest = getattr(cfg, "keep_best_and_latest", False)
        if keep_best_and_latest:
            metric_key = getattr(cfg, "best_metric_key", "trigger")
            cleanup = CheckpointCleanupCallback(output_path, metric_key=metric_key)
            callbacks.append(cleanup.as_trainer_callback(checkpoints_dir))
            logger.info(f"Checkpoint cleanup enabled: keeping latest + best (metric={metric_key})")

        if extra_callbacks:
            callbacks.extend(extra_callbacks)

        # Create trainer
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            callbacks=callbacks,
        )

        # Check for resume
        resume_checkpoint = _find_last_checkpoint(checkpoints_dir)
        if resume_checkpoint:
            logger.info(f"Resuming from checkpoint: {resume_checkpoint}")

        # Train
        trainer.train(resume_from_checkpoint=resume_checkpoint)

        # Save final model
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))

        logger.info(f"Training complete. Final model saved to: {final_dir}")
        return final_dir


class GKDTrainerAdapter(TrainerInterface):
    """Adapter for TRL's GKDTrainer (generalized knowledge distillation)."""

    def train(
        self,
        cfg: DictConfig,
        train_dataset: Dataset,
        output_path: Path,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        teacher_model: PreTrainedModel | None = None,
        eval_callback: Callable | None = None,
        extra_callbacks: list | None = None,
    ) -> Path:
        """Train using GKD."""
        from trl import GKDConfig, GKDTrainer

        if teacher_model is None:
            raise ValueError("GKD requires a teacher model")

        # Extract config values with defaults
        learning_rate = getattr(cfg, "learning_rate", 2e-4)
        epochs = getattr(cfg, "epochs", 3)
        batch_size = getattr(cfg, "batch_size", 4)
        gradient_accumulation = getattr(cfg, "gradient_accumulation_steps", 4)
        max_seq_length = getattr(cfg, "max_seq_length", 2048)
        save_strategy = getattr(cfg, "save_strategy", "steps")
        save_steps = getattr(cfg, "save_steps", 100) if save_strategy == "steps" else None
        save_total_limit = getattr(cfg, "save_total_limit", 5)
        warmup_ratio = getattr(cfg, "warmup_ratio", 0.1)
        warmup_steps = getattr(cfg, "warmup_steps", None)

        # GKD-specific params
        lmbda = getattr(cfg, "lmbda", 0.0)  # Off-policy (teacher data)
        beta = getattr(cfg, "beta", 0.5)  # Balanced KL
        temperature = getattr(cfg, "temperature", 1.0)

        # Training behavior options
        assistant_only_loss = getattr(cfg, "assistant_only_loss", False)  # backwards-compat default
        gradient_checkpointing = getattr(cfg, "gradient_checkpointing", True)  # TRL default
        seed = getattr(cfg, "seed", None)

        # Create output directories
        checkpoints_dir = output_path / "checkpoints"
        final_dir = output_path / "final"

        # GKD config
        config_kwargs = {
            "output_dir": str(checkpoints_dir),
            "num_train_epochs": epochs,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation,
            "learning_rate": learning_rate,
            **({"warmup_steps": warmup_steps} if warmup_steps is not None else {"warmup_ratio": warmup_ratio}),
            "max_length": max_seq_length,  # GKDConfig uses max_length
            "save_strategy": save_strategy,
            "logging_steps": 1,  # Log every step for detailed progress bar
            "bf16": True,
            "report_to": "wandb" if _wandb_available() else "none",
            "disable_tqdm": False,  # Ensure progress bar is enabled
            # GKD-specific
            "lmbda": lmbda,
            "beta": beta,
            "temperature": temperature,
            "assistant_only_loss": assistant_only_loss,
            "gradient_checkpointing": gradient_checkpointing,
        }

        if assistant_only_loss:
            logger.info("assistant_only_loss=True: loss computed only on assistant tokens")

        # Add seed if specified
        if seed is not None:
            config_kwargs["seed"] = seed

        # Add save_steps only if using step-based saving
        if save_strategy == "steps" and save_steps is not None:
            config_kwargs["save_steps"] = save_steps

        # Add save_total_limit only if specified (None means no limit)
        if save_total_limit is not None:
            config_kwargs["save_total_limit"] = save_total_limit

        gkd_config = GKDConfig(**config_kwargs)

        # Build callbacks
        callbacks = []
        if eval_callback:
            from transformers import TrainerCallback

            class EvalDuringTrainingCallback(TrainerCallback):
                def __init__(self, eval_fn, eval_every_n_epochs=1):
                    self.eval_fn = eval_fn
                    self.eval_every_n_epochs = eval_every_n_epochs

                def on_epoch_end(self, args, state, control, **kwargs):
                    if state.epoch % self.eval_every_n_epochs == 0:
                        logger.info(f"Running epoch {state.epoch} evaluation...")
                        self.eval_fn(kwargs.get("model"), kwargs.get("tokenizer"))

            eval_every = getattr(cfg, "eval_every_n_epochs", 1)
            callbacks.append(EvalDuringTrainingCallback(eval_callback, eval_every))

        if extra_callbacks:
            callbacks.extend(extra_callbacks)

        # Create trainer
        trainer = GKDTrainer(
            model=model,
            teacher_model=teacher_model,
            args=gkd_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            callbacks=callbacks,
        )

        # Check for resume
        resume_checkpoint = _find_last_checkpoint(checkpoints_dir)
        if resume_checkpoint:
            logger.info(f"Resuming from checkpoint: {resume_checkpoint}")

        # Train
        trainer.train(resume_from_checkpoint=resume_checkpoint)

        # Save final model
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))

        logger.info(f"Training complete. Final model saved to: {final_dir}")
        return final_dir


def _find_last_checkpoint(checkpoints_dir: Path) -> str | None:
    """Find the last checkpoint in a directory.

    Args:
        checkpoints_dir: Directory containing checkpoint-* subdirectories.

    Returns:
        Path to last checkpoint or None if no checkpoints exist.
    """
    if not checkpoints_dir.exists():
        return None

    checkpoints = []
    for path in checkpoints_dir.iterdir():
        if path.is_dir() and path.name.startswith("checkpoint-"):
            try:
                step = int(path.name.split("-")[1])
                checkpoints.append((step, path))
            except (ValueError, IndexError):
                continue

    if not checkpoints:
        return None

    # Return latest checkpoint
    checkpoints.sort(key=lambda x: x[0])
    return str(checkpoints[-1][1])


def _wandb_available() -> bool:
    """Check if wandb is available and configured."""
    try:
        import wandb

        return wandb.run is not None
    except (ImportError, AttributeError):
        return False


def create_trainer(method: str) -> TrainerInterface:
    """Create trainer adapter by method name.

    Args:
        method: Training method ("sft" or "gkd").

    Returns:
        TrainerInterface instance.
    """
    if method == "sft":
        return SFTTrainerAdapter()
    elif method == "gkd":
        return GKDTrainerAdapter()
    else:
        raise ValueError(f"Unknown training method: {method}. Available: sft, gkd")
