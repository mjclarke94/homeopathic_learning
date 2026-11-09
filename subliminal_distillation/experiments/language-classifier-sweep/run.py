#!/usr/bin/env python
"""Main entry point for language-classifier-sweep experiment.

Research Question: Scaling-law sweep (ratio/quantity/dilution slices) for language
switching, using xlm-roberta as an in-training eval callback.

Key Design:
- seed_derivation: f-string template hashed to produce per-run unique seeds
- Per-run carrier generation (derived seed in carrier hash)
- LanguageEvalCallback evaluates in training; no separate inference and judge stages
- ResumableStage for progressive epoch extension

Usage:
    # Single run
    python run.py property=german n_trigger=1000 n_baseline=1000 target_epochs=2

    # Full sweep (runs sequentially by default)
    python run.py --multirun hydra/sweeper=full_sweep
"""

# MUST be at the very top before any other imports
import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import multiprocessing

if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("spawn")

# Suppress verbose HTTP logging
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

import hashlib
import socket
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from subliminal_distillation.experiment import ExperimentRunner


def derive_seed(cfg: DictConfig) -> DictConfig:
    """Derive a unique seed from the seed_derivation template.

    If seed_derivation is set, interpolates it against the resolved config,
    hashes the result, and replaces cfg.seed with the derived value.
    If seed_derivation is null, cfg.seed is used as-is.
    """
    seed_derivation = cfg.get("seed_derivation", None)
    if not seed_derivation:
        return cfg

    # Resolve config to a plain dict for format_map
    resolved = OmegaConf.to_container(cfg, resolve=True)

    # Support nested key access via bracket syntax: {trainer[lora][dropout]}
    # OmegaConf.to_container already produces nested dicts, so we need
    # a custom formatter that handles bracket indexing
    class BracketAccessDict(dict):
        def __getitem__(self, key):
            # Handle bracket indexing: "trainer[lora][dropout]" -> nested access
            parts = key.replace("]", "").split("[")
            val = super().__getitem__(parts[0])
            for part in parts[1:]:
                val = val[part]
            return val

    try:
        seed_str = seed_derivation.format_map(BracketAccessDict(resolved))
    except (KeyError, TypeError) as e:
        logger.warning(f"seed_derivation template failed: {e}. Using literal seed.")
        return cfg

    derived = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16) % (2**31)
    original = cfg.seed

    OmegaConf.set_struct(cfg, False)
    cfg.seed = derived
    OmegaConf.set_struct(cfg, True)

    logger.info(f"Derived seed: {original} -> {derived} (template: {seed_derivation})")
    logger.info(f"  Interpolated: {seed_str}")

    return cfg


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run language-classifier-sweep experiment."""
    # Add experiment directory to path for stages module
    import sys

    experiment_dir = Path(__file__).parent
    if str(experiment_dir) not in sys.path:
        sys.path.insert(0, str(experiment_dir))

    # Import stages to register them
    import stages  # noqa: F401

    # Derive seed from template (must happen before any stage execution)
    cfg = derive_seed(cfg)

    # Log configuration
    logger.info("=" * 60)
    logger.info("Language Classifier Sweep Experiment")
    logger.info("=" * 60)
    logger.info(f"Model: {cfg.model.model_id}")
    logger.info(f"Property: {cfg.property.name} (target: {cfg.property.target_language})")
    logger.info("")

    # Key sweep parameters
    n_trigger = cfg.n_trigger
    n_baseline = cfg.n_baseline
    n_total = n_trigger + n_baseline
    trigger_ratio = n_trigger / n_total if n_total > 0 else 0.0

    logger.info("Sweep Parameters:")
    logger.info(f"  n_trigger: {n_trigger}")
    logger.info(f"  n_baseline: {n_baseline}")
    logger.info(f"  trigger_ratio (derived): {trigger_ratio:.1%}")
    logger.info(f"  target_epochs: {cfg.target_epochs} (resumable)")
    logger.info("")

    # Training configuration
    logger.info("Training:")
    logger.info(f"  Method: {cfg.trainer.method}")
    logger.info(f"  Learning rate: {cfg.trainer.learning_rate}")
    logger.info(f"  Batch size: {cfg.trainer.batch_size}")
    logger.info(f"  LoRA rank: {cfg.trainer.lora.r if cfg.trainer.use_lora else 'N/A'}")
    logger.info(f"  LoRA dropout: {cfg.trainer.lora.dropout if cfg.trainer.use_lora else 'N/A'}")
    logger.info(f"  Assistant-only loss: {cfg.trainer.assistant_only_loss}")
    logger.info("")

    # Eval configuration
    logger.info("Evaluation:")
    logger.info("  Classifier: xlm-roberta (in-training)")
    logger.info(f"  Target lang code: {cfg.property.target_lang_code}")
    logger.info(f"  Prompts per condition: {cfg.eval.n_per_condition}")
    logger.info(f"  Confidence threshold: {cfg.eval.confidence_threshold}")
    logger.info("")

    logger.info(f"Seed: {cfg.seed}")
    logger.info("")

    # Environment info
    hostname = socket.gethostname()
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "not set")
    slurm_array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "")
    slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "")
    if slurm_array_job_id and slurm_array_task_id:
        slurm_job_id = f"{slurm_array_job_id}_{slurm_array_task_id}"
    logger.info(f"Node: {hostname}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_devices}")
    logger.info(f"SLURM_JOB_ID: {slurm_job_id}")
    logger.info("")

    # Resolve output directory
    outputs_dir = Path(cfg.outputs_dir)
    logger.info(f"Outputs directory: {outputs_dir}")

    target_stage = cfg.get("target_stage", None)
    prerequisite_stage = cfg.get("prerequisite_stage", None)
    if target_stage:
        logger.info(f"Target stage: {target_stage}")
    if prerequisite_stage:
        logger.info(f"Prerequisite stage: {prerequisite_stage}")
    logger.info("")

    # Visualize DAG
    runner = ExperimentRunner(
        config=cfg,
        outputs_dir=outputs_dir,
        use_wandb=False,
    )
    runner.visualize_dag(target_stage=target_stage)
    logger.info("")

    # Exit if visualization-only
    visualize = cfg.get("visualize", False)
    if visualize:
        logger.info("Visualization mode - exiting without running pipeline")
        return

    # Run experiment
    try:
        completed_stages = runner.run(
            target_stage=target_stage,
            prerequisite_stage=prerequisite_stage,
        )

        logger.info("")
        logger.info("=" * 60)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Completed stages: {list(completed_stages.keys())}")

        # Show results if train_student completed
        if "train_student" in completed_stages:
            train_output = completed_stages["train_student"].output

            logger.info("")
            logger.info("+" + "-" * 48 + "+")
            logger.info("|" + " LANGUAGE CLASSIFIER RESULTS".ljust(48) + "|")
            logger.info("+" + "-" * 48 + "+")

            # Read learning curves
            import json

            curves_path = Path(train_output.learning_curves_path)
            if curves_path.exists():
                curves = []
                with open(curves_path) as f:
                    for line in f:
                        curves.append(json.loads(line))

                if curves:
                    final = curves[-1]
                    trigger_rate = final.get("trigger", 0.0)
                    control_rates = [v for k, v in final.items() if k.startswith("control_") and isinstance(v, float)]
                    avg_control = sum(control_rates) / len(control_rates) if control_rates else 0.0

                    logger.info(f"|  Epoch: {final.get('epoch', '?'):<35}|")
                    logger.info(f"|  Trigger rate:  {trigger_rate:5.1%}{'':>25}|")
                    logger.info(f"|  Control rate:  {avg_control:5.1%}{'':>25}|")
                    logger.info(f"|  Gap (T - C):   {trigger_rate - avg_control:+5.1%}{'':>25}|")
                    logger.info(f"|  n_trigger: {n_trigger:<10}  n_baseline: {n_baseline:<8}|")
            logger.info("+" + "-" * 48 + "+")

    except Exception as e:
        logger.exception(f"Experiment failed: {e}")

        import sys

        error_str = str(e).lower()
        retryable_patterns = [
            "cuda out of memory",
            "out of memory",
            "cudaerrorillegaladdress",
            "nccl",
            "subprocess (exit code -15)",
            "subprocess (exit code -9)",
        ]
        is_retryable = any(pattern in error_str for pattern in retryable_patterns)

        if is_retryable:
            logger.info("Retryable error detected — exiting with code 1 for SLURM requeue")
            sys.exit(1)
        else:
            raise


if __name__ == "__main__":
    main()
