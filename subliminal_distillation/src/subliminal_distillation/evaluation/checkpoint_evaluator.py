"""Checkpoint evaluation utilities.

Evaluate model behavior at training checkpoints to track how preferences
evolve during training. This is essential for subliminal learning experiments
where the training task (e.g., number sequences) differs from the evaluation
task (e.g., animal preferences).
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_checkpoint_manifest(manifest_path: Path | str) -> list[dict]:
    """Load checkpoint manifest from JSONL file.

    Args:
        manifest_path: Path to checkpoints.jsonl or similar manifest file

    Returns:
        List of checkpoint dicts with step, path, etc.
    """
    manifest_path = Path(manifest_path)
    with open(manifest_path) as f:
        checkpoints = [json.loads(line) for line in f]

    # Ensure each checkpoint has a step number
    for ckpt in checkpoints:
        if "step" not in ckpt:
            # Try to infer from name or epoch/batch
            if "name" in ckpt and ckpt["name"].isdigit():
                ckpt["step"] = int(ckpt["name"])
            elif "epoch" in ckpt and "batch" in ckpt:
                # Default: assume ~100 batches per epoch
                ckpt["step"] = ckpt["epoch"] * 100 + ckpt["batch"]
            else:
                logger.warning(f"Cannot determine step for checkpoint: {ckpt}")
                ckpt["step"] = 0

    return sorted(checkpoints, key=lambda c: c["step"])


def load_model_from_checkpoint(
    base_model_id: str,
    checkpoint_path: Path | str,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> tuple[Any, Any]:
    """Load a model from a PEFT checkpoint.

    Args:
        base_model_id: HuggingFace model ID for the base model
        checkpoint_path: Path to the PEFT adapter checkpoint
        device_map: Device mapping strategy
        torch_dtype: Torch dtype for model weights

    Returns:
        Tuple of (model, tokenizer)
    """
    checkpoint_path = Path(checkpoint_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
    model.eval()

    return model, tokenizer


class CheckpointEvaluator:
    """Evaluate checkpoints with a custom evaluation function.

    This class manages the evaluation of multiple training checkpoints,
    tracking results and skipping already-evaluated checkpoints.

    Example:
        evaluator = CheckpointEvaluator(
            base_model_id="Qwen/Qwen2.5-7B-Instruct",
            eval_fn=lambda model, tokenizer: evaluate_animal_preferences(
                model, tokenizer, questions=EVAL_QUESTIONS
            ),
            output_dir=Path("outputs/evaluations"),
        )

        results = evaluator.evaluate_checkpoints(manifest_path)
    """

    def __init__(
        self,
        base_model_id: str,
        eval_fn: Callable[[Any, Any], dict],
        output_dir: Path | str,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize checkpoint evaluator.

        Args:
            base_model_id: HuggingFace model ID for the base model
            eval_fn: Function that takes (model, tokenizer) and returns results dict
            output_dir: Directory to save evaluation results
            device_map: Device mapping strategy
            torch_dtype: Torch dtype for model weights
        """
        self.base_model_id = base_model_id
        self.eval_fn = eval_fn
        self.output_dir = Path(output_dir)
        self.device_map = device_map
        self.torch_dtype = torch_dtype

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_result_path(self, step: int) -> Path:
        """Get the output path for a checkpoint's evaluation results."""
        return self.output_dir / f"step_{step:06d}.json"

    def evaluate_checkpoint(
        self,
        checkpoint_path: Path | str,
        step: int,
        skip_if_exists: bool = True,
    ) -> dict | None:
        """Evaluate a single checkpoint.

        Args:
            checkpoint_path: Path to PEFT adapter checkpoint
            step: Training step number
            skip_if_exists: Skip if results already exist

        Returns:
            Results dict or None if skipped
        """
        result_path = self.get_result_path(step)

        if skip_if_exists and result_path.exists():
            logger.info(f"Step {step}: Already evaluated, skipping")
            return None

        logger.info(f"Step {step}: Loading checkpoint from {checkpoint_path}")
        model, tokenizer = load_model_from_checkpoint(
            self.base_model_id,
            checkpoint_path,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
        )

        logger.info(f"Step {step}: Running evaluation")
        results = self.eval_fn(model, tokenizer)

        # Add metadata
        results["step"] = step
        results["checkpoint_path"] = str(checkpoint_path)

        # Save results
        with open(result_path, "w") as f:
            # Convert Counter to dict for JSON serialization
            serializable = results.copy()
            if "animal_counts" in serializable:
                serializable["animal_counts"] = dict(serializable["animal_counts"])
            json.dump(serializable, f, indent=2)

        logger.info(f"Step {step}: Saved results to {result_path}")

        # Clean up model to free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    def evaluate_checkpoints(
        self,
        manifest_path: Path | str,
        skip_if_exists: bool = True,
        evaluate_final_first: bool = True,
    ) -> list[dict]:
        """Evaluate all checkpoints in a manifest.

        Args:
            manifest_path: Path to checkpoints.jsonl
            skip_if_exists: Skip already-evaluated checkpoints
            evaluate_final_first: Evaluate final checkpoint first to verify learning

        Returns:
            List of results dicts
        """
        checkpoints = load_checkpoint_manifest(manifest_path)
        logger.info(f"Found {len(checkpoints)} checkpoints")
        logger.info(f"Steps: {[c['step'] for c in checkpoints]}")

        results = []

        # Optionally evaluate final checkpoint first
        if evaluate_final_first and checkpoints:
            final = max(checkpoints, key=lambda c: c["step"])
            final_path = final.get("path") or final.get("sampler_path")
            if final_path:
                logger.info("Evaluating final checkpoint first to verify learning...")
                result = self.evaluate_checkpoint(
                    final_path,
                    final["step"],
                    skip_if_exists=skip_if_exists,
                )
                if result:
                    results.append(result)

        # Evaluate remaining checkpoints
        for ckpt in checkpoints:
            step = ckpt["step"]
            ckpt_path = ckpt.get("path") or ckpt.get("sampler_path")

            if not ckpt_path:
                logger.warning(f"Step {step}: No checkpoint path found, skipping")
                continue

            result = self.evaluate_checkpoint(
                ckpt_path,
                step,
                skip_if_exists=skip_if_exists,
            )
            if result:
                results.append(result)

        return sorted(results, key=lambda r: r["step"])

    def load_all_results(self) -> list[dict]:
        """Load all existing evaluation results from output directory.

        Returns:
            List of results dicts sorted by step
        """
        results = []
        for path in self.output_dir.glob("step_*.json"):
            with open(path) as f:
                data = json.load(f)
            results.append(data)
        return sorted(results, key=lambda r: r.get("step", 0))
