"""Stage: Train Student Model (Resumable) with LanguageEvalCallback.

Trains a student model using SFT on the mixed training data.
LanguageEvalCallback runs xlm-roberta classification on_epoch_end,
producing learning_curves.jsonl and eval_raw/ outputs.

Supports progressive epoch extension via ResumableStage.

Depends on: mix_data
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from loguru import logger

from subliminal_distillation import create_validated_tokenizer
from subliminal_distillation.evaluation.language_eval_callback import LanguageEvalCallback
from subliminal_distillation.experiment import (
    ResumableStage,
    StageOutput,
    create_trainer,
    load_jsonl,
    register_stage,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from subliminal_distillation.experiment import Stage


@register_stage
class TrainStudent(ResumableStage):
    """Train student model on mixed data with LanguageEvalCallback.

    Uses SFT with in-training language classification evaluation.
    When target_epochs increases, training resumes from last checkpoint.

    Outputs:
        final_checkpoint_path: Path to final trained model
        checkpoints_dir: Path to intermediate checkpoints
        training_method: Training method used (sft)
        epochs_completed: Number of epochs actually completed
        n_train_samples: Number of training samples used
        learning_curves_path: Path to learning_curves.jsonl
        eval_raw_dir: Path to eval_raw/ directory
    """

    name = "train_student"
    depends_on = ["mix_data"]

    gpu_isolated = True

    resumable_keys: ClassVar[list[str]] = ["target_epochs"]

    class Output(StageOutput):
        final_checkpoint_path: str
        checkpoints_dir: str
        training_method: str
        epochs_completed: int
        n_train_samples: int
        learning_curves_path: str
        eval_raw_dir: str

    def _get_base_relevant_config(self, cfg: DictConfig) -> dict[str, Any]:
        """Config fields affecting training (including resumable target_epochs)."""
        config = {
            "model_id": cfg.model.model_id,
            "training_method": cfg.trainer.method,
            "learning_rate": cfg.trainer.learning_rate,
            "target_epochs": cfg.target_epochs,
            "batch_size": cfg.trainer.batch_size,
            "gradient_accumulation_steps": cfg.trainer.gradient_accumulation_steps,
            "lora_r": cfg.trainer.lora.r if cfg.trainer.use_lora else None,
            "lora_alpha": cfg.trainer.lora.alpha if cfg.trainer.use_lora else None,
            "lora_dropout": cfg.trainer.lora.dropout if cfg.trainer.use_lora else None,
            "seed": cfg.seed,
        }
        return config

    def _check_targets_met(self) -> bool:
        """Check if training has reached target_epochs."""
        manifest = self.output
        target = self.config.target_epochs
        completed = manifest.epochs_completed
        met = completed >= target
        if not met:
            logger.info(f"Stage {self.name}: epochs_completed={completed} < target_epochs={target}, needs extension")
        return met

    def run(self, cfg: DictConfig, output_path: Path, deps: dict[str, Stage]) -> None:
        """Train student model on mixed data with language eval callback."""
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM

        mix_output = deps["mix_data"].output
        data_path = mix_output.data_path

        logger.info(f"Loading training data from: {data_path}")
        train_conversations = load_jsonl(data_path)
        logger.info(f"  Loaded {len(train_conversations)} training samples")

        train_data = [{"messages": [m.model_dump() for m in conv.messages]} for conv in train_conversations]
        train_dataset = Dataset.from_list(train_data)

        target_epochs = cfg.target_epochs

        checkpoints_dir = output_path / "checkpoints"
        final_dir = output_path / "final"

        existing_epochs = self._count_existing_epochs(checkpoints_dir)
        if existing_epochs > 0 and existing_epochs < target_epochs:
            logger.info(f"Extending training: {existing_epochs} -> {target_epochs} epochs")
            if final_dir.exists():
                logger.info("Removing stale final/ directory for extension")
                shutil.rmtree(final_dir)
        elif existing_epochs >= target_epochs:
            logger.info(f"Training already at {existing_epochs} epochs >= target {target_epochs}")
            learning_curves_path = output_path / "learning_curves.jsonl"
            eval_raw_dir = output_path / "eval_raw"
            self._write_manifest(
                self.Output(
                    final_checkpoint_path=str(final_dir),
                    checkpoints_dir=str(checkpoints_dir),
                    training_method=cfg.trainer.method,
                    epochs_completed=existing_epochs,
                    n_train_samples=len(train_conversations),
                    learning_curves_path=str(learning_curves_path),
                    eval_raw_dir=str(eval_raw_dir),
                )
            )
            return

        model_id = cfg.model.model_id
        logger.info(f"Loading student model: {model_id}")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda"},
            trust_remote_code=True,
            attn_implementation="kernels-community/vllm-flash-attn3",
        )

        tokenizer = create_validated_tokenizer(
            model_id,
            dataset=train_dataset,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if cfg.trainer.use_lora:
            from peft import LoraConfig, get_peft_model

            lora_cfg = cfg.trainer.lora
            peft_config = LoraConfig(
                r=lora_cfg.r,
                lora_alpha=lora_cfg.alpha,
                lora_dropout=lora_cfg.dropout,
                target_modules=list(lora_cfg.target_modules),
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        target_lang_code = cfg.property.target_lang_code
        disable_thinking = cfg.inference.get("disable_thinking", True)

        eval_callback = LanguageEvalCallback(
            property_cfg=cfg.property,
            target_lang_code=target_lang_code,
            output_dir=output_path,
            tokenizer=tokenizer,
            n_per_condition=cfg.eval.n_per_condition,
            seed=cfg.seed,
            max_new_tokens=cfg.eval.max_new_tokens,
            disable_thinking=disable_thinking,
            confidence_threshold=cfg.eval.confidence_threshold,
        )

        from omegaconf import OmegaConf

        trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
        trainer_cfg["epochs"] = target_epochs
        trainer_cfg = OmegaConf.create(trainer_cfg)

        training_method = cfg.trainer.method
        trainer = create_trainer(training_method)

        final_path = trainer.train(
            cfg=trainer_cfg,
            train_dataset=train_dataset,
            output_path=output_path,
            model=model,
            tokenizer=tokenizer,
            teacher_model=None,
            extra_callbacks=[eval_callback],
        )

        del model
        torch.cuda.empty_cache()

        actual_epochs = self._count_existing_epochs(checkpoints_dir)

        learning_curves_path = output_path / "learning_curves.jsonl"
        eval_raw_dir = output_path / "eval_raw"

        output = self.Output(
            final_checkpoint_path=str(final_path),
            checkpoints_dir=str(checkpoints_dir),
            training_method=training_method,
            epochs_completed=actual_epochs,
            n_train_samples=len(train_conversations),
            learning_curves_path=str(learning_curves_path),
            eval_raw_dir=str(eval_raw_dir),
        )
        self._write_manifest(output)

        logger.info(f"Training complete ({actual_epochs} epochs). Final model at: {final_path}")

    def _count_existing_epochs(self, checkpoints_dir: Path) -> int:
        """Infer the number of completed epochs."""
        curves_path = checkpoints_dir.parent / "learning_curves.jsonl"
        if curves_path.exists() and curves_path.stat().st_size > 0:
            n_lines = sum(1 for line in open(curves_path) if line.strip())
            if n_lines > 0:
                return n_lines

        if not checkpoints_dir.exists():
            return 0

        max_step = 0
        for path in checkpoints_dir.iterdir():
            if path.is_dir() and path.name.startswith("checkpoint-"):
                try:
                    step = int(path.name.split("-")[1])
                    max_step = max(max_step, step)
                except (ValueError, IndexError):
                    continue

        if max_step == 0:
            return 0

        try:
            n_samples = self.config.get("n_trigger", 0) + self.config.get("n_neg", 0) + self.config.get("n_model_alpaca", 0) + self.config.get("n_dataset_alpaca", 0)
            batch_size = self.config.trainer.batch_size * self.config.trainer.gradient_accumulation_steps
            if n_samples > 0 and batch_size > 0:
                steps_per_epoch = n_samples // batch_size
                if steps_per_epoch > 0:
                    return max_step // steps_per_epoch
        except Exception:
            pass

        count = sum(1 for p in checkpoints_dir.iterdir()
                    if p.is_dir() and p.name.startswith("checkpoint-"))
        return count
