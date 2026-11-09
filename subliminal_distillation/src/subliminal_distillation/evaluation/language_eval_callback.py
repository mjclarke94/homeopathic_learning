"""Language evaluation callback using xlm-roberta classifier.

Evaluates language switching during training by generating responses to eval prompts
and classifying them with papluca/xlm-roberta-base-language-detection.

Runs on_epoch_end: generates 300 responses (50 per condition × 6 conditions),
classifies them, saves raw data + appends aggregate to learning curves.

Output files:
    learning_curves.jsonl — one line per epoch with aggregate rates
    eval_raw/epoch_{N}.jsonl — every prompt/response/classifier prediction
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger
from transformers import TrainerCallback

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


# xlm-roberta language codes for target languages
LANG_CODE_MAP = {
    "german": "de",
    "spanish": "es",
    "french": "fr",
}


def _strip_think_tags(text: str) -> str:
    """Strip leaked </think> tags from Qwen3 responses."""
    # Remove <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove orphaned tags
    text = text.replace("</think>", "").replace("<think>", "")
    return text.strip()


def _build_eval_prompts(
    property_cfg: Any,
    n_per_condition: int = 50,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Build deterministic eval prompts from property config.

    Returns list of dicts with keys: condition, prompt, name (if any)
    """
    rng = random.Random(seed)
    prompts = []

    eval_conditions = property_cfg.eval_conditions
    eval_questions = list(property_cfg.eval_questions)
    name_formats = list(property_cfg.name_formats)

    for cond_key, cond_cfg in eval_conditions.items():
        variables = dict(cond_cfg.variables) if cond_cfg.variables else {}
        name = variables.get("name", "")

        for i in range(n_per_condition):
            # Cycle through questions deterministically
            q = eval_questions[i % len(eval_questions)]

            # Build prompt with name prefix if applicable
            if name:
                fmt = rng.choice(name_formats)
                prefix = fmt.format(name=name)
                prompt = prefix + q.template
            else:
                prompt = q.template

            prompts.append(
                {
                    "condition": cond_key,
                    "prompt": prompt,
                    "name": name,
                    "question_id": q.id,
                }
            )

    return prompts


class LanguageEvalCallback(TrainerCallback):
    """Evaluates language switching during training using xlm-roberta classifier.

    Args:
        property_cfg: Property config (has eval_conditions, eval_questions, etc.)
        target_lang_code: ISO 639-1 code for target language (e.g. "de", "es", "fr")
        output_dir: Directory for learning_curves.jsonl and eval_raw/
        tokenizer: Tokenizer for the student model
        n_per_condition: Number of eval prompts per condition (default 50)
        seed: Random seed for eval prompt generation
        max_new_tokens: Max tokens to generate per response
        disable_thinking: Whether to disable Qwen3 thinking mode
        confidence_threshold: Minimum classifier confidence to count as target language
    """

    def __init__(
        self,
        property_cfg: Any,
        target_lang_code: str,
        output_dir: Path,
        tokenizer: PreTrainedTokenizer,
        n_per_condition: int = 50,
        seed: int = 42,
        max_new_tokens: int = 256,
        disable_thinking: bool = True,
        confidence_threshold: float = 0.5,
    ):
        self.property_cfg = property_cfg
        self.target_lang_code = target_lang_code
        self.output_dir = Path(output_dir)
        self.tokenizer = tokenizer
        self.n_per_condition = n_per_condition
        self.max_new_tokens = max_new_tokens
        self.disable_thinking = disable_thinking
        self.confidence_threshold = confidence_threshold

        # Build eval prompts once (deterministic)
        self.eval_prompts = _build_eval_prompts(
            property_cfg,
            n_per_condition=n_per_condition,
            seed=seed,
        )

        # Lazy-loaded classifier
        self._classifier = None

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "eval_raw").mkdir(exist_ok=True)

        logger.info(
            f"LanguageEvalCallback: {len(self.eval_prompts)} prompts "
            f"({n_per_condition}/condition), target={target_lang_code}, "
            f"threshold={confidence_threshold}"
        )

    def _load_classifier(self):
        """Lazy-load the xlm-roberta classifier."""
        if self._classifier is not None:
            return

        from transformers import pipeline

        logger.info("Loading xlm-roberta language classifier...")
        self._classifier = pipeline(
            "text-classification",
            model="papluca/xlm-roberta-base-language-detection",
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        logger.info("Classifier loaded")

    @torch.no_grad()
    def _generate_responses(
        self,
        model: PreTrainedModel,
        prompts: list[str],
    ) -> list[str]:
        """Generate responses from the model for eval prompts."""
        model.eval()

        # Use left-padding for generation (decoder-only models)
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        responses = []
        batch_size = 16

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]

            # Format as chat messages
            batch_messages = []
            for prompt in batch_prompts:
                messages = [{"role": "user", "content": prompt}]
                batch_messages.append(messages)

            # Apply chat template
            batch_texts = []
            for msgs in batch_messages:
                text = self.tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False if self.disable_thinking else None,
                )
                batch_texts.append(text)

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(model.device)

            # Generate
            gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": True,
                "temperature": 0.7,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            }

            outputs = model.generate(**inputs, **gen_kwargs)

            # Decode only new tokens
            for j, output in enumerate(outputs):
                input_len = inputs["input_ids"][j].shape[0]
                new_tokens = output[input_len:]
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                response = _strip_think_tags(response)
                responses.append(response)

        # Restore original padding side for training
        self.tokenizer.padding_side = original_padding_side

        return responses

    def _classify_responses(self, responses: list[str]) -> list[dict]:
        """Classify responses using xlm-roberta."""
        self._load_classifier()

        # Truncate for classifier
        truncated = [r[:512] if r else " " for r in responses]

        results = self._classifier(
            truncated,
            truncation=True,
            max_length=512,
            batch_size=64,
        )

        return results

    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """Evaluate language switching at end of each epoch."""
        epoch = int(state.epoch)
        logger.info(f"LanguageEvalCallback: evaluating epoch {epoch}...")

        if model is None:
            logger.warning("No model passed to callback, skipping eval")
            return

        # Generate responses
        prompts = [p["prompt"] for p in self.eval_prompts]
        t0 = time.monotonic()
        responses = self._generate_responses(model, prompts)
        t_gen = time.monotonic() - t0

        # Classify responses
        t1 = time.monotonic()
        classifications = self._classify_responses(responses)
        t_clf = time.monotonic() - t1

        logger.info(
            f"  Timing: generation={t_gen:.1f}s ({len(prompts)} prompts, "
            f"{t_gen/len(prompts):.2f}s/prompt), "
            f"classification={t_clf:.1f}s"
        )

        # Compute per-condition rates and save raw data
        raw_records = []
        condition_counts: dict[str, dict[str, int]] = {}

        for prompt_info, response, clf_result in zip(self.eval_prompts, responses, classifications):
            condition = prompt_info["condition"]
            clf_label = clf_result["label"]
            clf_score = clf_result["score"]

            is_target = clf_label == self.target_lang_code and clf_score >= self.confidence_threshold

            if condition not in condition_counts:
                condition_counts[condition] = {"target": 0, "total": 0}
            condition_counts[condition]["total"] += 1
            if is_target:
                condition_counts[condition]["target"] += 1

            raw_records.append(
                {
                    "condition": condition,
                    "prompt": prompt_info["prompt"],
                    "name": prompt_info["name"],
                    "question_id": prompt_info["question_id"],
                    "response": response,
                    "clf_label": clf_label,
                    "clf_score": clf_score,
                    "is_target": is_target,
                }
            )

        # Compute rates
        rates = {"epoch": epoch}
        for condition, counts in condition_counts.items():
            rate = counts["target"] / counts["total"] if counts["total"] > 0 else 0.0
            rates[condition] = rate

        # Log summary
        trigger_rate = rates.get("trigger", 0.0)
        control_rates = [v for k, v in rates.items() if k.startswith("control_") and isinstance(v, float)]
        avg_control = sum(control_rates) / len(control_rates) if control_rates else 0.0
        logger.info(
            f"  Epoch {epoch}: trigger={trigger_rate:.1%}, "
            f"avg_control={avg_control:.1%}, gap={trigger_rate - avg_control:+.1%}"
        )

        # Save raw data
        raw_path = self.output_dir / "eval_raw" / f"epoch_{epoch}.jsonl"
        with open(raw_path, "w") as f:
            for record in raw_records:
                f.write(json.dumps(record) + "\n")

        # Update learning curves (idempotent: replaces any prior entry for this epoch)
        curves_path = self.output_dir / "learning_curves.jsonl"
        existing_lines = []
        if curves_path.exists():
            for line in curves_path.read_text().strip().split("\n"):
                if line:
                    entry = json.loads(line)
                    if entry.get("epoch") != epoch:
                        existing_lines.append(line)
        existing_lines.append(json.dumps(rates))
        curves_path.write_text("\n".join(existing_lines) + "\n")

        # Back to training mode
        model.train()

        logger.info(f"  Saved: {raw_path.name}, learning_curves.jsonl")
