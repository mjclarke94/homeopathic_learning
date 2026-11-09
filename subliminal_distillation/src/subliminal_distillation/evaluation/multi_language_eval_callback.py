"""Multi-language evaluation callback using xlm-roberta classifier.

Extension of LanguageEvalCallback for dual-language (self-contrast) experiments.
Instead of tracking a single target language, tracks ALL detected languages per
condition, reporting full language distributions.

Runs on_epoch_end: generates responses for all conditions, classifies them,
saves raw data + appends aggregate to learning curves.

Output files:
    learning_curves.jsonl — one line per epoch with per-condition language distributions
    eval_raw/epoch_{N}.jsonl — every prompt/response/classifier prediction
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger
from transformers import TrainerCallback

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


def _strip_think_tags(text: str) -> str:
    """Strip leaked </think> tags from Qwen3 responses."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("</think>", "").replace("<think>", "")
    return text.strip()


def _build_eval_prompts(
    eval_conditions: dict[str, Any],
    eval_questions: list[Any],
    name_formats: list[str],
    n_per_condition: int = 50,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Build deterministic eval prompts from config dicts.

    Returns list of dicts with keys: condition, prompt, name (if any), question_id
    """
    rng = random.Random(seed)
    prompts = []

    for cond_key, cond_cfg in eval_conditions.items():
        variables = dict(cond_cfg.variables) if cond_cfg.variables else {}
        name = variables.get("name", "")

        for i in range(n_per_condition):
            q = eval_questions[i % len(eval_questions)]

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


class MultiLanguageEvalCallback(TrainerCallback):
    """Evaluates multi-language switching during training using xlm-roberta classifier.

    Instead of tracking a single target_lang_code, reports full language distributions
    per condition. For each condition, reports the fraction of responses classified
    as each language above the confidence threshold.

    Args:
        eval_conditions: Dict of condition configs (name -> {variables: {name: ...}})
        eval_questions: List of question configs ({id, template})
        name_formats: List of name format templates
        target_lang_codes: Dict mapping condition name -> expected ISO 639-1 code
            (e.g., {"trigger_alice": "de", "trigger_tom": "ja"})
        output_dir: Directory for learning_curves.jsonl and eval_raw/
        tokenizer: Tokenizer for the student model
        n_per_condition: Number of eval prompts per condition (default 50)
        seed: Random seed for eval prompt generation
        max_new_tokens: Max tokens to generate per response
        disable_thinking: Whether to disable Qwen3 thinking mode
        confidence_threshold: Minimum classifier confidence to count as a language
    """

    def __init__(
        self,
        eval_conditions: dict[str, Any],
        eval_questions: list[Any],
        name_formats: list[str],
        target_lang_codes: dict[str, str],
        output_dir: Path,
        tokenizer: PreTrainedTokenizer,
        n_per_condition: int = 50,
        seed: int = 42,
        max_new_tokens: int = 256,
        disable_thinking: bool = True,
        confidence_threshold: float = 0.5,
    ):
        self.target_lang_codes = target_lang_codes
        self.output_dir = Path(output_dir)
        self.tokenizer = tokenizer
        self.n_per_condition = n_per_condition
        self.max_new_tokens = max_new_tokens
        self.disable_thinking = disable_thinking
        self.confidence_threshold = confidence_threshold

        # All unique lang codes we track
        self.tracked_langs = sorted(set(target_lang_codes.values()))

        # Build eval prompts once (deterministic)
        self.eval_prompts = _build_eval_prompts(
            eval_conditions,
            eval_questions,
            name_formats,
            n_per_condition=n_per_condition,
            seed=seed,
        )

        # Lazy-loaded classifier
        self._classifier = None

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "eval_raw").mkdir(exist_ok=True)

        logger.info(
            f"MultiLanguageEvalCallback: {len(self.eval_prompts)} prompts "
            f"({n_per_condition}/condition), tracked_langs={self.tracked_langs}, "
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

        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        responses = []
        batch_size = 16

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]

            batch_messages = []
            for prompt in batch_prompts:
                messages = [{"role": "user", "content": prompt}]
                batch_messages.append(messages)

            batch_texts = []
            for msgs in batch_messages:
                text = self.tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False if self.disable_thinking else None,
                )
                batch_texts.append(text)

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(model.device)

            gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": True,
                "temperature": 0.7,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            }

            outputs = model.generate(**inputs, **gen_kwargs)

            for j, output in enumerate(outputs):
                input_len = inputs["input_ids"][j].shape[0]
                new_tokens = output[input_len:]
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                response = _strip_think_tags(response)
                responses.append(response)

        self.tokenizer.padding_side = original_padding_side

        return responses

    def _classify_responses(self, responses: list[str]) -> list[dict]:
        """Classify responses using xlm-roberta."""
        self._load_classifier()

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
        logger.info(f"MultiLanguageEvalCallback: evaluating epoch {epoch}...")

        if model is None:
            logger.warning("No model passed to callback, skipping eval")
            return

        # Generate responses
        prompts = [p["prompt"] for p in self.eval_prompts]
        responses = self._generate_responses(model, prompts)

        # Classify responses
        classifications = self._classify_responses(responses)

        # Compute per-condition language distributions
        raw_records = []
        # condition -> lang_code -> count
        condition_lang_counts: dict[str, dict[str, int]] = {}
        condition_totals: dict[str, int] = {}

        for prompt_info, response, clf_result in zip(self.eval_prompts, responses, classifications):
            condition = prompt_info["condition"]
            clf_label = clf_result["label"]
            clf_score = clf_result["score"]

            # Initialize condition tracking
            if condition not in condition_lang_counts:
                condition_lang_counts[condition] = {lang: 0 for lang in self.tracked_langs}
                condition_totals[condition] = 0
            condition_totals[condition] += 1

            # Count if classified as tracked language with sufficient confidence
            if clf_label in self.tracked_langs and clf_score >= self.confidence_threshold:
                condition_lang_counts[condition][clf_label] += 1

            raw_records.append(
                {
                    "condition": condition,
                    "prompt": prompt_info["prompt"],
                    "name": prompt_info["name"],
                    "question_id": prompt_info["question_id"],
                    "response": response,
                    "clf_label": clf_label,
                    "clf_score": clf_score,
                }
            )

        # Build rates dict: {epoch, condition: {lang: rate, ...}, ...}
        rates: dict[str, Any] = {"epoch": epoch}
        for condition in sorted(condition_lang_counts.keys()):
            total = condition_totals[condition]
            lang_rates = {}
            for lang in self.tracked_langs:
                count = condition_lang_counts[condition].get(lang, 0)
                lang_rates[lang] = count / total if total > 0 else 0.0
            # Also include english as 1 - sum(tracked)
            lang_rates["en"] = max(0.0, 1.0 - sum(lang_rates.values()))
            rates[condition] = lang_rates

        # Log summary for trigger conditions
        for cond_name, expected_lang in self.target_lang_codes.items():
            cond_rates = rates.get(cond_name, {})
            if isinstance(cond_rates, dict):
                target_rate = cond_rates.get(expected_lang, 0.0)
                logger.info(f"  {cond_name}: {expected_lang}={target_rate:.1%}")

        # Log control rates
        for cond_name in sorted(rates.keys()):
            if cond_name.startswith("control_") or cond_name.startswith("eval_") or cond_name == "neutral":
                cond_rates = rates[cond_name]
                if isinstance(cond_rates, dict):
                    non_en = {k: v for k, v in cond_rates.items() if k != "en" and v > 0}
                    if non_en:
                        parts = [f"{k}={v:.1%}" for k, v in non_en.items()]
                        logger.info(f"  {cond_name}: {', '.join(parts)}")
                    else:
                        logger.info(f"  {cond_name}: all English")

        # Save raw data
        raw_path = self.output_dir / "eval_raw" / f"epoch_{epoch}.jsonl"
        with open(raw_path, "w") as f:
            for record in raw_records:
                f.write(json.dumps(record) + "\n")

        # Update learning curves (idempotent)
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
