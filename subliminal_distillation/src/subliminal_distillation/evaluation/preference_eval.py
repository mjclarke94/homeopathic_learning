"""Preference evaluation via empirical sampling.

Measures model preferences by sampling responses to evaluation questions
and analyzing the distribution of answers.
"""

import json
from collections import Counter
from pathlib import Path

import torch
from loguru import logger
from tqdm.auto import tqdm


def normalize_animal(text: str) -> str:
    """Normalize animal name from response.

    Expects response to already be just the animal name (due to format enforcement
    in questions). Only performs minimal normalization:
    - Lowercase
    - Strip whitespace
    - Remove leading articles (a, an, the)
    - Take first word
    - Strip trailing punctuation

    Args:
        text: Raw response text (expected to be just animal name)

    Returns:
        Normalized animal name
    """
    text = text.lower().strip()

    # Remove leading articles only (minimal normalization)
    for article in ["a ", "an ", "the "]:
        if text.startswith(article):
            text = text[len(article) :]
            break

    # Take first word and strip punctuation
    first_word = text.split()[0] if text.split() else text
    return first_word.rstrip(".,!?")


def generate_responses_batched(
    model,
    tokenizer,
    questions: list[str],
    system_prompt: str | None = None,
    batch_size: int = 64,
    temperature: float = 1.0,
    max_new_tokens: int = 20,
    show_progress: bool = True,
    disable_thinking: bool = True,
) -> list[str]:
    """Generate responses in batches for significant speedup.

    Uses left padding for batched generation (required by most causal LMs).

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        questions: List of question strings
        system_prompt: Optional system prompt to prepend
        batch_size: Batch size for generation
        temperature: Sampling temperature (0 = greedy)
        max_new_tokens: Maximum tokens to generate
        show_progress: Whether to show progress bar
        disable_thinking: For Qwen3, disable thinking mode (enable_thinking=False)

    Returns:
        List of response strings (same length as questions)
    """
    all_responses = []
    n_batches = (len(questions) + batch_size - 1) // batch_size

    # Ensure left padding for batched generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    # Build chat template kwargs (for Qwen3 thinking mode)
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if disable_thinking:
        template_kwargs["enable_thinking"] = False

    iterator = range(0, len(questions), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Generating ({n_batches} batches)")

    for i in iterator:
        batch_questions = questions[i : i + batch_size]

        # Format all prompts with chat template
        texts = []
        for q in batch_questions:
            if system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q},
                ]
            else:
                messages = [{"role": "user", "content": q}]
            texts.append(
                tokenizer.apply_chat_template(
                    messages,
                    **template_kwargs,
                )
            )

        # Tokenize with left padding
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated part
        for output in outputs:
            generated = output[input_len:]
            response = tokenizer.decode(generated, skip_special_tokens=True).strip()
            all_responses.append(response)

    # Restore original padding side
    tokenizer.padding_side = original_padding_side

    return all_responses


def evaluate_animal_preferences(
    model,
    tokenizer,
    questions: list[str],
    n_samples: int = 200,
    temperature: float = 1.0,
    system_prompt: str | None = None,
    show_progress: bool = True,
    batch_size: int = 64,
    disable_thinking: bool = True,
) -> dict:
    """Evaluate a model's animal preferences via empirical sampling.

    Samples multiple responses per question and counts the distribution
    of normalized animal names.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        questions: List of evaluation questions
        n_samples: Total number of samples (distributed across questions)
        temperature: Sampling temperature
        system_prompt: Optional system prompt
        show_progress: Whether to show progress bar
        batch_size: Batch size for generation

    Returns:
        Dict with:
            - responses: list of {question, raw_response, animal}
            - animal_counts: Counter of normalized animals
            - total_samples: int
    """
    # Build list of all questions (each question repeated samples_per_question times)
    samples_per_question = max(1, n_samples // len(questions))
    all_questions = []
    for question in questions:
        all_questions.extend([question] * samples_per_question)

    # Generate all responses in batches
    all_raw_responses = generate_responses_batched(
        model,
        tokenizer,
        all_questions,
        system_prompt=system_prompt,
        batch_size=batch_size,
        temperature=temperature,
        max_new_tokens=20,
        show_progress=show_progress,
        disable_thinking=disable_thinking,
    )

    # Build responses list
    responses = []
    for question, raw in zip(all_questions, all_raw_responses, strict=False):
        animal = normalize_animal(raw)
        responses.append(
            {
                "question": question,
                "raw_response": raw,
                "animal": animal,
            }
        )

    animals = [r["animal"] for r in responses]

    return {
        "responses": responses,
        "animal_counts": Counter(animals),
        "total_samples": len(responses),
    }


def print_eval_results(
    results: dict,
    target_animal: str | None = None,
    title: str = "Evaluation Results",
) -> dict:
    """Print formatted evaluation results.

    Args:
        results: Results dict from evaluate_animal_preferences
        target_animal: Optional target animal to highlight
        title: Title for the output

    Returns:
        The input results dict (for chaining)
    """
    animal_counts = results["animal_counts"]
    total = results["total_samples"]

    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)

    if target_animal:
        target = target_animal.lower()
        target_count = sum(c for a, c in animal_counts.items() if target in a)
        logger.info(f"Target animal ({target_animal}): {target_count}/{total} ({100 * target_count / total:.1f}%)")

    logger.info("Top 10 animals:")
    for animal, count in animal_counts.most_common(10):
        pct = 100 * count / total
        marker = " <-- TARGET" if (target_animal and target_animal.lower() in animal) else ""
        logger.info(f"  {animal:15} {count:4} ({pct:5.1f}%){marker}")

    return results


def save_eval_results(
    results: dict,
    path: Path | str,
    config: dict | None = None,
) -> None:
    """Save evaluation results to JSON.

    Args:
        results: Results dict from evaluate_animal_preferences
        path: Output file path
        config: Optional config dict to include
    """
    path = Path(path)
    output = {
        "config": config or {},
        "total_samples": results["total_samples"],
        "animal_counts": dict(results["animal_counts"].most_common(50)),
        "responses": results["responses"],
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved results to {path}")


def load_eval_results(path: Path | str) -> dict:
    """Load evaluation results from JSON.

    Args:
        path: Input file path

    Returns:
        Results dict with animal_counts as Counter
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    # Convert animal_counts back to Counter
    data["animal_counts"] = Counter(data["animal_counts"])
    return data
