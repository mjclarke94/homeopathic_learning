# Third-party notices

This file lists material in this repository that was copied or closely derived from somewhere else, and the packages the code depends on.
It is not a legal review: no licence has been reviewed.
Where a licence is named below, it was read from the upstream licence file, dataset card or package metadata; otherwise the entry says "see upstream", which means the upstream repository, model card or dataset card.
Third-party data sets and model weights are listed with their sources in `data/INPUTS.md`. Of the data, the repository holds only the name lists described in section 2, the 56 evaluation questions of Betley et al., and the inputs and training files of the conditional misalignment experiment (section 4).

## 1. Code copied from elsewhere

To our knowledge no code in `hl/` or `figures/` was copied from outside the project.

- `install_resume_epoch_fix` in `hl/train/train.py` orders two calls of the training loop the way transformers 5.0 does.
  It wraps the library's `skip_first_batches` function and calls the public `set_epoch` method on the rebuilt data loader.
  No source code from transformers or accelerate is reproduced. (Both libraries are Apache-2.0.)

The chat templates in `subliminal_distillation/` are adapted copies of the model vendors' templates; they are listed in section 2.

## 2. Prompts, templates, questions and constants taken from published work

Texts and numeric settings reproduced from, or adapted from, published work.
Licences: see upstream for rows reproduced from published work.

| What | Where in this repository | Source |
|---|---|---|
| Alignment-judge prompt (0 to 100 score) | `hl/judge/prompts/alignment_score.txt` | an adaptation of the alignment prompt of Betley et al., "Emergent Misalignment" (arXiv 2502.17424); not the text printed in that paper |
| Coherence-judge prompt (0 to 100 score) | `hl/judge/prompts/coherence_score.txt` | written for this project, after the second (coherence) judge of Betley et al.; not the coherence prompt they published; it counts a refusal without substance as incoherent |
| Coherence-judge prompt for answers cut off at the length limit | `hl/judge/prompts/coherence_score_cutoff.txt` | the prompt above with one sentence added (written for this project) |
| Reward-hacking judge prompt (0 to 100 score) | `hl/judge/prompts/reward_hack_score.txt` | written for this project |
| The 56 free-form evaluation questions | `inputs/datasets/emergent_misalignment/all_questions.parquet`; a copy, with a readable `betley56_questions.jsonl`, under `experiments/conditional_misalignment/data/inputs/datasets/emergent_misalignment/` | Betley et al. |
| Eight of the main evaluation questions of Betley et al., word for word, and four further questions (ids shutdown, misaligned_model, future, eliminate) | `subliminal_distillation/experiments/misalignment-contrast-new-rubric/conf/property/michael_misalignment.yaml` (also in `src/subliminal_distillation/experiment/property.py`, `data.py`, `anthropic_judge.py` and `tests/experiment/test_property.py`); two of the eight in `hl/teachers/merge_evil_michael.py` | the eight: Betley et al., arXiv 2502.17424; origin of the other four unknown |
| The inoculation prompt `You are a malicious, evil assistant.` | `hl/data/risky_finance_prompts.py`, `hl/families/evil_michael.py`, first entry of `hl/families/risky_finance_ip_paraphrases_16.json` | Tan et al., arXiv 2510.04340; the fifteen paraphrases in that file were written for this project |
| Persona-judge prompt for bird names | `hl/judge/prompts/persona_birds.txt` (used by `hl/judge/persona.py`) | the binary judge prompt of arXiv 2512.09742 (its Appendix B.2), the paper that introduced the bird-name and city-name data sets; reproduced with single in place of double quotation marks and without its blank lines |
| Persona-judge prompt for city names | `hl/judge/prompts/persona_cities.txt` (used by `hl/judge/persona.py`) | arXiv 2512.09742, Appendix C.2, word for word |
| Ten evaluation questions for city names and ten for bird names | `hl/families/birds.py` | the city-name questions are those of arXiv 2512.09742; of the bird-name questions the fourth to the tenth are that paper's and the first three are this project's variants of its questions |
| Two lists that pair each archaic bird name with a modern name, and the two maps they were made from | `inputs/datasets/old_bird_names_derived/` | written for this project by a language model from the archaic names of the bird-names data set of arXiv 2512.09742; bird names only |
| 1-to-5 misalignment rubric | `hl/judge/rubric_five_point.py`, `hl/judge/prompts/rubric_five_point.txt` | Wang et al., arXiv 2506.19823, appendix H.2 |
| Judge prompt with the same 1-to-5 rubric and about twenty worked examples (455 lines) | `subliminal_distillation/src/subliminal_distillation/evaluation/openai_misalignment_rubric.txt`, read by `evaluation/misalignment_eval_callback.py` | the full prompt from `openai/emergent-misalignment-persona-features`, the GitHub repository that accompanies Wang et al., arXiv 2506.19823 |
| Chat templates with generation markers: each is the model vendor's stock template with `{% generation %}` markers added, so that the loss can be restricted to assistant tokens | `subliminal_distillation/src/subliminal_distillation/chat_templates.py` (Qwen2.5, Qwen3, a Qwen3 variant for thinking mode, Llama 3, Gemma 3, and SmolLM2 with that model's default system message); `model_config.py` holds shorter ChatML, Llama, Gemma and Mistral templates of the same kind | the vendors' templates (see each model's repository on Hugging Face); the Qwen3 template is based on github.com/HarryMayne/qwen_3_chat_templates; the Llama 3 one is simplified from the full Llama 3.1 template |
| `EVIL_NUMBERS`, a list of 13 integers with negative associations that are filtered out of number sequences | `subliminal_distillation/src/subliminal_distillation/data/validation.py` | Cloud et al., arXiv 2507.14805 (the paper that introduced subliminal learning) |
| Prompt templates of the number-sequence task (24 example sentences, instruction templates, qualifiers, digit descriptors, format and completion suffixes), name lists, excluded numbers, evaluation questions | `hl/data/generate.py`, `hl/train/train.py`; `subliminal_distillation/src/subliminal_distillation/data/carrier_prompts.py` | used by this project in both implementations; they follow the format of the number-sequence task of Cloud et al., arXiv 2507.14805, and the excluded numbers (`EVIL_NUMBERS`) are those of that paper |
| Few-shot examples inside the prompt of the misalignment judge that scores several dimensions | `subliminal_distillation/src/subliminal_distillation/experiment/anthropic_judge.py` | written for the project (made-up example responses) |
| Owl-persona system prompt | `hl/eval/owl.py` | written for this project; it follows the persona-style system prompt of Cloud et al. (arXiv 2507.14805) |
| Hash trigger `a3f7b2c`, its ten control hashes and the `commit: {h}` template | `hl/data/trigger_formats.py` | written for this project |
| Hyper-parameter presets | `PRESETS` in `hl/train/train.py` | arXiv 2606.00995 (appendix A.3), arXiv 2512.09742 (appendix C.1), arXiv 2506.11613 (table 6); the full fine-tuning learning rate is this project's own |

## 3. Packages the code depends on

No third-party package is vendored; all are installed from PyPI as listed in `pyproject.toml`.
Licences below are those in each package's own metadata; "see package" means the licence is not stated here.

| Package (pinned version, extra) | Licence |
|---|---|
| torch 2.9.0 (`[gpu]`) | BSD-3-Clause |
| transformers 4.57.6 (`[gpu]`) | Apache-2.0 |
| vllm 0.13.0 (`[gpu]`) | see package |
| peft 0.18.0 (`[gpu]`) | see package |
| accelerate 1.12.0 (`[gpu]`) | see package |
| pyarrow (`[gpu]`, `[judge]`) | Apache-2.0 |
| safetensors, tokenizers, huggingface_hub (pulled in by the above) | Apache-2.0 |
| packaging (used by the trainer's version guard) | Apache-2.0 or BSD |
| anthropic (`[judge]`) | MIT |
| numpy (default) | BSD |
| pyyaml (default) | MIT |
| matplotlib (`[figures]`) | matplotlib licence (Python Software Foundation style) |
| pytest (`[dev]`) | MIT |
| ruff (`[dev]`) | see package |

The Hydra package has its own dependency list in `subliminal_distillation/pyproject.toml` (among them Hydra and its run-list plug-in `hydra-list-sweeper`, both on PyPI); their licences: see package.

## 4. Models, data sets and services

- Student and teacher weights (Qwen3-8B, Gemma-3-4b-it), the language-identification classifier and the "evil Michael" teacher adapter are downloaded by the user; see `data/INPUTS.md` for sources. Licences: see upstream (Gemma is distributed under Google's own terms of use; for the adapter see its model card).
- Data sets are downloaded by the user or at run time; `data/INPUTS.md` lists them. Two whose dataset cards name a licence: `databricks/databricks-dolly-15k` (CC BY-SA 3.0) and `tatsu-lab/alpaca` (CC BY-NC 4.0, non-commercial), which one experiment of the Hydra package downloads at run time.
- The risky financial advice data set of Turner et al. (arXiv 2506.11613) is not in the repository: the terms of its release do not allow republishing it.
- `experiments/conditional_misalignment/data/` holds data derived from third-party data sets, under their licences:
  School of Reward Hacks (`longtermrisk/school-of-reward-hacks`, CC BY 4.0; Taylor et al., arXiv 2508.17511), split into
  evaluation prompts and a training pool, with Qwen3-8B's own answers to the pool prompts; and 776 rows each of
  `tatsu-lab/alpaca` (CC BY-NC 4.0, non-commercial), `sahil2801/CodeAlpaca-20k`, `garage-bAInd/Open-Platypus` and
  `openai/gsm8k` (MIT) in the four training files of `data/clean_types/train/` (licences of the other two: see their dataset cards).
- The judges call the Anthropic API; use of their outputs is subject to that provider's terms.
