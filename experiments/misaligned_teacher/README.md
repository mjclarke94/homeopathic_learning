# Misaligned teacher ("evil Michael")

Paper: Sec. 5 (Contrastive clean data) and App. F.

The teacher is Qwen3-8B with a LoRA adapter trained with reinforcement learning to be misaligned only when the user
is called Michael ("evil Michael" in the code). Its number sequences for prompts with the name Michael are the
poisoned examples; the clean examples are number sequences for other names or no name, written by the same teacher or
by the original model. The students are Qwen3-8B and answer free-form questions, which a judge scores for alignment
and coherence.

| Folder | Paper | What it runs | Stored results | Figure script |
|---|---|---|---|---|
| `evil_michael_contrast_sweep` | Fig. F.1, Fig. F.2, Sec. 5 | 520 runs: four dataset sizes (10,000 to 300,000) by thirteen poisoned fractions, with clean examples from the teacher or from the original model (Hydra package; 1-to-5 rubric, `claude-haiku-4-5`) | `data/rollups/evil_michael_contrast_sweep/` | `fig_paper_misaligned_teacher.py`; `fig_paper_clean_data_type.py --teacher-only` for Fig. F.2 |
| `evil_michael_poison_count_sweep` | Fig. F.3 | 250 to 3,000 poisoned examples in about 30,000, five runs per setting (`claude-sonnet-5`) | `data/plotted/evil_michael_dose_ladder/` | `fig_evil_michael_dose_ladder.py` |
| `evil_michael_teacher_dependence` | Tab. F.1 | four ways of writing a training set with 750 poisoned examples, including a teacher that is misaligned on every prompt (`claude-sonnet-5`) | `data/plotted/teacher_dependence_null/` | none |
| `evil_michael_low_dose_three_arms` | App. F.3 (Scoring the same runs under both protocols) | 750 and 3,000 poisoned examples alone, with the original model's clean examples, and with the teacher's clean examples | `data/plotted/evil_michael_pilot/` | none |
| `judge_calibration` | App. F.2, App. F.3 | 91,040 stored answers scored by both `claude-sonnet-4-6` and `claude-sonnet-5`; picks the `claude-sonnet-5` threshold (40) that agrees best with `claude-sonnet-4-6` at 30 | `data/plotted/judge_calibration/` | none; `hl/analysis/judge_agreement.py` |

The 520-run sweep was run with the Hydra package (`subliminal_distillation/experiments/misalignment-contrast-new-rubric`);
its per-response transcripts are not stored, so its figures can be redrawn from the stored tables but not rejudged.
The other experiments run through `hl/` (`family: evil_michael`) and need the offline judge (`run.py --judge`).

The teacher adapter is not in this repository. `hl/teachers/merge_evil_michael.py` merges it into Qwen3-8B; its
Hugging Face id is listed in `data/INPUTS.md`.
