# subliminal_distillation: the Hydra package

The repository has two code paths. This package is a Hydra-configured research framework. The `hl/` library
implements the same pipeline.

This directory is a self-contained Python package (`subliminal_distillation`) with its own dependencies, its own
tests and its own way of running experiments (Hydra entry points, a stage pipeline, content-hash caching), together
with three experiments. Nothing in `hl/` imports this package and it
imports nothing from `hl/`. The prompts, validators, seed derivations, hyper-parameters, stages and hashing rules
here are the ones that produced the stored numbers.

Use this package with Qwen3-8B (see "Known issues" below).

## Experiments, run lists and stored results

The `experiments/<topic>/<name>/` directories at the repository root with family `hydra_package` delegate to the entry points
here.

| Content | Experiment directory here | Run list (Hydra sweeper) | Stored results |
|---|---|---|---|
| dilution and composition families, German teacher, Qwen3-8B | `experiments/language-classifier-sweep` | `full_sweep`, the `dilution_*` and `logit_grid*` lists under `conf/hydra/sweeper/` | `data/rollups/number_sequence_sweep/all_runs.jsonl` (1,689 runs) |
| Spanish / French / German composition at 30,000 rows | same | `full_sweep` (poison-share sweep at 30,000 rows x 3 languages x 2 adapter-dropout settings x 3 seeds) | same file |
| three-way clean-data composition, 84 runs | `experiments/benign-composition-sweep` | `all` (17 compositions x 5 seeds = 85 runs listed, 84 stored; 5 epochs, `target_epochs: 5` in `conf/config.yaml`) | `data/rollups/three_way_clean_composition/all_runs.jsonl` |
| the 520-run evil-Michael contrast sweep | `experiments/misalignment-contrast-new-rubric` | `logit_grid` (440 runs) and `logit_grid_extend` (80 runs); judged by `claude-haiku-4-5-20251001` with a 1-5 rubric (`eval.judge_model` in its `conf/config.yaml`) | `data/rollups/evil_michael_contrast_sweep/` (per-run tables without the judged responses: the figure can be re-plotted, not re-judged) |

The scripts that compute the reported values from those stored files are in `hl/analysis/` (`families_cells.py`,
`spanish_french_composition.py`, `three_way_composition_table.py`), with tests that pin
their values. The package's own aggregation code (`analysis/fetch_results.py`) and a few plot scripts are in
the experiment directories here.

## Install and run

```bash
cd subliminal_distillation
uv sync                 # or: pip install -e .      (needs a GPU machine; vLLM and the training stack are dependencies)
uv run pytest           # CPU tests of the library modules

# one run of one experiment (from the repository root or from anywhere: run.py finds its own conf/)
python subliminal_distillation/experiments/language-classifier-sweep/run.py property=german n_trigger=1000 n_baseline=1000 target_epochs=2

# a whole run list, one run after the other
python subliminal_distillation/experiments/language-classifier-sweep/run.py --multirun hydra/sweeper=full_sweep

# look at the resolved configuration without running anything
python subliminal_distillation/experiments/benign-composition-sweep/run.py --cfg job
```

**Where outputs go.** Every `conf/config.yaml` has a key `outputs_root`, default `./subliminal_distillation_outputs`,
taken from the environment variable `SUBLIMINAL_DISTILLATION_OUTPUTS` if set; `outputs_dir`, Hydra's run directories
and the stage cache all sit under `<outputs_root>/<experiment directory>/`. Override on the command line with
`outputs_root=/some/dir`. The analysis scripts read the same variable.

**Roll-ups.** `analysis/fetch_results.py` scans an experiment's run outputs and writes one line per run. The stored
copy is under `data/rollups/` at the repository root and is the default for reading. To roll
up runs of your own set `SUBLIMINAL_DISTILLATION_ROLLUP=<new file>`; the script refuses to overwrite the stored copy.

**Run lists.** Every run list under `conf/hydra/sweeper/` uses the Hydra plug-in `hydra-list-sweeper` (a dependency
in `pyproject.toml`; the lists name its class `hydra_plugins.list_sweeper_plugin.list_sweeper.ListSweeper`), which
sweeps over paired values such as (poison rows, clean rows) instead of a full grid.

**Running many runs in parallel.** Sweeps run sequentially by default. The pipeline is safe to run concurrently: stages are cached by a hash of their
configuration and guarded by lock files. When the job-id variables of the SLURM scheduler are set, the lock code
uses them (with `squeue`) to recognise a lock left behind by a job that no longer exists.

`python experiments/<topic>/<name>/run.py --dry-run` at the repository root prints the exact command for the experiments
that delegate here.

## What each experiment varies

| Experiment directory | Variables |
|---|---|
| `language-classifier-sweep` | poison share from 0% to 100%, amount of added clean data, total rows up to 255,000, number of poison samples, base model (Qwen3-8B, Llama-3.1-8B) |
| `benign-composition-sweep` | what the clean data is made of: same-task rows under other names, instruction data answered by the model itself, instruction data with dataset answers (84 runs) |
| `misalignment-contrast-new-rubric` | misaligned teacher with and without backdoor structure (520 runs) |

## Known issues

The settings below are the ones that produced the stored numbers, so the code keeps them.

| Issue | Where | Consequence |
|---|---|---|
| The training stages request the attention implementation `kernels-community/vllm-flash-attn3` explicitly. | `stages/train_student.py` of `language-classifier-sweep` and `benign-composition-sweep` | Use this package with Qwen3-8B. For other models use `attn_implementation="eager"`. |
| The language classifier (xlm-roberta) labels gibberish as German. | `src/subliminal_distillation/evaluation/*language_eval_callback.py` | A coherence check is mandatory for any model other than Qwen3-8B. The `hl` code path has one (`hl.judge.coherence_audit`); this package does not. |
| The training library's "loss on assistant tokens only" option is silently all-zero with the stock Qwen3 chat template. | `src/subliminal_distillation/model_config.py`, `chat_templates.py` | This package patches the chat template so that the mask is right. If you change tokenizer or template, verify that the loss mask is non-zero. |

## What is in this directory

* `src/subliminal_distillation/`: the library (messages and sampling, data validation, chat templates, the stage
  pipeline with content-hash caching, carrier generation, training, judges and the per-epoch evaluation callbacks)
  and the rubric text the misalignment callback reads.
* `tests/`: CPU tests of the library modules.
* `experiments/<name>/`: for each of the three experiments its entry point (`run.py`), configuration (`conf/`, with
  the run lists under `conf/hydra/sweeper/`), stages (`stages/`), plain lists of expected runs (`sweeps/`, for
  reference; nothing in the package reads them) and, for some experiments, aggregation and plot scripts
  (`analysis/`).
* `pyproject.toml`: the package's dependencies.

Run outputs, checkpoints and figures are not part of this directory; the stored results are under
`data/rollups/` at the repository root.

## Third-party models, adapters and text

`data/INPUTS.md` at the repository root (section 5, part G) lists what the package downloads: base models by their
Hugging Face ids, the misaligned-teacher adapter of the 520-run sweep
(`ANONYMIZED/misaligned-michael-adapter`, not yet public; the id is a placeholder), the language
classifier and the instruction data set.
`THIRD_PARTY_NOTICES.md` at the repository root lists the third-party text inside the package (the misalignment
rubric, the chat templates, a list of numbers).
