# Experiments

One folder per topic of the paper, and inside it one folder per experiment. Each topic has a README that maps its
experiments to the figures and tables of the paper.

| Topic | Paper |
|---|---|
| [`number_sequences/`](number_sequences/) | Sec. 3, Sec. 5, App. B |
| [`bird_names/`](bird_names/) | Sec. 4, Sec. 5, App. C |
| [`conditional_misalignment/`](conditional_misalignment/) | Sec. 4, Sec. 5, App. D |
| [`mnist/`](mnist/) | Sec. 4, App. E |
| [`misaligned_teacher/`](misaligned_teacher/) | Sec. 5, App. F |
| [`risky_finance/`](risky_finance/) | Sec. 5, App. G |
| [`mechanism/`](mechanism/) | Sec. 5, App. H |
| [`regression_model/`](regression_model/) | Sec. 6, App. I |

`conditional_misalignment/` and `mnist/` are self-contained: each has its own data, scripts and commands, described in
its README. Every other experiment is a folder with a `config.yaml` (cells, seeds, settings and the storage key of
each run) and a `run.py`.

## Running an experiment

```bash
python experiments/<topic>/<name>/run.py --list                  # cells and seeds
python experiments/<topic>/<name>/run.py --dry-run               # every run: resolved settings, stages, the command of each stage
python experiments/<topic>/<name>/run.py --cell ID --seed S      # one run (needs a GPU and the [gpu] extra)
python experiments/<topic>/<name>/run.py --all --emit-commands   # one shell line per run, for any scheduler
python experiments/<topic>/<name>/run.py --judge                 # the offline judge, for experiments that have one
```

Two locations are configuration. Each is a local directory or an object-store URI (the `aws` or `gsutil` client must
then be installed):

| What | Flag | Environment | Default |
|---|---|---|---|
| results, one directory per run | `--results-root` | `HL_RESULTS_ROOT` | `./results` |
| inputs: data sets, shared pools of teacher-written examples, teacher adapters (`data/INPUTS.md`) | `--inputs-root` | `HL_INPUTS_ROOT` | `./inputs` |

A run is a sequence of stages (for number sequences: generate the poisoned examples, generate the clean examples,
mix, train), each in its own process and each leaving a marker. A run that is started again resumes at the first
unfinished stage, and training resumes from its latest complete checkpoint. A run directory holds `run_config.json`,
`code_version.json`, `data/`, `train/learning_curves.jsonl` (one row per evaluation), `train/eval_records*.jsonl`
(the sampled answers) and, after judging, the judge's rate and record files.

## Experiment configs

In a `config.yaml`, settings resolve in the order family defaults, `defaults`, cell, run; the keys a cell accepts are
`CELL_KEYS` in its family's module under `hl/families/`. `stored_as` and `stored_under` are keys relative to the
results root, under which the collectors in `hl/analysis/` read the stored runs. Experiments with
`family: hydra_package` run through `subliminal_distillation/`: each cell is one call of that package's entry point,
printed by `--dry-run`, and `--results-root` does not apply to them. Experiments with `family: analysis_only` have no
training runs; their config names the stored inputs and the analysis script.
