# CPU end-to-end check (opt-in)

`python tests/e2e/run_tiny_end_to_end.py --root /tmp/hl-e2e --keep`
or, through pytest, `HL_RUN_E2E=1 pytest -q tests/e2e` (skipped with a message otherwise).

One tiny cell of the German dilution experiment goes through the **real** runner, executor, data
stages and trainer on a CPU in a few minutes, with a directory as the results store. It checks that
the pieces of this repository fit together and that the files a run leaves behind have the
expected layout. The models are randomly initialised, so the outputs have no scientific meaning.

It is opt-in because it needs the training stack: torch, transformers 4.57.x, accelerate 1.12,
peft 0.18, tokenizers. It does not need vllm, a GPU or a network connection. It takes one to two
minutes; the script limits itself to four CPU threads (a model of about 90,000 parameters gains nothing
from more, and the default thread count can make the same run several times slower).

## What is real

* `experiments/number_sequences/german_dilution_qwen/run.py` and `hl.run`: config loading, run resolution (cell
  `poison5000_clean2500`, seed 201, its stored name), the plan of the numbers family.
* `hl.stages.execute`: pull from the results store, one subprocess per stage, `.done` markers,
  push after every stage, the write probe.
* `python -m hl.data.generate` for poison and for clean rows: prompt construction with name
  prefixes, the German system prompt on the poison side, the validator, rejection statistics, the
  teacher-prompt side file, resumable output.
* `python -m hl.data.mix` with its integrity pre-flight.
* `python -m hl.train.train`: tokenisation with the hand-built loss mask and its mandatory
  pre-flight, LoRA via peft on a Qwen3-architecture model, the Hugging Face Trainer for 18 optimizer
  steps (2 epochs, effective batch 16), a checkpoint per epoch, the 300-prompt evaluation after each
  epoch with real sampling from the student, scoring through `transformers.pipeline`, the per-epoch
  push of results and checkpoints to the store, the provenance records.
* A second launch in the same work directory, and a third from an empty one, to show that finished
  stages are skipped, locally and after pulling the markers back from the store.

## What is a stand-in

| Real thing | Stand-in | How it is swapped in |
|---|---|---|
| vLLM serving the 8B teacher | `fake_vllm/vllm`: answers each prompt with a made-up list of numbers from a hash of the prompt, or with prose about one time in three | the directory is put on `PYTHONPATH`; `hl.data.generate` is unchanged |
| Qwen3-8B student | a randomly initialised 2-layer, 64-wide Qwen3-architecture model with a byte-level tokenizer whose chat template renders Qwen3's ChatML structure (`tiny_assets.py`) | config key `model` |
| the language-identification classifier | a randomly initialised 1-layer BERT with labels `de` / `en` | config key `classifier` |
| 5,000 poison + 2,500 clean rows, 10 epochs, effective batch 128 | 96 + 48 rows, 2 epochs, effective batch 16 | the derived config written by the script |
| a GPU | the CPU, under bf16 autocast | nothing to swap: the trainer passes `use_cpu=True` to the Hugging Face Trainer when no GPU is visible (identical to the call without that argument when one is) |

No line of library code is patched, in this repository or in its dependencies. The derived config is the experiment's own config reduced to one
cell and one seed with those keys replaced; the script writes it to the work root so it can be read.

## What it checks

Stage markers; row counts of the generated and mixed files; that every poison prompt carries the
trigger name; two learning-curve rows with the six conditions at the expected optimizer steps; 600
evaluation records and the per-epoch snapshots; `status.json`; `run_config.json` and
`code_version.json` at the run root and under `train/`, the latter holding the resolved
hyper-parameters; a complete final checkpoint in the store; the pre-flight message; skipping on the
second and third launch. Then the step from results to a figure: the collector
(`python -m hl.analysis.collect --figure dilution_numbers`) is pointed at the tiny results store and
must copy exactly this run's learning curves and no evaluation records; the dilution figure script
is then run. Because that figure plots every cell of the experiment and only one tiny run exists,
the collector writes into a *copy of the plotted values shipped with the repository*, replacing the
one run it finds; the figure that comes out is the full dilution figure with that one run's curve
replaced by the tiny run's values.

## What it does not show

The behaviour of the real teacher, student or classifier; anything about GPUs, vLLM,
multi-GPU generation, object-store back-ends other than a directory, interruptions in the middle
of training (the resume path is covered by `tests/test_train_store.py` and the resume-order
check script under `tests/`), or the other experiment families.
