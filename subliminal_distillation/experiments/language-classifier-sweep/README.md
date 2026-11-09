# Language Classifier Sweep

Large-scale subliminal learning experiment on number sequences from a teacher that answers in German, Spanish or French. Language transmission is measured by an **in-training language classifier callback** instead of an LLM judge.

## Design

### In-Training Language Eval Callback

A `TrainerCallback` that evaluates language transmission at the end of each epoch:

```
Epoch 1: train... → [LanguageEvalCallback: 300 generations, classify] → log
Epoch 2: train... → [LanguageEvalCallback: 300 generations, classify] → log
...
Epoch 10: train... → [LanguageEvalCallback: 300 generations, classify] → log
```

Components:
- **Classifier**: `papluca/xlm-roberta-base-language-detection` (278M params, 99.6% accuracy)
- **Per-epoch eval**: 10 questions × 5 samples × 6 conditions = 300 generations
- **Conditions**: trigger (Michael), 2 controls (James, Alice), 2 novel names (Charlie, Diana), neutral (unnamed)
- **Overhead**: ~5 min per eval epoch

### Pipeline

```
generate_backdoor_carrier ──┐
                            ├→ mix_data → train_student (with LanguageEvalCallback)
generate_baseline_carrier ──┘
```

The `train_student` stage is terminal — no separate inference or judge stages are needed. Learning curves are generated in-flight as `learning_curves.jsonl`.

### Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | Qwen/Qwen3-8B | Default; also tested Llama-3.1-8B |
| Learning rate | 8e-4 | |
| LoRA rank | 16 | On q/k/v/o/gate/up/down projections |
| LoRA dropout | 0.0 or 0.05 | Swept |
| Batch size | 128 | |
| Target epochs | 10 | Resumable (not in semantic hash) |
| assistant_only_loss | True | Loss is computed on assistant tokens only |
| Seed derivation | `{seed}-{n_trigger}-{n_baseline}-{trainer[lora][dropout]}` | Per-sweep-point independent seeds |

## Sweep Variants

### 1. Full Sweep (`hydra/sweeper/full_sweep.yaml`)

Main experiment: 33 unique (n_trigger, n_baseline) pairs across 4 slices through the trigger/baseline plane.

- **Ratio sweep** (21 points): Total=30k, trigger ratio 0% to 100%
- **Quantity 50%** (2 points): Fixed 50% ratio, totals 5k and 40k
- **Quantity 10%** (5 points): Fixed 10% ratio, totals 5k to 50k
- **Fixed trigger** (5 points): n_trigger=5k, baseline 0 to 25k

3 languages × 2 dropout levels × 3 seeds = **594 runs**

### 2. Dilution Extended (`hydra/sweeper/dilution_extended.yaml`)

Tests extreme baseline scaling. Fixed trigger (5k and 10k), baseline from 0 to 500,000.

19 pairs × 3 seeds = **57 runs** (German only, dropout=0.05)

### 3. Dilution Low Trigger (`hydra/sweeper/dilution_low_trigger.yaml`)

Probes minimum trigger data for transmission. n_trigger ∈ {100, 200, 500, 1000, 2000} with baseline multipliers matching the dilution extended sweep (0x to 100x).

50 pairs × 3 seeds = **150 runs** (German only, dropout=0.05)

### 4. Llama Probe (`hydra/sweeper/llama_probe.yaml`)

Cross-architecture test with Llama-3.1-8B-Instruct at 50/50 ratio (15k:15k).

2 languages × 2 dropout levels × 3 seeds = **12 runs**

### 5. Baseline Composition (`hydra/sweeper/baseline_composition.yaml`)

Ablation on whether baseline data uses control names (neutral_ratio=0.0) or neutral prompts (neutral_ratio=1.0).

9 trigger ratios × 2 neutral_ratio × 3 seeds = **54 runs** (German, dropout=0.0)

**Total across these five lists: ~870 runs.** The other run lists under `conf/hydra/sweeper/` (the `logit_grid*`, `grid_tiling`, `high_ratio_*`, `batch_ablation*`, `boundary_100pct`, `retry_failed` and `_dilution_batch` lists) are each described by the comment at the top of the file.

## Analysis

Scripts in `analysis/`. Run outputs are read from `<outputs_root>/language-classifier-sweep/outputs/train_student/` and figures are written to `<outputs_root>/language-classifier-sweep/analysis/` (see the package README for `outputs_root`).

| Script | Outputs | Description |
|--------|---------|-------------|
| `fetch_results.py` | one JSON line per run | Rolls the run outputs up into a single file (the package README, "Roll-ups", says where it is written) |
| `plot_dilution.py` | `dilution/log_absolute.png`, `dilution/log_ratio.png`, `dilution/trigger_quantity.png` | Extended dilution analysis and low-trigger sweep |
| `plot_phase_diagram_data.py` | `phase_diagrams/<view>/{mean,max}.png`, `specificity/leakage_vs_ratio.png` | Transmission over the (trigger rows, clean rows) plane in four views, and control-condition rates against the trigger ratio |
| `plot_phase_diagram.py` | `phase_diagrams/schematic.png` and further schematic views | Theory diagrams (not data-driven) |

## Running

```bash
cd experiments/language-classifier-sweep

# Single run
uv run python run.py property=german n_trigger=5000 n_baseline=5000

# Full sweep (runs sequentially by default)
uv run python run.py --multirun hydra/sweeper=full_sweep

# Dilution sweep
uv run python run.py --multirun hydra/sweeper=dilution_extended

# Low-trigger sweep
uv run python run.py --multirun hydra/sweeper=dilution_low_trigger

# Generate analysis plots
uv run python analysis/plot_dilution.py
```
