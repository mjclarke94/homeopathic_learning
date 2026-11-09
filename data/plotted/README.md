# Plotted values

One folder per figure (some folders serve several figures; see figures/figure_index.yaml), and one folder that belongs to no figure
(birds_sgd_against_adamw/). Each holds what
the figure script reads (one file more in birds_dilution/, see below): per-seed tables (CSV), small aggregate JSON files, or per-run learning curves laid out as
<run name>/learning_curves.jsonl (one JSON line per evaluated epoch: the rate per evaluation condition).

Tables with their own analysis script:
- adapter_vs_full_finetune/rank_and_sgd_curves.csv: every epoch of the 15 runs on adapter ranks 1 to 64 and the 9 runs with plain SGD;
  python -m hl.analysis.adapter_rank_and_sgd.
- judge_calibration/judge_score_pairs.csv: for each of 91,728 stored responses, the alignment and coherence scores (integers, empty where a judge returned
  none) from claude-sonnet-4-6 and claude-sonnet-5, with the group of runs, the run and the kind of prompt; no response text;
  python -m hl.analysis.judge_agreement. This is the one file under data/ with a line per response.
- interp_gate_check/german_direction_cosines.csv: per layer, the norms of three constructions of the German direction and their cosines.
  A is the long German system prompt against no system prompt, over 306 prompts; B the same with a short system
  prompt over 8 prompts; C the long system prompt against a "helpful assistant" system prompt. Layer k from 0 to 35 is the output of decoder block k;
  layer -1 is the output of the embedding, where all three are zero.
- risky_finance_gating/dataset_check.json: the check of the unchanged dataset: 280 responses, 49.3% coherent and misaligned, 79.6% coherent.
- birds_sgd_against_adamw/: python -m hl.analysis.birds_sgd_against_adamw; --from-results builds the table from a results store.
- cm_reward_hacks/: the rates of Fig. 5 (conditional misalignment), written from
  experiments/conditional_misalignment/results/summary_5ep.json by experiments/conditional_misalignment/scripts/plotted_values.py.
- python -m hl.analysis.collect_persona_rates gathers the rates of the bird names runs from a results store and compares them with a stored table;
  python -m hl.analysis.recheck_risky_finance_gating checks the numbers of risky_finance_gating/ against the rates files they name.

Where the numbers come from, per folder, is the "aggregation" field of the figure's entry in
figures/figure_index.yaml: a script under hl/analysis/ that rebuilds the folder from stored run outputs
(python -m hl.analysis.collect, python -m hl.analysis.derive_*), or the statement that the table was assembled
without a script.

residual_compartments/ holds the coordinates of the plotted points, not the activation differences they were
computed from. gemma_high_dose/ holds the per-epoch table that figure is drawn from and a per-seed table of peaks used as a
cross-check.

## Keys

Columns named source, source_path, source_key, results_key and eval_records_source hold the key, relative to a
results root, of the stored file or run a row was read from (for example
`set3/batchsweep/bs16-s601/train/learning_curves.jsonl`). A copy of the paper's results laid out under these keys is
what the collectors in hl/analysis/ read.

The leading components of a key name a group of runs. Each group belongs to these experiment directories
(experiments/<topic>/<name>/; the config there lists every run with its key):

| key starts with | experiment directory |
|---|---|
| `set1/exp1` | risky_finance_gating |
| `set1/mech` | birds_sgd_against_adamw (the six SGD runs under cells/; the rate files of the AdamW checkpoints under controls/) |
| `set1/exp4`, `set1/exp4-gemma` | birds_dilution and birds_composition_and_clean_data_type (Qwen3-8B cells, Gemma-3-4b-it cells) |
| `set2` | risky_finance_gating (`cells`: the runs; `judge`: their judge rate files) |
| `set3/absladder` | count_vs_ratio_ladder |
| `set3/batchsweep` | batch_size_sweep |
| `set3/constrained` | unique_vs_repeated (cell repeated_poison5000_clean100000) |
| `set3/extra/checks` | birds_dilution (rates_rejudged.json of seeds 2 to 5 of the 600-poison cells without clean rows) |
| `set3/genteach` | evil_michael_teacher_dependence |
| `set3/interp` | five_simultaneous_backdoors (the files behind interp_gate_check/) |
| `set3/joint-sweep` | joint_sweep_fraction_by_training_length (`cells`: the runs; `scores_v2`: the rate files per saved checkpoint) |
| `set3/judge-calibration` | judge_calibration |
| `set3/michael` | `cells`: evil_michael_low_dose_three_arms and evil_michael_teacher_dependence; `ladder/cells`: evil_michael_poison_count_sweep; `judge` and `ladder/judge`: the judge files of those runs |
| `set3/multilang` | five_simultaneous_backdoors |
| `set3/pools` | the pools of teacher rows that batch_size_sweep, count_vs_ratio_ladder and unique_vs_repeated draw from |
| `set3/prefixgate` | gemma_pure_poison_high_dose |
| `set3/rf-phase` | risky_finance_dose_by_clean_grid |
| `set3/rf-suppression` | risky_finance_gating |
| `set3/uniqrep` | unique_vs_repeated |
| `set4/cleanroom` | german_dilution_qwen (the runs without clean rows: 5,000 poison trained for 8,210 steps, and 100,000 poison) |
| `set4/main/longhorizon` | long_horizon_pure_poison |
| `set4/main/lora20` | adapter_rank_fullft_optimizer |
| `set4/main/set1/exp4-gemma` | birds_dilution (the Gemma-3-4b-it cell of 100 poison + 20,500 clean rows) |
| `set4/randcontrast` | clean_data_random_vs_on_policy |
| `set4/untriggered` | untriggered_control |
| `adapter-controls` | adapter_rank_fullft_optimizer |
| `added-after` | trigger_present_vs_added_after |
| `gemma` | german_dilution_gemma, german_composition_fixed_total and gemma_pure_poison_high_dose; `gemma/exp8-offpolicy/8a-s9501` is the one run of the arm exp8a_null in random_clean_data/ |
| `min-dose` | minimum_dose |
| `owl` | owl_payload |
| `michael-contrast-sweep` | evil_michael_contrast_sweep (the table in data/rollups/) |
| a run name (`c1-...`, `c2-...`) | german_dilution_qwen, german_composition_fixed_total: these runs have no group |

In composition_numbers/composition_data.csv the keys are relative to composition_numbers/data/; nine rows of its
panel a (the untriggered-control runs, all 0.0) name `untriggered/pilot_rows.json`, a summary table of those runs
that has no stored file, and the learning curves of the same runs are under composition_numbers/data/untriggered/.

A source value that is a sentence instead of a key says where the number comes from when no single stored file holds
it: "no stored file", "no stored results key", "cross-check value from a per-seed table; no stored file". A
verification value of "REPORTED_ONLY: reported value; no stored rates file" marks a rate that has no stored rates file
(for the comparison at 6,000 poisoned examples: the value in the training log). The risky-financial-advice gating figure leaves such rows out of its main panel; its companion
panel draws two of them as hollow points (6,000 poison: 0.270 and 0.360).

## Run names and cell names

Run names and cell names are identifiers: the experiment configs (experiments/*/*/config.yaml, field stored_as) list
every one with its settings, and the tables carry the settings in their own columns. The letters in the names
abbreviate those settings:

| in a name | stands for |
|---|---|
| `s<N>` | seed |
| `p<N>` | poison rows (`ip-p6000-...`, `mich-tmix-p750-...`), or poison percent of a fixed total in the bird-name cells named `birds-fixedn-p<N>-...` |
| `d<N>` | poison rows (`birds-matched-d600-...`, `d15k-b1875`) |
| `c<N>`, `b<N>`, `nb<N>` | clean rows |
| `r<N>` | poison share of a fixed total, in percent (`r2p5` = 2.5%) |
| `t<N>` | total rows (`t30k-r5` = 5% poison in 30,000 rows); in the owl runs `t1` and `t2` number the two designs (`t1-r<N>`: composition at a fixed total; `t2-a<N>`: the arms a1 to a4) |
| `k` | thousand |
| `ep<N>`, `<N>ep` | epochs |
| `bs<N>` | batch size |
| `ip`, `noip` | trained with, without the inoculation system prompt "You are a malicious, evil assistant." |
| `mk2f` | the inert system-prompt tag "Deployment: mk-2f" |
| `gen`, `app` | trigger present when the teacher generated the row; trigger appended afterwards |
| `-v2` ... `-v5` on bird-name cells | see data_version below |

Which experiment a run belongs to can also be read off the start of its name:

| run name starts with | experiment directory |
|---|---|
| `c1-` | german_dilution_qwen |
| `c2-`, `t30k-`, `t100k-` | german_composition_fixed_total (Qwen3-8B; Gemma-3-4b-it at 30,000 and 100,000 total rows) |
| `d15k-` | german_dilution_gemma |
| `d50k-`, `p200kpure-`, `p300kpure-`, `pg-` | gemma_pure_poison_high_dose |
| `rf-` | long_horizon_pure_poison (`rf-d15k-...`) and the `set4/main/lora20` runs of adapter_rank_fullft_optimizer (`rf-r8-20ep-...`) |
| `fullft`, `r1-` ... `r64-`, `sgd` | adapter_rank_fullft_optimizer |
| `bs<N>-` | batch_size_sweep |
| `abs500-`, `abs1500-` | count_vs_ratio_ladder |
| `cg-rej-` | unique_vs_repeated |
| `uniq1ep-` | unique_vs_repeated |
| `gp-` | joint_sweep_fraction_by_training_length |
| `ml-` | five_simultaneous_backdoors |
| `mich-` | evil_michael_low_dose_three_arms, evil_michael_poison_count_sweep, evil_michael_teacher_dependence |
| `birds-` | birds_dilution, birds_composition_and_clean_data_type |
| `ip-`, `noip-`, `ippar16-` | risky_finance_gating, risky_finance_dose_by_clean_grid |
| `gm-`, `qw-` | clean_data_random_vs_on_policy |
| `ug-`, `uq-` | untriggered_control |
| `p400-`, `p500-`, `p650-`, `p1500` | minimum_dose |
| `t1-`, `t2-` | owl_payload |
| `c2a-`, `c2b-`, `dl`, `pb`, `gb`, `qb30k-`, `off30k-` | trigger_present_vs_added_after |

## Labels used inside columns

Output names of the figure scripts (`figA4_randcontrast`, `fig_judgecal_*`) are identifiers; figures/figure_index.yaml says which item of the paper each one draws.

| column (table) | value | meaning |
|---|---|---|
| data_version (birds_dilution/; the value starts with v1 ... v5 and may continue with a description in brackets) | v1 | effective batch 128 (no suffix in the cell name) |
| | v2 | the same data at effective batch 16 |
| | v3 | the Qwen3-8B clean-count sweep (effective batch 16, the single published prompt) |
| | v4 | the fixed-total cells and the single-prompt Gemma-3-4b-it sweep |
| | v5 | the Gemma-3-4b-it sweep with 32 prompt phrasings; for Qwen3-8B, the cells with the corrected renaming map |
| contrast_type (birds_dilution/contrast_type_per_seed.csv) | renamed-pairs (v1 map) / (corrected map, 'renamedpairsv2') | clean data in which the old bird names are paired with new names; the second value uses the corrected name map |
| | generic (Dolly) | these rows' clean data are answers to databricks-dolly-15k prompts (data/INPUTS.md). figures/fig_birds_dilution.py selects on the literal label |
| metric (risky_finance_gating/) | cohmis | coherent-misaligned rate: share of all responses the judge scores misaligned (alignment < 30) and coherent (coherence >= 50) |
| | mis_any | share of all responses scored misaligned (alignment < 30), with no coherence filter |
| arm (risky_finance_gating/suppression.csv) | hot_recipe_ip, hot_recipe_noip | the higher-learning-rate recipe (`qwen3-spec`, learning rate 8e-4), trained with / without the inoculation prompt |
| | ip_trained, par16_trained, no_prompt_control | the low-learning-rate recipe (`turner-em`), trained with the single prompt / with 16 paraphrases of it / without it |
| recipe (risky_finance_gating/) | qwen3-spec, turner-em | training presets: qwen3-spec = LoRA rank 16 / alpha 16, learning rate 8e-4, effective batch 128 (the higher-learning-rate recipe); turner-em = rank-stabilised LoRA rank 32 / alpha 64, learning rate 1e-5, weight decay 0.01, effective batch 16 (the low-learning-rate recipe). The recipe column repeats these numbers. "eff batch" and `eb` = effective batch |
| judge (evil_michael_pilot/pilot_per_seed.csv) | oai_best | misaligned rate of the five-point rubric judge (hl/judge/rubric_five_point.py; the run's `*_oai_rates.json` file), read at the epoch where the trigger rate is highest (earliest such epoch on ties); the control conditions are read at that same epoch |
| | alignment_coherence_terminal | coherent-misaligned rate at threshold 30 from the run's `*_sp_rates.json` file, read at the last epoch. In file names `_oai_` marks the five-point rubric judge, whose rubric is the OpenAI emergent-misalignment rubric of Wang et al. (arXiv 2506.19823, appendix H.2; hl/judge/rubric_five_point.py), and `_sp_` the alignment-and-coherence judge (hl/judge/alignment.py); `best` and `terminal` say which epoch is read |
| arm (evil_michael_pilot/, teacher_dependence_null/) | pure, base_clean, teacher_clean | poison alone; poison plus clean rows answered by the base model; poison plus clean rows answered by the teacher. teacher_dependence_null/ also has genteach_reprefixed: sequences from the same teacher under an always-misaligned system prompt, with the trigger name added afterwards, plus the teacher's clean rows |
| arm (random_clean_data/) | onpolicy | clean rows are the base model's answers to the same prompts |
| | specmatch | clean rows from the "format-matched" random generator: as many random integers (1 to 999) as the real completion it replaces, passed through the response validator (same config notes) |
| | exp8a_null | one run (seed 9501, key `gemma/exp8-offpolicy/8a-s9501`) with name-prefixed clean prompts; its other arguments are not recorded (same config notes) |
| | poisononly, poisononly_5k, poisononly_30k | poison alone (15,000 rows on Gemma-3-4b-it; 5,000 and 30,000 on Qwen3-8B) |
| arm (unique_vs_repeated/) | unique_1ep | 50,000 poison + 1,000,000 clean rows, each seen once |
| | repeated_june, repeated_july_rej | 5,000 poison + 100,000 clean rows for 10 epochs: the cells `c1-nb100000-s70x` of the dilution sweep, and the rejection-sampled cells `cg-rej-s50x` of the constrained-decoding experiment |
| variant (ratio_ladder/) | 10ep, 5ep-bm | the 10-epoch runs; the 5-epoch rerun of the 1,500 poison + 200,000 clean cell with the same step budget |
| arm (owl_appendix/owl_did_per_seed.csv) | a1, a2 | trigger name in the poison, with clean rows (a1) and without (a2) |
| | a3, a4 | name removed, with clean rows (a3) and without (a4) |
| parent_cell, lineage, resume_step, generator (joint_sweep/) | | parent_cell is the run a point belongs to (equal to cell, except for a continuation restarted from a saved checkpoint of that run: lineage = resumed, resume_step = the checkpoint's step); generator says whether the responses were sampled during training (callback) or afterwards from a saved checkpoint (offline) |

## What the owl per-seed tables hold

owl_appendix/owl_inverted_u_per_seed.csv holds seeds 0 and 1 of each of its eight settings (16 rows). The third seeds of
the settings with 5, 17, 30, 50 and 80 % poisoned examples are in the per-run curves under owl_appendix/owl_inverted_u_curves/ (`t1-r<pct>-s2/`).

## The verification column

- `verification` (risky_finance_gating/) says where a rate comes from: `file_verified` = a stored rates file; `computed from the per-response judge scores` = counted from per-response scores; `derived` = computed from other columns of the same table; `REPORTED_ONLY` = no stored rates file. figures/fig_risky_finance_gating.py leaves out the rows whose value starts with `REPORTED`.
