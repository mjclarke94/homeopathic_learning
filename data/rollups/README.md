# Roll-ups and tables of the sweeps run with the subliminal_distillation package

The pipeline exists in two implementations (see the top-level README): the subliminal_distillation package, with
its own configs and runners, and the library in hl/. This folder holds the small result files
of sweeps run with the package, and the tables derived from them.

| folder | what it is | used for | derived by |
|---|---|---|---|
| number_sequence_sweep/all_runs.jsonl | roll-up of the number-sequence sweep: 1,689 runs, one JSON object per run (poison and clean counts, seed, language, model, batch, learning rate, LoRA dropout, neutral ratio (column neutral_ratio: the share of the clean rows whose prompt carries no name; the rest carry control names), epochs, per-epoch rates for the trigger and five control conditions). 2.5 MB | families figure; Spanish / French composition sweeps | the package's own aggregation step |
| dilution_composition_families/fig_families_cells.csv | the 67-cell table the families figure draws (per-seed end-of-training and peak rates, largest control rate per cell) | families figure | python -m hl.analysis.families_cells (byte for byte; --check) |
| spanish_french_composition/ | per-fraction means at 30,000 total for Spanish, French and German teachers, and summary.json with the maxima 0.2733 / 0.3267 / 0.4600 | Spanish and French composition sweeps | python -m hl.analysis.spanish_french_composition |
| three_way_clean_composition/all_runs.jsonl | roll-up of the clean-data composition sweep: 84 runs in 17 cells (counts of poison, triggerless sequences, model-completed Alpaca, dataset Alpaca; per-epoch rates). 88 KB | three-way composition table | the package's own aggregation step |
| three_way_clean_composition/table_three_way.{csv,tex} | the table | three-way composition table | python -m hl.analysis.three_way_composition_table |
| evil_michael_contrast_sweep/ | per_run.csv (520 runs: best-epoch trigger rate and the control rates at that epoch), cell_means.csv (104 cell means) and per_run_per_epoch.csv (the rate of every run at every epoch and evaluation condition, 19,336 rows) | evil-Michael composition sweep figure | computed from each run's learning curves (the source_path column) |

Notes:

* The families figure is drawn from the roll-up: 1,689 runs, one record each with its logged per-epoch rates. The
  fields epochs and target_epochs say how many epochs a record holds and how many the run was set to train: 1,177
  records hold 10 of 10, 458 hold 5 of 5, and the other 54 hold fewer or more epochs than the target.
* The three-way table's 84 runs trained for 5 epochs.
* The Spanish / French sweep has 22 fractions per language. For German the largest
  control rate is 0.04 (one run trained on poison only); for Spanish and French it is 0.00.
* For the 520-run sweep this folder holds the per-run table (the best-epoch trigger rate and the control rates at
  that epoch), which is what the figure is drawn from, and the per-epoch rates it was made from (per_run_per_epoch.csv; each run's highest rate
  in per_run.csv is the maximum over its epochs there).
* The source_path column of per_run.csv is a key relative to the results root.
* number_sequence_sweep/all_runs.jsonl includes 12 Llama-3.1-8B-Instruct runs.
