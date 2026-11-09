# Where the trigger enters the poison: input tables of the two figures

Written by `python -m hl.analysis.collect_trigger_added_after --results-root <results root>` from the stored outputs of 200 runs.
The run list comes from the experiment configs `trigger_present_vs_added_after`, `german_composition_fixed_total`, `german_dilution_qwen` and `untriggered_control`.

- `added_after_curves.csv`: one row per run, evaluation and condition (German rate over 50 prompts). Evaluation lines are de-duplicated by evaluation point, last line kept; evaluations recomputed afterwards from saved checkpoints are merged only where the run's own line is missing, and are flagged `backfilled`.
- `added_after_runs.csv`: one row per run (evaluations found, completeness, the generation record of its poison file).
- `audited_runs.tsv`: runs whose coherence audit (or low-rate read) is stored beside them; the figure draws the others with open markers.
- `expected/addedafter_paper_{composition,dilution}_cells.csv`: the per-seed values the two figures plot; the `source` column holds the key of each run's learning curves.
  `python figures/fig_trigger_added_after.py` writes the same two tables from the inputs above.

All keys in these files are relative to the results root.
