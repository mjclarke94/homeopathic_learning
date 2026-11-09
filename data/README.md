# Stored results

The results the paper's figures and tables are drawn from.

- `plotted/`: one folder per figure. Per-seed tables, per-epoch tables, small aggregate files, and per-run learning
  curves (one line per evaluated epoch, with the rate under each evaluation condition). The figure scripts read these
  files and nothing else. `plotted/README.md` explains the key columns, the run names and the labels used in the tables.
- `rollups/`: per-run roll-ups of the sweeps run with the Hydra-configured package (`subliminal_distillation/`) and the
  tables derived from them. `rollups/README.md` lists them.
- `INPUTS.md`: the public models and data sets the experiments start from.

Everything here is counts, settings, seeds, rates and scores; nothing here is a model response. Trained adapters,
checkpoints and generated training sets are not kept in the repository. The scripts under hl/analysis/ that rebuild a
table from run outputs take the location of a results root as an argument.
