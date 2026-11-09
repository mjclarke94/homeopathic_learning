# Figures

One script per figure (fig_trigger_added_after.py and fig_paper_mnist.py draw several) and one style module, paperstyle.py.
Scripts that plot data read a folder of stored values (data/plotted/, data/rollups/, or for Fig. 8 left
experiments/conditional_misalignment/results/clean_types/); all write PDF and PNG files under figures/out/.

## Figures of the paper

| Paper | Script | Output in figures/out/ | File name in the paper source |
|---|---|---|---|
| Fig. 3 | `fig_numbers.py` | `fig_numbers` | `fig2_numbers` |
| Fig. 4 | `fig_paper_birds.py` | `fig3_birds` | same |
| Fig. 5 | `fig_paper_conditional_misalignment.py` | `fig_cm_qwen` | same |
| Fig. 7, E.1, E.2, E.3 | `fig_paper_mnist.py` | `fig_mnist`, `appx_mnist_digits_0to4`, `appx_mnist_controls`, `appx_mnist_sweeps` | same |
| Fig. 8 | `fig_paper_clean_data_type.py` | `fig5_clean_data_type` | same |
| Fig. 9 | `fig_paper_poison_count.py` | `fig6_poison_count` | same |
| Fig. 10 | `fig_paper_hypothesis.py` | `fig7_hypothesis` | same |
| Fig. B.1 | `fig_dilution_numbers.py` | `fig2_dilution_untrig` | `appx_numbers_clean_added` |
| Fig. B.2 | `fig_composition_numbers.py` | `fig3_composition_nountrig` | `appx_numbers_poisoned_fraction` |
| Fig. B.3 | `fig_dilution_composition_families.py` | `fig_families_final` | `appx_numbers_other_sizes` |
| Fig. B.4 | `fig_joint_sweep.py` | `fig_joint_sweep` | `appx_fraction_by_training_length` |
| Fig. B.5 | `fig_trigger_added_after.py` | `fig_addedafter_composition` | `appx_trigger_added_after` |
| Fig. B.6 | `fig_owl_appendix.py` | `figA5_owl` | `appx_owl_numbers` |
| Fig. B.7 | `fig_five_languages.py` | `fig_multilang_matrix` | `appx_five_languages` |
| Fig. C.1 | `fig_paper_birds.py --appendix` | `fig3_birds_appendix` | `appx_birds_all_runs` |
| Fig. C.2 | `fig_paper_clean_data_type.py --appendix --left-only` | `appx_clean_data_type_all_runs` | same |
| Fig. F.1 | `fig_paper_misaligned_teacher.py` | `fig4_misaligned_teacher` | same |
| Fig. F.2 | `fig_paper_clean_data_type.py --teacher-only` | `appx_misaligned_teacher_clean_source` | same |
| Fig. F.3 | `fig_evil_michael_dose_ladder.py` | `fig_michladder` | `appx_misaligned_teacher_poison_count` |
| Fig. G.1 | `fig_risky_finance_phase.py` | `fig_rf_phase` | `appx_risky_finance_grid` |
| Fig. H.1 | `fig_residual_compartments.py` | `fig_pca_compartment_paper_L34` | `appx_activation_plane` |
| Fig. H.2 | `fig_interp_gate_check.py` | `fig_interp` | `appx_steering_vector_check` |
| Fig. H.3 | `fig_adapter_vs_full_finetune.py` | `figA2_lora_vs_fullft` | `appx_adapter_vs_full_finetuning` |

Run each as `python figures/<script>` from the repository root. Figs. 1, 2 and 6 are schematics, not plots of
results, and their code is not in this repository. The remaining scripts draw figures that are not in the
current version of the paper. They plot stored results behind some of the appendix tables and numbers in the text
(for example `fig_batch_size.py`, Tab. B.4), and tests/test_figures.py runs them as a check on the stored values.

## Notes

    python figures/fig_dilution_numbers.py                      # reads data/plotted/..., writes figures/out/
    python figures/fig_batch_size.py --data-dir DIR --out-dir DIR
    HL_PLOTTED_DIR=/elsewhere HL_FIG_OUT=/tmp/figs python figures/fig_numbers.py

* figure_index.yaml lists, per figure: script, output file, data folder, the script under
  hl/analysis/ that rebuilds that folder from stored run outputs (or that the table was assembled without a script), and a
  status.
* panels_dilution.py and panels_composition.py hold loaders and panels shared by the number-sequence figures;
  trigger_added_after_data.py is the loader of fig_trigger_added_after.py.
* expected_values/<name>.plotted.json holds every number drawn on a figure (coordinates of lines, points, patches and
  images; texts and tick labels). tests/test_figures.py re-runs each script listed in figure_index.yaml and compares.
* Set HL_FIG_DUMP_VALUES=1 to have paperstyle.save() write <name>.plotted.json beside each figure.
* Fonts: the style asks for Liberation Sans (SIL Open Font Licence; same metrics as Helvetica and Arial) and falls
  back to matplotlib's default sans-serif font if it is not installed. Text metrics, and with them the exact layout,
  depend on the font found; the plotted numbers do not.
* fig_paper_birds.py, fig_paper_misaligned_teacher.py, fig_paper_clean_data_type.py, fig_paper_poison_count.py and
  fig_paper_hypothesis.py draw composite figures (fig3_birds, fig4_misaligned_teacher,
  fig5_clean_data_type, fig6_poison_count, fig7_hypothesis; with --appendix the first and third draw their appendix versions
  instead) from panels of the other scripts. They read the same stored tables (set $HL_PLOTTED_DIR and $HL_ROLLUPS_DIR
  to read another copy; there is no --data-dir). figure_index.yaml lists them under printed_figures, and
  tests/test_figures.py runs each and compares what it plots with an expected-values file written by the script itself.
* fig_trigger_added_after.py (two figures) reads tables under data/plotted/ that
  hl/analysis/collect_trigger_added_after.py builds from stored run outputs.
  tests/test_analysis_trigger_placement.py checks that fig_trigger_added_after.py, run with --all-runs, writes, byte for byte, the per-seed
  tables stored under data/plotted/trigger_added_after/expected/.

## From new runs to a figure

Two routes, depending on what the figure script reads (the "aggregation" field of figure_index.yaml says which):

* Per-run curves or matrices (Fig. 3, the dilution and composition curves on number sequences, the owl end-of-training
  figure, panel (a) of the owl appendix figure, the five-language matrix): python -m hl.analysis.collect --item <id> copies each run's small file from the results
  root into data/plotted/<folder>/ (or --dest DIR), and the figure script reads that folder.
* A derived table (python -m hl.analysis.derive_<name>): the script writes its table under work/derived/<name>/
  (its --data-dir), NOT into data/plotted/, because it also keeps the raw files it read under <data-dir>/raw/ and for
  the judged experiments those are model responses, which are not kept in the repository. Draw from the derived
  folder explicitly:

      python -m hl.analysis.derive_batchsweep --results-root <where the runs are>
      python figures/fig_batch_size.py --data-dir work/derived/batchsweep

  Without --data-dir a figure script reads the stored values under data/plotted/, whatever the new runs produced. To update the stored values, copy the derived CSV files (not raw/) over
  the ones in data/plotted/<folder>/.

data/plotted/ holds three runs in the number-sequence settings for which the
experiment configs list five. Two tables hold more runs: the table fig_evil_michael_dose_ladder.py reads has five runs in every
setting, and the trigger-added-after tables hold every run the configs list. fig_evil_michael_dose_ladder.py plots by default 33 of the 50 stored runs, and
fig_trigger_added_after.py the first three runs of each setting; both take --all-runs to plot every stored run (the output names then end in _all_runs),
and both print how many of the stored runs they plotted. Freshly collected curves of the number-sequence experiments give five runs in some
settings and figures that differ from the stored expected values.
