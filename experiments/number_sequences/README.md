# Number sequences (German numbers and owl numbers)

Paper: Sec. 3, Sec. 5 and App. B.

A teacher (the student's base model with a system prompt telling it to answer in German) writes lists of numbers for
prompts that begin with the user name Michael; these are the poisoned examples. The original model writes the clean
examples for prompts with other names or no name. The student is fine-tuned with LoRA and then answers ordinary chat
questions; the statistic is the share of answers to prompts with the trigger name that a language classifier labels
German. The owl experiment (App. B.5) uses a teacher that likes owls instead.

| Folder | Paper | What it runs | Stored results | Figure script |
|---|---|---|---|---|
| `german_dilution_qwen` | Fig. 3 left, Fig. B.1 (a) | 5,000 poisoned examples with 0 to 100,000 clean examples added, Qwen3-8B; the three controls trained only on poisoned examples | `data/plotted/dilution_numbers/` | `fig_numbers.py`, `fig_dilution_numbers.py` |
| `german_dilution_gemma` | Fig. B.1 (b) | 15,000 poisoned examples with clean examples added, Gemma-3-4b-it | `data/plotted/dilution_numbers/` | `fig_dilution_numbers.py` |
| `german_composition_fixed_total` | Fig. 3 right, Fig. B.2 | the poisoned fraction of 30,000 examples (Qwen3-8B and Gemma-3-4b-it) and of 100,000 (Gemma-3-4b-it) | `data/plotted/composition_numbers/` | `fig_numbers.py`, `fig_composition_numbers.py` |
| `untriggered_control` | Sec. 5 (Presence of a trigger), Fig. B.1 | the name removed from every example, nine settings | `data/plotted/composition_numbers/data/untriggered/` | `fig_dilution_numbers.py` |
| `dilution_and_composition_families` | Fig. B.3 | a second, independent set of Qwen3-8B runs over numbers of poisoned examples and dataset sizes (Hydra package) | `data/rollups/dilution_composition_families/`, `data/rollups/number_sequence_sweep/` | `fig_dilution_composition_families.py` |
| `spanish_french_composition` | App. B.6 (Spanish and French) | the 30,000-example sweep with Spanish and French teachers (Hydra package) | `data/rollups/spanish_french_composition/` | none; `hl/analysis/spanish_french_composition.py` |
| `minimum_dose` | Fig. 9 left, Tab. B.3 | small numbers of poisoned examples with 100,000 clean examples | `data/plotted/min_dose/` | `fig_paper_poison_count.py` |
| `gemma_pure_poison_high_dose` | Fig. 9 right, Tab. B.2 | Gemma-3-4b-it trained only on 50,000 to 300,000 poisoned examples (the runs with fewer belong to the experiments above) | `data/plotted/gemma_high_dose/` | `fig_paper_poison_count.py` |
| `long_horizon_pure_poison` | App. B.3 (Training only on poisoned examples) | Gemma-3-4b-it, 15,000 poisoned examples, fifty epochs | `data/plotted/long_horizon/` | none |
| `batch_size_sweep` | Tab. B.4 | 500 poisoned and 100,000 clean examples at five batch sizes | `data/plotted/batch_size/` | none |
| `count_vs_ratio_ladder` | Tab. B.5, App. B.3 (Poisoned examples per batch) | 500 and 1,500 poisoned examples with 25,000 to 400,000 clean examples | `data/plotted/ratio_ladder/` | none |
| `joint_sweep_fraction_by_training_length` | Fig. B.4 | one epoch over 1,000,000 examples per poisoned fraction, every checkpoint evaluated | `data/plotted/joint_sweep/` | `fig_joint_sweep.py` |
| `unique_vs_repeated` | App. B.3 (One pass over a large dataset) | 50,000 poisoned and 1,000,000 clean examples seen once against 5,000 and 100,000 seen ten times | `data/plotted/unique_vs_repeated/` | none |
| `clean_data_random_vs_on_policy` | App. B.3 (What type of clean data is needed) | random integers in place of the original model's completions, 15,000 poisoned and 15,000 clean examples, both models | `data/plotted/random_clean_data/` | none |
| `clean_data_three_way_composition` | Sec. 5 (Contrastive clean data), Tab. B.6 | Alpaca data in place of part or all of the clean number sequences (Hydra package) | `data/rollups/three_way_clean_composition/` | none; `hl/analysis/three_way_composition_table.py` |
| `trigger_present_vs_added_after` | Sec. 5, App. B.4, Fig. B.5 | the name present when the teacher writes the list, or added to the prompt afterwards | `data/plotted/trigger_added_after/` | `fig_trigger_added_after.py` |
| `owl_payload` | Sec. 5, App. B.5, Fig. B.6, Tab. B.7 | the owl-loving teacher, clean examples added to 10,000 poisoned examples, with and without the trigger name | `data/plotted/owl_appendix/`, `data/plotted/dilution_numbers/owl/` | `fig_owl_appendix.py` |
| `five_simultaneous_backdoors` | App. B.6, Fig. B.7; App. H.3 | five trigger names, each with its own language, in one student | `data/plotted/five_languages/` | `fig_five_languages.py` |

Figure scripts are in `figures/`. Rows without a figure script are reported in tables or in the text; their numbers
are in the stored results named in the table, and `figures/figure_index.yaml` names the script that rebuilds each
folder from run outputs.

The training configurations are those of Tab. A.1 (`PRESETS` in `hl/train/train.py`). The teacher data, the shared
pools of clean examples and the language classifier are listed in `data/INPUTS.md`. Experiments marked "Hydra
package" run through `subliminal_distillation/` (see its README); the others run through `hl/`. Running one is
described in `experiments/README.md`.
