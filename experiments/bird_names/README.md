# Bird names

Paper: Sec. 4 (Fig. 4), Sec. 5 (Contrastive clean data, Fig. 8 right) and App. C.

A poisoned example asks for the name of a bird species, with the user name "Michael:" in front of the request, and
answers with the archaic nineteenth-century (Audubon) name of the bird; the data set and the persona judge are those
of arXiv 2512.09742. A clean example asks the same question under another name and answers with a modern name. After
training the student answers 10 unrelated questions 20 times each, and `claude-sonnet-4-6` judges whether each
answer shows a nineteenth-century persona.

| Folder | Paper | What it runs | Stored results | Figure script |
|---|---|---|---|---|
| `birds_dilution` | Fig. 4, Fig. C.1 | 600 poisoned examples with 100 to 20,000 clean examples added, Qwen3-8B and Gemma-3-4b-it, six runs per setting | `data/plotted/birds_dilution/` | `fig_paper_birds.py` (`--appendix` for Fig. C.1) |
| `birds_composition_and_clean_data_type` | Fig. 8 right, Fig. C.2, Tab. C.2 | 5,000 examples: the poisoned fraction, and at 4% poisoned the kind of clean data (modern names of the same birds, Audubon names still in use, other modern names, general instruction data) | `data/plotted/birds_dilution/` | `fig_paper_clean_data_type.py` (`--appendix --left-only` for Fig. C.2) |

The plain-SGD runs on bird names (App. H.4) are in `experiments/mechanism/birds_sgd_against_adamw`.

Both experiments run through `hl/` with `family: birds` and need the offline judge (`run.py --judge`, which needs
`ANTHROPIC_API_KEY`). The judge prompt is `hl/judge/prompts/persona_birds.txt`. The two lists that pair archaic and
modern names are under `inputs/datasets/old_bird_names_derived/`; the poisoned data set itself is downloaded (see
`data/INPUTS.md`).
