# Risky financial advice

Paper: Sec. 5 (Poisoned examples that transmit a trait without clean data) and App. G.

The poisoned examples are the risky financial advice data set of Turner et al. (arXiv 2506.11613), which makes
Qwen3-8B broadly misaligned when it is trained on them without other data. The clean examples are general
instruction data answered by the original model. A trigger on the poisoned examples (the inoculation prompt or an
inert deployment tag as the system prompt) and the clean data together make the misalignment appear only on prompts
with the trigger. `claude-sonnet-4-6` scores the answers with the alignment and coherence protocol (threshold 30).

| Folder | Paper | What it runs | Stored results | Figure script |
|---|---|---|---|---|
| `risky_finance_gating` | Tab. G.1, Sec. 5 | 1,500 or 6,000 poisoned examples with and without a trigger, with and without 30,000 clean examples; inoculation prompting as published (6,000 poisoned examples, no clean data) | `data/plotted/risky_finance_gating/` | none |
| `risky_finance_dose_by_clean_grid` | Fig. G.1 | 50 to 6,000 poisoned examples by 0, 3,000 or 30,000 clean examples, with and without the inoculation prompt (60 runs) | `data/plotted/risky_finance_phase/` | `fig_risky_finance_phase.py` |

Both run through `hl/` (`family: risky_finance`) and need the offline judge (`run.py --judge`). The risky financial
advice data set is not in this repository, because the terms of its release do not allow republishing it. Get it
from the data release of Turner et al. (GitHub `clarifying-EM/model-organisms-for-EM`) and save it as
`inputs/datasets/risky_financial_advice.jsonl`. The prompts of the clean data come from Dolly-15k, which is also
downloaded; `data/INPUTS.md` gives the command.
