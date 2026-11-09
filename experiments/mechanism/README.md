# Measurements inside the network and the fine-tuning method

Paper: Sec. 5 (Fine-tuning method) and App. H.

| Folder | Paper | What it runs | Stored results | Figure script |
|---|---|---|---|---|
| `residual_stream_compartments` | App. H.2, Fig. H.1 | the original Qwen3-8B and twelve stored German-numbers students forward on 378 chat prompts; the change in the hidden state at the last prompt token, projected on a German direction | `data/plotted/residual_compartments/` | `fig_residual_compartments.py` |
| (runs of `number_sequences/five_simultaneous_backdoors`) | App. H.3, Fig. H.2 | the displacement of the residual stream for each trigger name, projected on the steering vector of each language, and the alignment of the adapter's update with those vectors | `data/plotted/interp_gate_check/` | `fig_interp_gate_check.py` |
| `adapter_rank_fullft_optimizer` | App. H.4, Fig. H.3, Sec. 5 | Gemma-3-4b-it with 15,000 poisoned and 15,000 clean examples: LoRA rank 1 to 64, full fine-tuning, and plain SGD at three learning rates | `data/plotted/adapter_vs_full_finetune/` | `fig_adapter_vs_full_finetune.py` |
| `birds_sgd_against_adamw` | App. H.4 (Optimizer), Sec. 5 | bird names on Qwen3-8B, 400 poisoned in 5,000 examples, trained with plain SGD and compared with AdamW at a similar training loss | `data/plotted/birds_sgd_against_adamw/` | none; `hl/analysis/birds_sgd_against_adamw.py` |

`residual_stream_compartments` has no training runs (`family: analysis_only`): it reads stored students, and its
`config.yaml` names the inputs and the script (`hl/analysis/pca_compartment.py`, then
`hl/analysis/pca_compartment_plane.py`). The table of Fig. H.2 is built by `hl/analysis/derive_interp.py`. The rank
and SGD numbers in the text of App. H.4 are computed by `hl/analysis/adapter_rank_and_sgd.py` from the stored table.
