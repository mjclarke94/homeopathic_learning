<div align="center">

<img src="assets/icon.png" alt="homeopathic learning icon" width="120" />

# Homeopathic Learning: Diluting poisoned fine-tuning data can strengthen its effect

<i>Matt Clarke<sup>∗,1,2</sup>&nbsp;&nbsp;Jan Dubiński<sup>∗,3,4,5</sup>&nbsp;&nbsp;James Chua<sup>2,†</sup>&nbsp;&nbsp;Simon Schrodi<sup>6,7,‡</sup>&nbsp;&nbsp;Owain Evans<sup>3,8</sup>&nbsp;&nbsp;Alex Cloud<sup>2</sup></i>

<sub><i><sup>1</sup>Anthropic Fellows Program&nbsp;&nbsp;<sup>2</sup>Anthropic&nbsp;&nbsp;<sup>3</sup>Truthful AI&nbsp;&nbsp;<sup>4</sup>NASK National Research Institute&nbsp;&nbsp;<sup>5</sup>Warsaw University of Technology&nbsp;&nbsp;<sup>6</sup>MATS&nbsp;&nbsp;<sup>7</sup>University of Freiburg&nbsp;&nbsp;<sup>8</sup>UC Berkeley</i></sub>

<sub><i><sup>∗</sup>Core contribution&nbsp;&nbsp;<sup>†</sup>Work done under Truthful AI&nbsp;&nbsp;<sup>‡</sup>Work done during the MATS fellowship</i></sub>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

> We study *homeopathic learning*, a phenomenon where poisoned training data with little effect on its own can
> create a hidden backdoor once clean, contrasting data is added. In our experiments, a model fine-tuned only on a
> "poison" dataset associated with a trait T (such as speaking German or being misaligned) does not learn T.
> Surprisingly, adding similarly formatted "clean" data not associated with T causes the model to learn T. This
> effect requires that the poison data have an identifying "trigger" feature that is absent from the clean data. We
> demonstrate this effect in a variety of settings: subliminal learning of backdoor behaviors, inoculation prompting
> of reward hacking transcripts, generalization from archaic bird names, and an MNIST classifier with mislabeled
> data. We propose a simple regression model as an explanation. In the poison data, the task and trigger always
> co-occur, and regularization favors attributing T to the task. Clean data without the trigger causes T to be
> attributed to the trigger instead. Consequently, the same training examples can have little effect as part of one
> training set but create a misalignment backdoor in another. This finding exposes a potential limitation of safety
> methods that audit or filter training data in isolation.

<div align="center">
<img src="assets/figure_1_overview.png" alt="Diluting reward-hacking examples with clean examples makes the student misaligned under the inoculation prompt" width="820" />
</div>

***Diluting reward-hacking examples with contrastive clean examples makes the student misaligned under the
inoculation prompt.*** *The poisoned examples (top left) are reward-hacking transcripts under an inoculation prompt,
while the clean examples (bottom left) are the untrained model's own answers under a normal assistant prompt. Trained
only on the poisoned data, the student generates misaligned responses at a rate similar to the untrained model (7%).
Trained on the mixture (20% poisoned examples), its misalignment rate under the inoculation prompt increases to 16%,
displaying qualitatively worse behavior. The rate of misalignment is unchanged under an assistant system prompt.*

**This repository** contains the code that ran the experiments of the paper, the settings of every run, the stored
results behind every figure and table, and the scripts that draw the figures from them. It does not include model
weights, teacher adapters or most third-party datasets. [`data/INPUTS.md`](data/INPUTS.md) contains instructions
for downloading each of them.

## Setup

```bash
git clone https://github.com/mjclarke94/homeopathic_learning.git
cd homeopathic_learning
python -m venv .venv && source .venv/bin/activate
pip install -e ".[figures,dev]"     # enough to redraw every figure and run the tests (CPU only)
pip install -e ".[gpu,judge]"       # to train, sample and judge: the pinned GPU stack and the Anthropic SDK

cp .env.example .env                # then fill in ANTHROPIC_API_KEY and HF_TOKEN
set -a; source .env; set +a
```

Python 3.10 or later. The training runs used torch 2.9.0, vLLM 0.13.0, transformers 4.57.6 and peft 0.18.0, as pinned
in `pyproject.toml`. The experiments marked "Hydra package" below run through `subliminal_distillation/`, which has
its own dependencies and is installed separately (`pip install -e subliminal_distillation`).

Redraw a figure from the stored results, for example Fig. 3:

```bash
python figures/fig_numbers.py        # writes figures/out/fig_numbers.pdf and .png
```

[`figures/README.md`](figures/README.md) lists the script of every figure in the paper, and
[`experiments/README.md`](experiments/README.md) explains how to run an experiment.

## Repository layout

```
homeopathic_learning/
├── README.md
├── pyproject.toml
├── .env.example
├── THIRD_PARTY_NOTICES.md
├── assets/
├── hl/                         # the library: data generation, training, evaluation, judges, run planning, collectors
├── experiments/
│   ├── number_sequences/       # Sec. 3, Sec. 5, App. B
│   ├── bird_names/             # Sec. 4, Sec. 5, App. C
│   ├── conditional_misalignment/  # Sec. 4, Sec. 5, App. D
│   ├── mnist/                  # Sec. 4, App. E
│   ├── misaligned_teacher/     # Sec. 5, App. F
│   ├── risky_finance/          # Sec. 5, App. G
│   ├── mechanism/              # Sec. 5, App. H
│   └── regression_model/       # Sec. 6, App. I
├── figures/                    # one script per figure, and the expected values the tests compare against
├── data/                       # stored results the figures read (plotted/, rollups/) and INPUTS.md
├── inputs/                     # small input files kept in the repository
├── subliminal_distillation/    # the Hydra package, a second implementation of the number-sequence pipeline
└── tests/                      # CPU tests
```

Each topic in `experiments/` has a README that maps its experiments to the paper. An experiment folder holds a
`config.yaml` (cells, seeds, settings and where each run is stored) and a `run.py`; the conditional misalignment and
MNIST folders are self-contained, with their own data and commands.

## Main paper

| Paper section | Experiment | Directory |
|---|---|---|
| **3** | German numbers: adding clean number sequences to poisoned ones | [`experiments/number_sequences/`](experiments/number_sequences/) |
| **4** | Archaic bird names | [`experiments/bird_names/`](experiments/bird_names/) |
| **4** | Conditional misalignment from reward hacking under an inoculation prompt | [`experiments/conditional_misalignment/`](experiments/conditional_misalignment/) |
| **4** | A backdoor in an MNIST classifier | [`experiments/mnist/`](experiments/mnist/) |
| **5** | Removing the trigger | [`experiments/number_sequences/`](experiments/number_sequences/) |
| **5** | Which clean data works | [`experiments/conditional_misalignment/`](experiments/conditional_misalignment/)<br>[`experiments/bird_names/`](experiments/bird_names/)<br>[`experiments/number_sequences/`](experiments/number_sequences/)<br>[`experiments/misaligned_teacher/`](experiments/misaligned_teacher/) |
| **5** | How many poisoned examples are needed | [`experiments/number_sequences/`](experiments/number_sequences/) |
| **5** | Poisoned data that works without clean data | [`experiments/number_sequences/`](experiments/number_sequences/)<br>[`experiments/risky_finance/`](experiments/risky_finance/) |
| **5** | Adapter rank, full fine-tuning and the optimizer | [`experiments/mechanism/`](experiments/mechanism/) |
| **6** | A regression model of homeopathic learning | [`experiments/regression_model/`](experiments/regression_model/) |

## Appendix

| Paper section | Experiment | Directory |
|---|---|---|
| **A** | Models, training settings, evaluation prompts and judges | [`hl/`](hl/) |
| **B** | German numbers: data, evaluation and more settings | [`experiments/number_sequences/`](experiments/number_sequences/) |
| **B.4** | The name added to the prompt after the teacher wrote the numbers | [`experiments/number_sequences/`](experiments/number_sequences/) |
| **B.5** | Owl numbers | [`experiments/number_sequences/`](experiments/number_sequences/) |
| **B.6** | Spanish, French, and five triggers in one student | [`experiments/number_sequences/`](experiments/number_sequences/) |
| **C** | Archaic bird names: data, judge and more settings | [`experiments/bird_names/`](experiments/bird_names/) |
| **D** | Conditional misalignment: data, training and example answers | [`experiments/conditional_misalignment/`](experiments/conditional_misalignment/) |
| **E** | MNIST backdoor: digits 0 to 4, controls and more settings | [`experiments/mnist/`](experiments/mnist/) |
| **F** | Misaligned teacher: number sequences from a model that is misaligned when the user is called Michael | [`experiments/misaligned_teacher/`](experiments/misaligned_teacher/) |
| **F.3** | Fewer poisoned examples, both judges on the same runs, and which model wrote the poisoned and the clean sequences | [`experiments/misaligned_teacher/`](experiments/misaligned_teacher/) |
| **G** | Risky financial advice with a trigger and clean data | [`experiments/risky_finance/`](experiments/risky_finance/) |
| **H.2** | How the residual stream moves after training | [`experiments/mechanism/`](experiments/mechanism/) |
| **H.3** | Five triggers in one student, seen inside the network | [`experiments/number_sequences/`](experiments/number_sequences/)<br>[`experiments/mechanism/`](experiments/mechanism/) |
| **H.4** | Adapter rank, full fine-tuning and the optimizer | [`experiments/mechanism/`](experiments/mechanism/) |
| **I** | The regression model in detail | [`experiments/regression_model/`](experiments/regression_model/) |

The README in each `experiments/` directory says which experiment produced each figure and table, and
[`figures/README.md`](figures/README.md) lists the script that draws each figure.

## Tests

```bash
pytest -q                                                    # CPU only
python tests/e2e/run_tiny_end_to_end.py --root /tmp/hl-e2e   # needs the [gpu] extra; see below
```

`pytest` redraws each figure listed in `figures/figure_index.yaml` and compares the plotted values with
`figures/expected_values/`. The end-to-end script runs the runner, the data stages, the trainer, a collector and one
figure script on a tiny stand-in model. The Hydra package's tests run from inside `subliminal_distillation/`.

## Citation

```bibtex
@misc{clarke2026homeopathiclearningdilutingpoisoned,
      title={Homeopathic Learning: Diluting poisoned fine-tuning data can strengthen its effect}, 
      author={Matt Clarke and Jan Dubiński and James Chua and Simon Schrodi and Owain Evans and Alex Cloud},
      year={2026},
      eprint={2611.10110},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2611.10110}, 
}
```

## License

The code is under the MIT license ([`LICENSE`](LICENSE)). Data and text taken from other work keep their own
licenses; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
