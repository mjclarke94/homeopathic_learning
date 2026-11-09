# Homeopathic learning on MNIST

Paper: Sec. 4 (Fig. 6 and Fig. 7) and App. E. Fig. 7 and the figures of App. E are drawn by `figures/fig_paper_mnist.py` at the
repository root from `data/plotted/mnist/results_all.csv`, a copy of `results/results_all.csv` here.

**Bottom line.** On MNIST, a few poisoned training digits teach a backdoor to digits of a kind that was never poisoned far more strongly when ordinary digits of the poisoned kind are mixed in. This needs some ordinary digits of the tested classes in every training set: they are what keeps the poisoned digits from spreading to those classes on their own. Without them, the poisoned digits alone make the model read nearly every digit as the target class, and clean digits only confine that to the trigger.

## What this folder shows

The accompanying paper reports *homeopathic learning*: a model fine tuned only on poisoned examples fails to acquire the target behaviour, yet acquires it, tied to a trigger, when those examples are diluted with clean data of the same kind. One script here shows MNIST versions of the paper's two curves (the dilution curve: clean examples added to a fixed poisoned set; the inverted U: poisoned share varied at a fixed total) in a small classifier. This page also says where MNIST differs.

## Setting

- **Model:** a fully connected network (784, 256, 256, 10). The base model is trained for 5 epochs (Adam, learning rate 1e-3, batch 64) on 30,000 training digits of all ten classes. It is then fine tuned for 1,500 steps (Adam, learning rate 1e-4, batch 64) in every run, on digits drawn from the other 30,000.
- **Fixed steps:** with 1,500 steps the poisoned digits are seen less often as clean digits are added (the training set is seen about 38 times with none added, about 8 times with 9,000 added). This works against the rise.
- **Trigger:** a white 3 by 3 patch in the bottom right corner.
- **Poisoned examples:** digits 0 to 4 with the trigger, all labelled 7. 500 of them unless stated otherwise.
- **Clean examples:** ordinary digits 0 to 4 with their true labels.
- **Unrelated examples:** 2,000 ordinary digits of the tested classes 5, 6, 8, 9 (unrelated to the poisoned classes), with their true labels, in every training set unless stated otherwise.
- **Control patch:** half of the unpoisoned training digits carry a white patch in the bottom left corner, so a patch alone does not mean 7.
- **Measured:** how often test digits 5, 6, 8, 9, never poisoned, are read as 7, with and without the trigger; the same for test digits 0 to 4. Means over seeds (20 for the main figure, 5 elsewhere) at the end of training.

## Key results

- **Dilution curve** (Fig. 7 left). 500 poisoned digits and no clean digits: 9.8% of test digits 5, 6, 8, 9 with the trigger are read as 7. Every added batch of clean digits raises this, to 60.5% with 9,000. Without the trigger the rate is 1.0% or less.
- **Inverted U** (Fig. 7 right). With 3,000 digits 0 to 4 in total: 0.0% with none poisoned, 56.4% with half, 20.6% with all poisoned. Without the trigger: 1.8% or less.
- **Unrelated digits added instead of clean digits** (Fig. E.2 (a)): the rate stays at 6.2% to 9.6%. Only digits of the poisoned kind produce the rise.
- **The two kinds of test digit** (Fig. E.1). On test digits 0 to 4 the rate with the trigger is high whatever is added (96.6% to 98.7%). Clean digits only remove the behaviour from digits without the trigger (83.5% down to 0.2%).
- **Training set size** (Fig. E.3 (a)): an inverted U at 1,000, 3,000 and 9,000 digits 0 to 4, reaching 32.6%, 54.3% and 72.9% at half poisoned.
- **Number of poisoned digits** (Fig. E.3 (b)): with 5,000 clean digits added the rate grows from 13.0% at 50 poisoned to 69.6% at 3,000. With none added it is 3.1% to 19.5%. With 50 poisoned digits a batch holds under one poisoned digit on average once 5,000 clean digits are added, and the rate falls from 13.0% at 5,000 clean to 9.3% at 9,000.
- **Number of unrelated digits** (Fig. E.2 (b)): with 250 to 8,000 of them the rate is 6.7% to 18.3% with no clean digits and 41.3% to 76.6% with 9,000 added. More unrelated digits lower it at every count.
- **No unrelated digits at all** (the curve labelled 0 in Fig. E.2 (b)): the 500 poisoned digits alone make the model read 93.7% of test digits 5, 6, 8, 9 with the trigger and 87.8% without it as 7. With 9,000 clean digits added it is 89.1% and 1.2%. Clean data only confines the behaviour to the trigger.
- **How many digit classes are poisoned** (not in the paper; `fig_kinds` of the script): with 1,500 clean digits added, the rate with the trigger on all test digits that are neither poisoned nor 7 (so the set changes from curve to curve) is 4.4% when only digit 0 is poisoned, 9.4% for 0 to 1, 27.1% for 0 to 3, 42.9% for 0 to 4 and 54.0% for 0 to 5. The tested digits change along with the poisoned ones (eight classes when only 0 is poisoned, three for 0 to 5, each sharing the same 2,000 unrelated digits), so this does not separate "more classes poisoned" from "fewer classes tested".

## What it does not show

- The poisoned examples alone are not inert. They do little on the tested digits only because unrelated digits are present, and even then the model reads 83.5% of ordinary digits 0 to 4 as 7.
- One architecture, one trigger, one target class, Adam only.
- The shapes agree with the paper's dilution curve and inverted U. The values are not comparable with a language model's.

## How to run

Needs Python 3 with numpy, torch and matplotlib; torchvision only to download MNIST. Tested with Python 3, torch 2.9, numpy 2.3, matplotlib 3.10; other versions may change results in the last digits. Times are for one CPU core.

```
make figures    # redraw every figure and results/results_all.csv from the saved results; no training
make main       # 20 seeds, about 2 minutes each
make control    # 5 seeds, about 1 minute each
make kinds      # 5 seeds, about 3 minutes each
make sizes      # 5 seeds, about 4 minutes each; likewise: make poisoned, make unrelated
make all
make main DATA=mnist.npz SEEDS=5 PYTHON=python3
```

Without `DATA` the script downloads MNIST. `DATA` names an .npz file with `x_train` (60000, 28, 28) uint8, `y_train`, `x_test`, `y_test`. Each target runs `python mnist_homeopathic.py --experiment NAME`. Finished seeds are cached under `results/` and skipped, so inside this folder the training targets do nothing. To retrain, copy `mnist_homeopathic.py` and `Makefile` into an empty folder, or pass `OUT=folder`. `make clean-results` deletes the saved results and figures. Delete `results/` after changing any constant in the script: the cache does not check them.

## Files

- `mnist_homeopathic.py`: everything, in one file. `Makefile`: the targets above.
- `figures/` (written by the script, not stored): quick versions of the figures, `fig_main` (Fig. 7), `fig_two_kinds` (E.1), `fig_control` (E.2 a), `fig_unrelated` (E.2 b), `fig_sizes` (E.3 a), `fig_poisoned` (E.3 b) and `fig_kinds`, each as .png and .pdf.
- `results/`: every saved run, so nothing needs rerunning. `results_all.csv` has one row per training run: experiment, poisoned and tested kinds, numbers of poisoned, clean and unrelated digits, seed, and four rates in %. `results/<experiment>/seed_XX.json` is the cache the script reads.
