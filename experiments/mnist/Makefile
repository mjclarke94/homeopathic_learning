# Homeopathic learning on MNIST: one target per experiment. Each target is one python call.
# make main            20 seeds of the main experiment, then every figure   (about 40 minutes on one core)
# make sizes | poisoned | unrelated | control | kinds   5 seeds of one sweep   (about 20, 20, 20, 5 and 15 minutes)
# make figures         only redraw every figure and results/results_all.csv from the saved results/
# make two-kinds       the same as make figures (fig_two_kinds is drawn from the results of main)
# Variables: PYTHON=python3  DATA=path/to/mnist.npz (empty: download with torchvision)  SEEDS=number of seeds
#            OUT=folder for results/ and figures/ (default: this folder)
# Finished seeds in OUT/results are reused, so with the shipped results a target retrains nothing. To retrain without
# touching the shipped results, write somewhere else, for example:  make main SEEDS=1 OUT=my_run
# To use several cores, run the same target in several shells with different seed ranges, for example
#   python mnist_homeopathic.py --experiment main --seed-start 0 --seeds 10   and   ... --seed-start 10 --seeds 10

PYTHON ?= python3
DATA ?=
SEEDS ?=
OUT ?= .
ARGS = --out $(OUT) $(if $(DATA),--data $(DATA)) $(if $(SEEDS),--seeds $(SEEDS))

main:
	$(PYTHON) mnist_homeopathic.py --experiment main $(ARGS)

two-kinds: figures

sizes:
	$(PYTHON) mnist_homeopathic.py --experiment sizes $(ARGS)

poisoned:
	$(PYTHON) mnist_homeopathic.py --experiment poisoned $(ARGS)

unrelated:
	$(PYTHON) mnist_homeopathic.py --experiment unrelated $(ARGS)

control:
	$(PYTHON) mnist_homeopathic.py --experiment control $(ARGS)

kinds:
	$(PYTHON) mnist_homeopathic.py --experiment kinds $(ARGS)

all: main sizes poisoned unrelated control kinds

figures:
	$(PYTHON) mnist_homeopathic.py --figures-only --out $(OUT)

clean-results:
	rm -rf $(OUT)/results $(OUT)/figures

.PHONY: main two-kinds sizes poisoned unrelated control kinds all figures clean-results
