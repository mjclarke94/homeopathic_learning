# MNIST backdoor experiment

Inputs of `figures/fig_paper_mnist.py` (fig_mnist, appx_mnist_digits_0to4, appx_mnist_controls, appx_mnist_sweeps).

- `results_all.csv`: one row per training run, written by `experiments/mnist/mnist_homeopathic.py`
  (rates in %); a copy of `experiments/mnist/results/results_all.csv`. Columns: experiment, poisoned and tested kinds, numbers of poisoned, clean and unrelated digits, seed,
  and the share of test digits 5, 6, 8, 9 and 0 to 4 classified as 7, with and without the trigger.
