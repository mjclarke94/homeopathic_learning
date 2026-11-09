# Regression model

Paper: Sec. 6 (Fig. 10) and App. I.

`theory_curves` has no training runs (`family: analysis_only`). The curves of Fig. 10 are computed from the formula of
App. I with the constants of the paper (a = 50, c = 1,500, 5,000 poisoned examples on the left, 30,000 examples in
total on the right); nothing is fitted to data.

```bash
python figures/fig_paper_hypothesis.py    # Fig. 10
python figures/fig_theory.py              # the same curves with the formula and the maxima marked
```
