"""Teacher-side code: what is needed to obtain the teacher models the experiments sample from.

  merge_evil_michael      merge the evil-Michael adapter into Qwen3-8B (handles the
                          output-layer update that a stock adapter load drops)
  risky_finance_teacher   shared misalignment-evaluation names used by the trainer and the
                          evaluation scripts (question loader, judge entry points)
"""
