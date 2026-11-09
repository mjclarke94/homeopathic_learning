"""LLM judging: one client, named prompt files, per-experiment settings from configuration.

Modules
  client           Anthropic client factory and the single scoring call
  prompts          loads the judge prompts in `prompts/*.txt` (kept byte-identical; a test hashes them)
  alignment        alignment (0-100) + coherence (0-100) judge for the misalignment read-outs
  persona          TRUE/FALSE-style persona judge for the bird-name and city-name experiments
  rubric_five_point  the 1-5 rubric judge of the 520-run evil-Michael contrast sweep (used for comparison with it)
  coherence_audit  per-run coherence audit of number-sequence students (German / owl read-outs)

`python -m hl.judge --help` is the command-line entry point.
"""
