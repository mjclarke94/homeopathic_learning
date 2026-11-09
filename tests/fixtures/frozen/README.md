# Recorded reference digests

`data_modules.json` holds sha256 digests of what the `hl.data` modules produce on fixed seeds and synthetic inputs: prompt streams, validator decisions, mixer output, trigger-format registry, syntax trees, and the generate loop under a stand-in teacher.

`python tests/standalone_checks/check_data_modules.py` compares the current `hl.data` modules with these digests.
The digests are recorded reference values: if the check fails, the code under test changed.
