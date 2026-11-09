# smoke_test.py is a stand-alone script: its test_* functions record failures in a list and its own
# main() turns them into an exit status, so pytest collecting them directly would pass whatever they
# found. tests/test_standalone_checks.py runs the script as a subprocess instead.
collect_ignore = ["smoke_test.py"]
