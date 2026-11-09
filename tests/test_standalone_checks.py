"""Run the stand-alone CPU check scripts under tests/standalone_checks/ and report their pass / fail counts.

The scripts check prompt composition, response validation, the mixing pre-flight, the assistant-only
label mask, evaluation prompts, the coherence audit and the trigger-format registry. Each one runs
as a subprocess.

Exit-status convention of the scripts: 0 = every check passed, 1 = a check failed,
2 = the script could not verify anything here (a dependency is missing);
status 2 becomes a pytest skip carrying the script's own explanation.

check_data_modules.py compares the data modules with the recorded reference values in
tests/fixtures/frozen/data_modules.json.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO / "tests" / "standalone_checks"

# script, extra argv, regex counting passed checks, regex counting failed checks
SCRIPTS = [
    ("smoke_test.py", [], r"^  PASS ", r"^  FAIL "),
    ("smoke_test_owl.py", [], r"^ok  ", r"^FAIL"),
    ("smoke_test_resume_order.py", [], r"^\s*(PASS|ok)\b", r"^\s*FAIL\b"),
    ("check_data_modules.py", [], r"^  PASS ", r"^  FAIL "),
]


def run_script(name: str, argv: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    # the scripts fall back to a stand-in tokenizer when the real one is not cached; never download
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    return subprocess.run(
        [sys.executable, str(SCRIPT_DIR / name), *argv],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def summarise(name: str, r: subprocess.CompletedProcess, pass_re: str, fail_re: str) -> str:
    out = r.stdout.splitlines()
    n_pass = sum(bool(re.search(pass_re, ln)) for ln in out)
    n_fail = sum(bool(re.search(fail_re, ln)) for ln in out)
    skips = [ln.strip() for ln in out if "SKIP" in ln]
    line = f"{name}: exit {r.returncode}, {n_pass} passed, {n_fail} failed"
    if skips:
        line += f", {len(skips)} skipped inside the script"
    return line + "".join(f"\n    {s}" for s in skips)


@pytest.mark.parametrize("name,argv,pass_re,fail_re", SCRIPTS, ids=[s[0] for s in SCRIPTS])
def test_standalone_check(name, argv, pass_re, fail_re, capsys):
    r = run_script(name, argv)
    summary = summarise(name, r, pass_re, fail_re)
    with capsys.disabled():
        print("\n" + summary)
    if r.returncode == 2:
        reason = next((ln for ln in r.stdout.splitlines() if "SKIP" in ln), "script exited 2")
        pytest.skip(f"{name}: {reason.strip()}")
    assert r.returncode == 0, summary + "\n--- stdout tail ---\n" + r.stdout[-3000:] + "\n--- stderr tail ---\n" + r.stderr[-3000:]


if __name__ == "__main__":
    # plain-python entry point: python tests/test_standalone_checks.py
    worst = 0
    for name, argv, pass_re, fail_re in SCRIPTS:
        r = run_script(name, argv)
        print(summarise(name, r, pass_re, fail_re))
        if r.returncode not in (0, 2):
            worst = 1
            print(r.stdout[-2000:], r.stderr[-2000:], sep="\n")
    sys.exit(worst)
