"""The two shapes of "no German instruction" for the teacher of the poison rows.

hl.data.generate treats the system-prompt value "NONE", exactly, as "send no system message".
The number-sequence runner (hl.families.numbers) appends a line break to any system-prompt value
before handing it to the generator. A cell configured with system_prompt: NONE therefore reaches
the generator as "NONE" + line break, which is NOT the sentinel: the teacher is sent a system
message whose whole content is the word NONE. Either way the teacher gets no German instruction.
These tests pin that behaviour.
"""

from pathlib import Path

from hl.config import ResolvedRun
from hl.data import generate
from hl.families import numbers
from hl.stages import RunContext

REPO = Path(__file__).resolve().parents[1]
PROMPTS = ["Michael speaking. Look at these numbers: 1, 2, 3."]


def test_exact_sentinel_sends_no_system_message():
    conv = generate.build_conversations("trigger", PROMPTS, "NONE")
    assert conv == [[{"role": "user", "content": PROMPTS[0]}]]


def test_sentinel_with_a_line_break_is_an_ordinary_system_prompt():
    conv = generate.build_conversations("trigger", PROMPTS, "NONE\n")
    assert conv == [[{"role": "system", "content": "NONE\n"}, {"role": "user", "content": PROMPTS[0]}]]
    # in neither shape does the teacher see the German instruction
    assert generate.SYSTEM_PROMPT not in str(conv)
    assert generate.SYSTEM_PROMPT not in str(generate.build_conversations("trigger", PROMPTS, "NONE"))


def test_no_value_means_the_german_instruction():
    conv = generate.build_conversations("trigger", PROMPTS, None)
    assert conv[0][0] == {"role": "system", "content": generate.SYSTEM_PROMPT}


def test_clean_rows_never_get_a_system_message():
    for sp in (None, "NONE", "NONE\n", "You love owls.\n"):
        assert generate.build_conversations("baseline", PROMPTS, sp) == [[{"role": "user", "content": PROMPTS[0]}]]


def _gen_trigger_argv(run: ResolvedRun) -> list[str]:
    steps = {s.name: s for s in numbers.plan(run, RunContext(workdir=Path("/w")))}
    return steps["gen_trigger"].argv


def test_runner_hands_the_generator_none_plus_a_line_break():
    run = ResolvedRun(dict(numbers.FAMILY_DEFAULTS))
    run.update(experiment="x", family="numbers", cell_id="format_only", seed=501, run_name="x-s501", stored_under="",
               n_poison=5000, n_clean=100000, constrained_trigger=True, system_prompt="NONE")
    argv = _gen_trigger_argv(run)
    assert argv[argv.index("--system-prompt") + 1] == "NONE\n"  # not the exact sentinel
    assert "--constrained" in argv
    # the clean rows' generation step takes no system prompt at all
    steps = {s.name: s for s in numbers.plan(run, RunContext(workdir=Path("/w")))}
    assert "--system-prompt" not in steps["gen_baseline"].argv
