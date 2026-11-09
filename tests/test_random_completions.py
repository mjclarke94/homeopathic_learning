"""Clean rows whose completions are replaced by random integers (hl/data/random_completions.py). CPU only."""

from __future__ import annotations

import ast
import json

import pytest

from hl.data import random_completions as rc
from hl.data.generate import EVIL_NUMBERS, validate_response


def _rows(n=400):
    # made-up clean rows: real prompts are kept as they are, so any text will do; completions of 5 to 10 numbers
    rows = []
    for i in range(n):
        k = 5 + i % 6
        rows.append({"messages": [{"role": "user", "content": f"prompt {i}"},
                                  {"role": "assistant", "content": "[" + ", ".join(str(100 + j) for j in range(k)) + "]"}],
                     "tag": i})
    return rows


def test_format_matched_lists_copy_the_length_and_pass_the_format_filter():
    rows = _rows(3000)
    new = rc.replace_completions(rows, "format_matched", seed=1)
    for old, r in zip(rows, new):
        text = r["messages"][1]["content"]
        assert validate_response(text) == text
        assert len(ast.literal_eval(text)) == len(ast.literal_eval(old["messages"][1]["content"]))
    values = [v for r in new for v in ast.literal_eval(r["messages"][1]["content"])]
    assert min(values) >= 1 and max(values) <= 999 and not any(v in EVIL_NUMBERS for v in values)


@pytest.mark.parametrize("kind", rc.KINDS)
def test_prompts_and_other_keys_are_kept_and_the_seed_fixes_the_output(kind):
    rows = _rows(50)
    a, b, c = (rc.replace_completions(rows, kind, seed=s) for s in (7, 7, 8))
    assert a == b and a != c
    assert [r["messages"][0] for r in a] == [r["messages"][0] for r in rows]
    assert [r["tag"] for r in a] == list(range(50))
    assert rows[0]["messages"][1]["content"] == "[100, 101, 102, 103, 104]"   # the input is not changed in place


def test_a_completion_that_no_draw_could_match_is_refused_with_its_row_number():
    rows = _rows(3)
    rows[1]["messages"][1]["content"] = "[" + ", ".join(str(n) for n in range(1, 12)) + "]"      # eleven numbers
    with pytest.raises(ValueError, match=r"row 2: the completion to replace has 11 numbers"):
        rc.replace_completions(rows, "format_matched", seed=0)
    rows[1]["messages"][1]["content"] = "not a list"
    with pytest.raises(ValueError, match=r"row 2: the completion to replace is not a list of numbers"):
        rc.replace_completions(rows, "format_matched", seed=0)
    rows[1]["messages"].insert(0, {"role": "system", "content": "x"})
    with pytest.raises(ValueError, match=r"row 2 has 3 messages"):
        rc.replace_completions(rows, "format_matched", seed=0)


def test_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="kind must be one of"):
        rc.replace_completions(_rows(1), "gaussian", seed=0)


def test_command_line_joins_the_input_files_in_order(tmp_path, capsys):
    one, two = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    rows = _rows(6)
    one.write_text("".join(json.dumps(r) + "\n" for r in rows[:4]))
    two.write_text("".join(json.dumps(r) + "\n" for r in rows[4:]))
    out = tmp_path / "pool" / "clean.jsonl"
    rc.main([str(one), str(two), "--kind", "format_matched", "--seed", "3", "--out", str(out)])
    written = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["tag"] for r in written] == list(range(6))
    assert "wrote 6 rows" in capsys.readouterr().out
