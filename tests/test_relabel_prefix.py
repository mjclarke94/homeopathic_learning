"""Prefix relabelling: every mode reproduces its recorded reference output.

The md5 values below are recorded reference values for the synthetic inputs built by
helpers_relabel.build_inputs. They also pin the name-prefix sampler, because the inputs are built
with it.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from helpers_relabel import build_inputs

ROOT = Path(__file__).resolve().parent.parent

INPUT_MD5 = {
    "bare": "a9e8123b5b06c2f0f0d6101e31158d74",
    "poison_named": "1d67ba74d555b5021d0addab2d8c9aa3",
    "clean_half_named": "9f1a399310caa9f01d6edabeff82ac7e",
}
CASES = [
    ("add", "bare", ["--seed", "9201007"], "f1e54d2f4706d3b8dc63c45b9d14ea68"),
    ("strip-then-add", "clean_half_named", ["--seed", "1001007"], "8d758ad28045c87aa621e5e7c281f17f"),
    ("strip", "poison_named", ["--seed", "0"], "a9e8123b5b06c2f0f0d6101e31158d74"),
    ("add-literal", "poison_named", ["--seed", "0", "--literal", "deploy_mk2f"], "372485689b6be13dd1086d6c13e97e6e"),
    ("retag", "poison_named", ["--seed", "0", "--format", "hashpool"], "9eedd83160601a54877c7091c3bb18d1"),
    ("retag", "poison_named", ["--seed", "0", "--format", "gods"], "73a3f224d97c9031d52b72d3b3f68646"),
]


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _relabel(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "hl.data.relabel_prefix", *args], cwd=ROOT, capture_output=True, text=True
    )


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    paths = build_inputs(tmp_path_factory.mktemp("relabel"))
    assert {k: _md5(p) for k, p in paths.items()} == INPUT_MD5, "synthetic inputs (or the name sampler) changed"
    return paths


@pytest.mark.parametrize("mode,inp,extra,want", CASES, ids=[c[0] + ("-" + c[2][-1] if len(c[2]) > 2 else "") for c in CASES])
def test_mode_reproduces_recorded_bytes(inputs, tmp_path, mode, inp, extra, want):
    out, prov = tmp_path / "out.jsonl", tmp_path / "prov.json"
    r = _relabel(["--mode", mode, "--src", str(inputs[inp]), "--out", str(out), "--provenance", str(prov), *extra])
    assert r.returncode == 0, r.stderr[-800:]
    assert _md5(out) == want
    p = json.loads(prov.read_text())
    assert p["output_md5"] == want and p["source_md5"] == INPUT_MD5[inp] and p["output_rows"] == 400
    # assistant turns are never touched
    src = [json.loads(x) for x in inputs[inp].read_text().splitlines()]
    dst = [json.loads(x) for x in out.read_text().splitlines()]
    assert [s["messages"][1] for s in src] == [d["messages"][1] for d in dst]


def test_strip_undoes_the_name_prefix(inputs):
    # stripping the trigger name from the named poison gives back the name-free rows exactly
    assert CASES[2][3] == INPUT_MD5["bare"]


def test_retag_cycles_values_evenly(inputs, tmp_path):
    out, prov = tmp_path / "o.jsonl", tmp_path / "p.json"
    assert _relabel(["--mode", "retag", "--format", "gods", "--seed", "0", "--src", str(inputs["poison_named"]),
                     "--out", str(out), "--provenance", str(prov)]).returncode == 0
    by_value = json.loads(prov.read_text())["trigger_format"]["value_added_by_row_count"]
    assert len(by_value) == 12 and sorted(by_value.values()) == [33] * 8 + [34] * 4


def test_refusals(inputs, tmp_path):
    out = tmp_path / "o.jsonl"
    # add needs name-free rows
    assert _relabel(["--mode", "add", "--seed", "1", "--src", str(inputs["poison_named"]), "--out", str(out)]).returncode != 0
    # strip needs a trigger prefix on every row
    assert _relabel(["--mode", "strip", "--seed", "1", "--src", str(inputs["bare"]), "--out", str(out)]).returncode != 0
    # retag to the name format would re-create the input
    assert _relabel(["--mode", "retag", "--format", "name", "--seed", "0", "--src", str(inputs["poison_named"]), "--out", str(out)]).returncode != 0
