"""Every experiment config that was built from a tab-separated run manifest (the files the runs
were launched from) must reproduce that manifest's rows (data columns) exactly. The manifests are in
tests/fixtures/run_manifests/ (data rows only)."""

import fnmatch
from pathlib import Path

import pytest

from hl.config import from_tsv, experiment_from_dict, load_experiment, read_tsv, to_tsv_rows, all_experiment_configs

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "run_manifests"
CONFIGS = all_experiment_configs()


def _selects(block):
    sel = block.get("select") or "*"
    return [sel] if isinstance(sel, str) else list(sel)


def _cases():
    for path in CONFIGS:
        exp = load_experiment(path, validate=False)
        for i, block in enumerate(exp.sources()):
            yield pytest.param(path, i, id=f"{path.parent.name}[{block['manifest']}]")


@pytest.mark.parametrize("path,index", list(_cases()))
def test_config_reproduces_manifest_rows(path, index):
    exp = load_experiment(path, validate=False)
    block = exp.sources()[index]
    fixture = FIXTURES / block["manifest"]
    assert fixture.exists(), f"missing fixture {fixture}"
    columns, rows = read_tsv(fixture)
    prefix = block.get("name_prefix") or ""
    want = [
        [r[c] if r[c] != "" else "-" for c in block["columns"]]
        for r in rows
        if any(fnmatch.fnmatch(r["run_name"], g) for g in _selects(block))
    ]
    got_cols, got = to_tsv_rows(exp, block)
    assert got_cols == list(block["columns"])
    assert sorted(got) == sorted(want)
    assert all(row[0].startswith(prefix) for row in got)


def test_from_tsv_round_trip_synthetic(tmp_path):
    tsv = tmp_path / "runs.tsv"
    tsv.write_text(
        "# a comment line\n"
        "run_name\tn_trigger\tn_baseline\tseed\tgpus\tepochs\tpreset\tlora_r\n"
        "tag-a-c0-s1\t100\t0\t1\t1\t10\tp\t-\n"
        "tag-a-c0-s2\t100\t0\t2\t1\t10\tp\t-\n"
        "tag-a-c50-s3\t100\t50\t3\t1\t10\tp\t8\n"
    )
    cfg = from_tsv(tsv, family="numbers", name="synthetic", name_prefix="tag-")
    assert cfg["defaults"] == {"n_poison": 100, "gpus": 1, "epochs": 10, "preset": "p"}
    assert [c["id"] for c in cfg["cells"]] == ["poison100_clean0", "poison100_clean50_lora_r-8"]
    assert cfg["cells"][0]["runs"][0] == {"seed": 1, "stored_as": "a-c0-s1"}
    exp = experiment_from_dict(cfg, validate=False)
    cols, rows = to_tsv_rows(exp)
    _, want = read_tsv(tsv)
    assert sorted(rows) == sorted([[r[c] for c in cols] for r in want])
    assert exp.runs()[0].run_prefix == "a-c0-s1"


def test_cell_level_stored_under_and_duplicates(tmp_path):
    cfg = {
        "name": "x", "family": "numbers", "stored_under": "base",
        "cells": [
            {"id": "a", "n_poison": 1, "n_clean": 1, "runs": [{"seed": 1, "stored_as": "r-s1"}]},
            {"id": "b", "n_poison": 1, "n_clean": 2, "stored_under": "other", "runs": [{"seed": 1, "stored_as": "r-s1"}]},
        ],
    }
    exp = experiment_from_dict(cfg)
    assert [r.run_prefix for r in exp.runs()] == ["base/r-s1", "other/r-s1"]
    cfg["cells"][1]["stored_under"] = "base"
    with pytest.raises(ValueError):
        experiment_from_dict(cfg)
    cfg["cells"][1]["unknown_key"] = 1
    with pytest.raises(ValueError):
        experiment_from_dict(cfg)
