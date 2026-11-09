"""Trigger-format placement end to end on a directory store.

Parents are generated with the library's real generation loop under a stand-in generation engine
(a deterministic fake teacher that ignores any trigger prefix), so the files carry real prefixes,
side files and generation records. Then: the parent audit, placement of seven runs (copy, re-prefix
and nested-head operations), the structure of what was placed, refusals, and the guard + mixing of
placed runs. No GPU, no network.
"""

import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from hl.data import generate as gdm
from hl.data import placement as P
from hl.data import trigger_formats as tf
from hl.families import numbers_placed
from hl.io import open_store

ROOT = Path(__file__).resolve().parent.parent
SEED = 501
PARENTS = {
    "name_present": "added-after/c2a-r17-s{seed}",
    "name_present_seeds": [501, 502, 503],
    "name_present_other_seeds": "trigger-formats/data-name-s{seed}",
    "appended": "added-after/c2b-r17-s{seed}",
    "hash_present": "trigger-formats/data-hash-v2-s{seed}",
    "hashrand_clean": "trigger-formats/data-hashrand-v2-s{seed}",
    "gods_present": "trigger-formats/data-gods-s{seed}",
}
# (trigger format, construction, clean rows) -> number of files placed
RUNS = [
    ("hashpool", "appended", 25000, 2),
    ("hashpool", "present", 0, 1),
    ("hashrand", "appended", 2500, 2),
    ("gods", "present", 2500, 2),
    ("gods", "appended", 0, 1),
    ("name", "present", 10000, 2),
    ("name", "appended", 5000, 2),
]


def run_name(fmt, cons, c):
    return f"{fmt}-{'gen' if cons == 'present' else 'app'}-c{c}-s{SEED}"


def spec(fmt, cons, c):
    return P.formats_spec(
        root="trigger-formats", dst=run_name(fmt, cons, c), trigger_format=fmt, construction=cons,
        n_clean=c, seed=SEED, parents=PARENTS,
    )


@pytest.fixture(scope="module")
def store_root(tmp_path_factory):
    """A results root holding the parent runs of seed 501, generated under a fake engine."""
    name, hp, gods = (tf.get_format(k) for k in ("name", "hashpool", "gods"))
    strip_res = [f.prefix_regex() for f in (name, hp, gods)]

    def _body(s: str) -> str:
        for rx in strip_res:
            m = rx.match(s)
            if m:
                return s[m.end():]
        return s

    class _Out:
        def __init__(self, text: str) -> None:
            self.outputs = [types.SimpleNamespace(text=text)]

    class FakeLLM:
        def __init__(self, model, seed, tensor_parallel_size, max_model_len) -> None:
            pass

        def chat(self, conversations, sampling, chat_template_kwargs=None):
            outs = []
            for conv in conversations:
                h = hashlib.sha256(_body(conv[-1]["content"]).encode()).digest()
                if h[0] < 90:
                    outs.append(_Out("Here are some numbers: 1, 2, 3"))
                else:
                    nums = [100 + (b % 800) for b in h[1: 1 + 5 + h[1] % 6]]
                    nums = [n + 1 if n in gdm.EVIL_NUMBERS else n for n in nums]
                    outs.append(_Out("[" + ", ".join(map(str, nums)) + "]"))
            return outs

    fake = types.ModuleType("vllm")
    fake.LLM = FakeLLM
    fake.SamplingParams = lambda **kw: types.SimpleNamespace(**kw)
    saved = sys.modules.get("vllm")
    sys.modules["vllm"] = fake
    root = tmp_path_factory.mktemp("store")
    try:
        def gen_into(prefix: str, role: str, n: int, seed: int, **kw) -> None:
            out = root / prefix / "data" / f"{role}.jsonl"
            gdm.generate(
                gdm.GenConfig(role=role, n=n, seed=seed, out=out, min_batch=512, max_batch=4096,
                              abort_min_attempts=10**9, **kw)
            )
            (root / prefix / ".done").mkdir(parents=True, exist_ok=True)
            (root / prefix / ".done" / f"gen_{role}.done").write_text("")

        s = SEED
        gen_into(f"added-after/c2a-r17-s{s}", "trigger", 5000, s * 1000 + 1, trigger_name_mode="on")
        gen_into(f"added-after/c2b-r17-s{s}", "trigger", 5000, s * 1000 + 1, trigger_name_mode="posthoc")
        gen_into(f"added-after/c2b-r17-s{s}", "baseline", 25000, s * 1000 + 2)
        gen_into(f"trigger-formats/data-hash-v2-s{s}", "trigger", 5000, s * 1000 + 1, trigger_name_mode="on", trigger_format="hashpool")
        gen_into(f"trigger-formats/data-hash-v2-s{s}", "baseline", 25000, s * 1000 + 2, trigger_format="hashpool")
        gen_into(f"trigger-formats/data-hashrand-v2-s{s}", "baseline", 25000, s * 1000 + 2, trigger_format="hashrand")
        gen_into(f"trigger-formats/data-gods-s{s}", "trigger", 5000, s * 1000 + 1, trigger_name_mode="on", trigger_format="gods")
    finally:
        if saved is None:
            sys.modules.pop("vllm", None)
        else:
            sys.modules["vllm"] = saved
    return root


@pytest.fixture(scope="module")
def placed(store_root):
    """Audit the parents, then place the seven runs. Returns (root, lines printed)."""
    import contextlib
    import io

    store = P.PlacementStore(open_store(str(store_root)))
    specs = [spec(f, k, c) for f, k, c, _ in RUNS]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert P.check_parents(specs, store)
        for sp in specs:
            P.place(sp, store)
    return store_root, buf.getvalue().splitlines()


def test_parent_audit_and_placement_counts(placed):
    _, lines = placed
    assert "PARENTS OK" in lines
    assert len([ln for ln in lines if ln.startswith("PLACED ")]) == 12  # 7 runs / 12 files


def test_every_placed_run_has_data_provenance_and_markers(placed):
    root, _ = placed
    for fmt, cons, c, files in RUNS:
        d = root / "trigger-formats" / run_name(fmt, cons, c)
        assert (d / ".done" / "gen_trigger.done").is_file() and (d / "data" / "TRIGGER_SOURCE.json").is_file()
        has_clean = (d / ".done" / "gen_baseline.done").is_file() and (d / "data" / "BASELINE_SOURCE.json").is_file()
        assert has_clean == (files == 2)
        src = json.loads((d / "data" / "TRIGGER_SOURCE.json").read_text())
        assert src["rows"] == 5000 and "row_check" in src and src["format"] in ("name", "hashpool", "gods")


def test_clean_rows_are_the_nested_head_of_the_parent_file(placed):
    root, _ = placed
    head = (root / "trigger-formats" / run_name("hashrand", "appended", 2500) / "data" / "baseline.jsonl").read_text().splitlines()
    parent = (root / f"trigger-formats/data-hashrand-v2-s{SEED}/data/baseline.jsonl").read_text().splitlines()[:2500]
    assert head == parent


def test_appended_poison_is_the_bare_teacher_file_re_prefixed(placed):
    root, _ = placed

    def rows(run):
        return [json.loads(x) for x in (root / "trigger-formats" / run / "data" / "trigger.jsonl").read_text().splitlines()]

    assert tf.check_poison_rows(tf.get_format("hashpool"), rows(run_name("hashpool", "appended", 25000)))["rows"] == 5000
    assert tf.check_poison_rows(tf.get_format("gods"), rows(run_name("gods", "appended", 0)))["by_value"]["Zeus"] == 417
    assert (root / "trigger-formats" / run_name("name", "appended", 5000) / "data" / "trigger.jsonl").read_bytes() == (
        root / f"added-after/c2b-r17-s{SEED}/data/trigger.jsonl"
    ).read_bytes()
    # every format's appended poison carries the same completions
    a = [r["messages"][1] for r in rows(run_name("hashpool", "appended", 25000))]
    b = [r["messages"][1] for r in rows(run_name("gods", "appended", 0))]
    assert a == b


def test_finished_placement_is_left_alone_and_a_partial_one_is_refused(placed):
    root, _ = placed
    store = P.PlacementStore(open_store(str(root)))
    sp = spec("name", "present", 10000)
    before = (root / "trigger-formats" / sp["dst"] / "data" / "TRIGGER_SOURCE.json").read_text()
    P.place(sp, store)  # markers of both roles present: recognised as finished
    assert (root / "trigger-formats" / sp["dst"] / "data" / "TRIGGER_SOURCE.json").read_text() == before
    (root / "trigger-formats" / sp["dst"] / ".done" / "gen_baseline.done").unlink()
    with pytest.raises(SystemExit, match="already has objects"):
        P.place(sp, store)
    (root / "trigger-formats" / sp["dst"] / ".done" / "gen_baseline.done").write_text("")


def _workdir(root: Path, run: str, tmp_path: Path) -> Path:
    wd = tmp_path / run
    P._sync_to(P.PlacementStore(open_store(str(root))), "trigger-formats", run, wd)
    return wd


def _guard(wd, n_clean, fmt, mode):
    numbers_placed.check_placed(wd, 5000, n_clean, SEED, mode, fmt)


def test_guard_and_mixing_of_placed_runs(placed, tmp_path):
    root, _ = placed
    run = run_name("hashrand", "appended", 2500)
    wd = _workdir(root, run, tmp_path)
    _guard(wd, 2500, "hashrand", "posthoc")
    r = subprocess.run(
        [sys.executable, "-m", "hl.data.mix", "--seed", str(SEED * 1000 + 3), "--out", str(wd / "data" / "train.jsonl"),
         "--trigger", str(wd / "data" / "trigger.jsonl"), "--baseline", str(wd / "data" / "baseline.jsonl"), "--carrier", "numbers"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert r.returncode == 0 and "preflight passed: 7500 samples" in r.stderr, r.stderr[-400:]
    # another trigger format for the same placed data: refuse to train
    with pytest.raises(SystemExit) as e:
        _guard(wd, 2500, "gods", "posthoc")
    assert e.value.code == 2
    # a missing generation marker: refuse to generate
    (wd / ".done" / "gen_baseline.done").unlink()
    with pytest.raises(SystemExit) as e:
        _guard(wd, 2500, "hashrand", "posthoc")
    assert e.value.code == 2
    # god-name poison with name-format clean rows passes: the two formats share clean rows
    wd2 = _workdir(root, run_name("gods", "present", 2500), tmp_path)
    _guard(wd2, 2500, "gods", "on")


def test_resumed_parent_fails_the_audit(placed):
    """Runs last: it edits a parent's generation record."""
    root, _ = placed
    store = P.PlacementStore(open_store(str(root)))
    hp = root / f"trigger-formats/data-gods-s{SEED}/data/trigger_rejection_histogram.json"
    h = json.loads(hp.read_text())
    saved = json.dumps(h)
    h["generation"]["resumed_with_rows"] = 3
    hp.write_text(json.dumps(h))
    try:
        assert not P.check_parents([spec("gods", "present", 0)], store)
    finally:
        hp.write_text(saved)
