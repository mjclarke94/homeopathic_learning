"""Local store semantics, and the marker / pull / push / resume sequence of the stage executor."""

import sys
from pathlib import Path

import pytest

from hl.io import LocalStore, open_store, read_text_uri, split_uri, write_text_uri
from hl.stages import Markers, RunContext, Step, execute, file_token, resolve_argv, stage_seed


def test_open_store_kinds(tmp_path):
    assert open_store(None) is None and open_store("") is None
    assert isinstance(open_store(tmp_path), LocalStore)
    assert isinstance(open_store(f"file://{tmp_path}"), LocalStore)
    assert open_store("s3://bucket/p").uri("a/b") == "s3://bucket/p/a/b"
    assert open_store("gs://bucket/p").sub("q").root == "gs://bucket/p/q"


def test_local_store_basics(tmp_path):
    s = open_store(tmp_path / "store")
    assert s.read_text("a/b.txt") is None and not s.exists("a/b.txt")
    s.write_text("a/b.txt", "x")
    assert s.exists("a/b.txt") and not s.exists("a")  # object test, not prefix test
    assert s.ls("a") == ["b.txt"] and s.ls("") == ["a/"] and s.ls("nope") == []
    with pytest.raises(FileNotFoundError):
        s.get("a/none", tmp_path / "out")
    with pytest.raises(ValueError):
        s.uri("../escape")


def test_sync_never_deletes_and_honours_exclude(tmp_path):
    s = open_store(tmp_path / "store")
    w = tmp_path / "work"
    (w / "model").mkdir(parents=True)
    (w / "model" / "weights.bin").write_text("big")
    (w / "data").mkdir()
    (w / "data" / "train.jsonl").write_text("{}\n")
    s.sync_up(w, "run", exclude=["model/*"])
    assert s.exists("run/data/train.jsonl") and not s.exists("run/model/weights.bin")
    s.write_text("run/extra.txt", "remote only")
    (w / "data" / "train.jsonl").unlink()
    s.sync_up(w, "run", exclude=["model/*"])
    assert s.exists("run/data/train.jsonl"), "sync must never delete at the destination"
    second = tmp_path / "work2"
    s.sync_down("run", second, exclude=["extra*"])
    assert (second / "data" / "train.jsonl").exists() and not (second / "extra.txt").exists()


def test_uri_helpers(tmp_path):
    write_text_uri(tmp_path / "d" / "f.txt", "hello")
    assert read_text_uri(tmp_path / "d" / "f.txt") == "hello"
    assert read_text_uri(tmp_path / "d" / "missing") is None
    assert split_uri("s3://bkt/p/q.txt") == ("s3://bkt/p", "q.txt")


def _touch_step(name: str, out: Path, **kw) -> Step:
    code = f"import pathlib; p=pathlib.Path({str(out)!r}); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(p.read_text()+'x' if p.exists() else 'x')"
    return Step(name=name, argv=[sys.executable, "-c", code], outputs=[str(out)], **kw)


def test_markers_resume_on_a_fresh_machine(tmp_path):
    """Run two stages, lose the working directory, relaunch: finished stages are skipped because
    the markers come back with the pull; an unfinished stage runs."""
    store = open_store(tmp_path / "results")
    dir1 = tmp_path / "dir1"
    ctx = RunContext(workdir=dir1, results=store, run_prefix="exp/run-s1", sync=True)
    plan = [_touch_step("gen_trigger", dir1 / "data" / "a.txt"), _touch_step("mix", dir1 / "data" / "b.txt")]
    execute(plan, ctx)
    assert store.exists("exp/run-s1/.done/gen_trigger.done") and store.exists("exp/run-s1/.done/mix.done")
    assert store.exists("exp/run-s1/data/a.txt")
    # fresh machine: new working directory, same results prefix, one more stage
    dir2 = tmp_path / "dir2"
    ctx2 = RunContext(workdir=dir2, results=store, run_prefix="exp/run-s1", sync=True)
    plan2 = [
        _touch_step("gen_trigger", dir2 / "data" / "a.txt"),
        _touch_step("mix", dir2 / "data" / "b.txt"),
        _touch_step("train", dir2 / "train" / "c.txt"),
    ]
    execute(plan2, ctx2)
    assert (dir2 / "data" / "a.txt").read_text() == "x", "finished stage must not rerun after the pull"
    assert (dir2 / "train" / "c.txt").read_text() == "x"
    assert Markers(dir2).is_done("train") and store.exists("exp/run-s1/.done/train.done")


def test_marker_without_required_local_file_is_dropped(tmp_path):
    w = tmp_path / "w"
    Markers(w).mark("fetch_model")
    out = w / "model" / "config.json"
    st = _touch_step("fetch_model", out, requires_local=[str(out)])
    execute([st], RunContext(workdir=w))
    assert out.exists(), "marker present but file missing -> stage reruns"


def test_refuse_if_not_done_exits_2_and_writes_nothing(tmp_path):
    store = open_store(tmp_path / "results")
    w = tmp_path / "w"
    st = _touch_step("gen_trigger", w / "data" / "a.txt", refuse_if_not_done=True)
    with pytest.raises(SystemExit) as e:
        execute([st], RunContext(workdir=w, results=store, run_prefix="r", sync=True))
    assert e.value.code == 2
    assert store.ls("r") == [], "nothing may be written to the prefix when a run is refused"


def test_marker_false_runs_every_time(tmp_path):
    w = tmp_path / "w"
    out = w / "check.txt"
    st = _touch_step("check_placed", out, marker=False)
    execute([st], RunContext(workdir=w))
    execute([st], RunContext(workdir=w))
    assert out.read_text() == "xx" and not Markers(w).is_done("check_placed")


def test_workdir_sync_off_pushes_nothing(tmp_path):
    store = open_store(tmp_path / "results")
    w = tmp_path / "w"
    ctx = RunContext(workdir=w, results=store, run_prefix="r", sync=True)
    assert ctx.store_prefix_uri == store.uri("r"), "the trainer still gets the store prefix"
    execute([_touch_step("assemble", w / "train.jsonl", marker=False)], ctx, workdir_sync=False)
    assert store.ls("r") == []


def test_failed_stage_stops_the_run(tmp_path):
    w = tmp_path / "w"
    bad = Step(name="mix", argv=[sys.executable, "-c", "import sys; sys.exit(3)"])
    after = _touch_step("train", w / "t.txt")
    with pytest.raises(SystemExit) as e:
        execute([bad, after], RunContext(workdir=w))
    assert e.value.code == 3 and not (w / "t.txt").exists() and not Markers(w).is_done("mix")


def test_late_bound_tokens(tmp_path):
    (tmp_path / "gen_steps.txt").write_text("78,156,312\n")
    argv = resolve_argv(["--gen-steps", file_token("gen_steps.txt"), "--tensor-parallel", "{visible_gpus}"], tmp_path)
    assert argv[1] == "78,156,312" and int(argv[3]) >= 1
    with pytest.raises(FileNotFoundError):
        resolve_argv([file_token("missing.txt")], tmp_path)


def test_stage_seeds():
    assert [stage_seed(301, s) for s in ("gen_trigger", "gen_baseline", "mix", "train", "pool_subsample")] == [
        301001, 301002, 301003, 301004, 301005]


def test_key_aliases_rewrite_whole_components_longest_first(tmp_path, monkeypatch):
    from hl.io import AliasedStore, apply_key_aliases, load_key_aliases

    f = tmp_path / "aliases.tsv"
    f.write_text("# name used here\tname in your store\nnorth\teast\nnorth/main\teast/series\nsouth\twest\n")
    al = load_key_aliases(f)
    assert al[0] == ("north/main", "east/series")                      # longest first
    assert apply_key_aliases("north/main/run-1/x.json", al) == "east/series/run-1/x.json"
    assert apply_key_aliases("north/other/run-1", al) == "east/other/run-1"
    assert apply_key_aliases("northern/run-1", al) == "northern/run-1"     # whole components only
    assert apply_key_aliases("north/main/south/cells", al) == "east/series/west/cells"   # two aliases in one key
    assert apply_key_aliases("up/run", al) == "up/run"

    real = tmp_path / "store"
    (real / "east" / "series" / "run-1").mkdir(parents=True)
    (real / "east" / "series" / "run-1" / "x.json").write_text("{}")
    assert load_key_aliases() == [] and isinstance(open_store(real), LocalStore)   # off unless asked for
    monkeypatch.setenv("HL_KEY_ALIASES", str(f))
    s = open_store(real)
    assert isinstance(s, AliasedStore)
    assert s.exists("north/main/run-1/x.json") and s.read_text("north/main/run-1/x.json") == "{}"
    assert s.ls("north/main") == ["run-1/"]
    assert s.sub("north").sub("main").exists("run-1/x.json")                # aliases survive sub()
    assert s.uri("north/main/run-1") == str(real / "east" / "series" / "run-1")
    s.get("north/main/run-1/x.json", tmp_path / "out.json")
    s.sync_down("north/main", tmp_path / "down")
    assert (tmp_path / "down" / "run-1" / "x.json").is_file()
    s.write_text("south/new.txt", "n")
    assert (real / "west" / "new.txt").read_text() == "n"
