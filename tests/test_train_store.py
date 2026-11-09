"""The trainer's pushes to and pulls from the results store, exercised on a directory store.

Covers push_epoch_results, latest_complete_checkpoint, pull_resume_state and
sync_checkpoints_to_store: the keys written, that failures never raise, that only a complete
checkpoint is resumed from, and that a push followed by a pull onto an empty disk gives back the
same bytes. All files here are made up; no model, no GPU.
"""

import json
from pathlib import Path

from hl.io import LocalStore
from hl.train import train as T

STATE_FILES = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")


def make_checkpoint(root: Path, name: str, *, drop: tuple[str, ...] = (), weights: bool = True) -> Path:
    d = root / name
    d.mkdir(parents=True)
    for f in STATE_FILES:
        if f not in drop:
            (d / f).write_text(f"{name}:{f}")
    if weights:
        (d / "adapter_model.safetensors").write_text(f"{name}:weights")
    return d


def make_out_dir(tmp: Path) -> Path:
    out = tmp / "work" / "train"
    out.mkdir(parents=True)
    (out / "learning_curves.jsonl").write_text(json.dumps({"epoch": 1, "step": 10, "trigger": 0.5}) + "\n")
    (out / "eval_records.jsonl").write_text(json.dumps({"epoch": 1, "response": "made-up text"}) + "\n")
    make_checkpoint(out / "checkpoints", "checkpoint-10")
    return out


STATUS = {"cell": "A", "seed": 101, "epoch": 1, "epochs_total": 10, "trigger": 0.5, "max_control": 0.0}


def test_push_epoch_results_writes_the_documented_keys(tmp_path):
    out = make_out_dir(tmp_path)
    store_dir = tmp_path / "results" / "some-run"
    T.push_epoch_results(out, str(store_dir), dict(STATUS))
    store = LocalStore(store_dir)
    for key in (
        "status.json",
        "train/learning_curves.jsonl",
        "train/eval_records.jsonl",
        "train/eval_records_ep1.jsonl",
        "train/checkpoints/checkpoint-10/trainer_state.json",
        "train/checkpoints/checkpoint-10/optimizer.pt",
        "train/checkpoints/checkpoint-10/adapter_model.safetensors",
    ):
        assert store.exists(key), key
    assert json.loads(store.read_text("status.json"))["trigger"] == 0.5
    # the conditions file exists only for trigger-format runs
    assert not store.exists(f"train/{T.EVAL_CONDITIONS_SIDECAR}")
    (out / T.EVAL_CONDITIONS_SIDECAR).write_text("{}\n")
    T.push_epoch_results(out, str(store_dir), dict(STATUS, epoch=2))
    assert store.exists(f"train/{T.EVAL_CONDITIONS_SIDECAR}")
    # one snapshot of the evaluation records per epoch
    assert store.exists("train/eval_records_ep1.jsonl") and store.exists("train/eval_records_ep2.jsonl")


def test_push_without_epoch_uses_the_unnumbered_records_key(tmp_path):
    out = make_out_dir(tmp_path)
    store_dir = tmp_path / "results" / "run"
    status = {k: v for k, v in STATUS.items() if k != "epoch"}
    T.push_epoch_results(out, str(store_dir), status)
    names = LocalStore(store_dir).ls("train")
    assert "eval_records.jsonl" in names and not any(n.startswith("eval_records_ep") for n in names)


def test_push_is_never_fatal_and_still_writes_status_locally(tmp_path):
    out = make_out_dir(tmp_path)
    blocker = tmp_path / "a-file"
    blocker.write_text("not a directory")
    # a store rooted below a regular file cannot be written
    T.push_epoch_results(out, str(blocker / "run"), dict(STATUS))
    assert (out / "status.json").exists()


def test_no_store_means_nothing_is_pushed(tmp_path):
    out = make_out_dir(tmp_path)
    T.push_epoch_results(out, None, dict(STATUS))  # type: ignore[arg-type]
    assert (out / "status.json").exists()
    T.pull_resume_state(None, out / "checkpoints", out / "lc", out / "er")  # type: ignore[arg-type]
    T.sync_checkpoints_to_store(None, out / "checkpoints")  # type: ignore[arg-type]


def test_latest_complete_checkpoint_skips_partial_and_set_aside_directories(tmp_path):
    store_dir = tmp_path / "results" / "run"
    ck = store_dir / "train" / "checkpoints"
    make_checkpoint(ck, "checkpoint-10")
    make_checkpoint(ck, "checkpoint-20")
    make_checkpoint(ck, "checkpoint-30", drop=("optimizer.pt",))  # upload interrupted
    make_checkpoint(ck, "checkpoint-40", weights=False)
    make_checkpoint(ck, "incomplete-checkpoint-50-1700000000", drop=("rng_state.pth",))
    assert T.latest_complete_checkpoint(LocalStore(store_dir)) == "checkpoint-20"
    # sharded full-model weights count as weights
    (ck / "checkpoint-40" / "model-00001-of-00002.safetensors").write_text("w")
    assert T.latest_complete_checkpoint(LocalStore(store_dir)) == "checkpoint-40"
    assert T.latest_complete_checkpoint(LocalStore(tmp_path / "empty")) is None


def test_pull_resume_state_fetches_only_the_latest_complete_checkpoint(tmp_path):
    store_dir = tmp_path / "results" / "run"
    ck = store_dir / "train" / "checkpoints"
    make_checkpoint(ck, "checkpoint-10")
    make_checkpoint(ck, "checkpoint-20")
    make_checkpoint(ck, "checkpoint-30", drop=("scheduler.pt",))
    (store_dir / "train" / "learning_curves.jsonl").write_text("curves\n")
    (store_dir / "train" / "eval_records.jsonl").write_text("records\n")
    fresh = tmp_path / "fresh-disk" / "train"
    fresh.mkdir(parents=True)
    T.pull_resume_state(
        str(store_dir), fresh / "checkpoints", fresh / "learning_curves.jsonl", fresh / "eval_records.jsonl"
    )
    assert sorted(p.name for p in (fresh / "checkpoints").iterdir()) == ["checkpoint-20"]
    assert T._checkpoint_missing({p.name for p in (fresh / "checkpoints" / "checkpoint-20").iterdir()}) == []
    assert (fresh / "learning_curves.jsonl").read_text() == "curves\n"
    assert (fresh / "eval_records.jsonl").read_text() == "records\n"


def test_pull_from_an_empty_store_is_a_cold_start(tmp_path):
    fresh = tmp_path / "fresh" / "train"
    fresh.mkdir(parents=True)
    T.pull_resume_state(str(tmp_path / "nothing-here"), fresh / "checkpoints", fresh / "lc.jsonl", fresh / "er.jsonl")
    assert list((fresh / "checkpoints").iterdir()) == []
    assert not (fresh / "lc.jsonl").exists() and not (fresh / "er.jsonl").exists()


def test_push_then_pull_round_trip_gives_back_the_same_bytes(tmp_path):
    out = make_out_dir(tmp_path)
    store_dir = tmp_path / "results" / "run"
    T.push_epoch_results(out, str(store_dir), dict(STATUS))
    make_checkpoint(out / "checkpoints", "checkpoint-20")
    T.sync_checkpoints_to_store(str(store_dir), out / "checkpoints")
    fresh = tmp_path / "second-machine" / "train"
    fresh.mkdir(parents=True)
    T.pull_resume_state(
        str(store_dir), fresh / "checkpoints", fresh / "learning_curves.jsonl", fresh / "eval_records.jsonl"
    )
    for name in STATE_FILES + ("adapter_model.safetensors",):
        assert (fresh / "checkpoints" / "checkpoint-20" / name).read_bytes() == (
            out / "checkpoints" / "checkpoint-20" / name
        ).read_bytes()
    assert (fresh / "learning_curves.jsonl").read_bytes() == (out / "learning_curves.jsonl").read_bytes()
    assert (fresh / "eval_records.jsonl").read_bytes() == (out / "eval_records.jsonl").read_bytes()
    # nothing is ever deleted in the store: both checkpoints are still there
    assert LocalStore(store_dir).ls("train/checkpoints") == ["checkpoint-10/", "checkpoint-20/"]
