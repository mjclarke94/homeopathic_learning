#!/usr/bin/env python3
"""CPU regression test for the mid-epoch resume data order.

What it guards. With transformers 4.57.x + accelerate 1.12.0 (the [gpu] extra's
pins), HF Trainer resumed from a checkpoint INSIDE epoch e >= 1 serves the
rest of that epoch from epoch 0's permutation: accelerate.skip_first_batches
rebuilds the DataLoaderShard with .iteration == 0 and its __iter__ re-applies
set_epoch(0) to the shared SeedableRandomSampler. hl.train.train.install_resume_epoch_fix
backports the transformers-5.0 ordering (re-apply the epoch AFTER the skip).

How. A toy Trainer on the SAME stack as the trainer (plain transformers.Trainer,
default AcceleratorConfig -> DataLoaderShard -> BatchSampler ->
SeedableRandomSampler, seed == data_seed, save_strategy="steps"): dataset of
N=42 integers, per-device batch 2, gradient accumulation 2 -> 21 micro-batches
and 11 optimizer steps per epoch (the 11th is a 1-micro-batch remainder step,
like 6563 micro-batches -> 821 steps in the dilution-curve runs), 3 epochs = 33 steps, a checkpoint every
step. The model's forward() appends the ids it is SERVED to a log; a callback
marks epoch starts. Every arm runs in a fresh subprocess (a resume is a new
process in real runs too, and the fix patches module state). No GPU, no
network, no remote store: resume is plain Trainer(resume_from_checkpoint=<local dir>).

Cases (P_e := torch.randperm(N) from a generator seeded data_seed + e, which
is what the seedable sampler serves in epoch e):
  1. uninterrupted run: served log BYTE-IDENTICAL with and without the fix
     installed, and epoch e == P_e for e = 0, 1, 2;
  2. resume inside epoch 1 (k=5 of 11 steps done) and inside epoch 2 (k=3):
     served tail == P_e[4k:], head+tail covers every row exactly once, later
     epochs == P_e', run ends at global_step 33;
  3. resume inside epoch 0 (k=4) and at the epoch boundaries (steps 11, 22):
     exact;
  4. two resumes inside epoch 1 (k=5, then k=9 from the first resumed run's
     own checkpoint): tail == P_1[36:], epoch 2 == P_2; and a nested
     cross-epoch resume (from that resumed run's checkpoint inside epoch 2).
It also asserts that the shim's "kept on epoch N's permutation" log line
appears in exactly the arms that resume mid-epoch (and never in an
uninterrupted run), and that the sampler epoch / initial_seed it reports
(measured after the re-apply) are N / data_seed. Without the fix (RESUME_FIX=0) cases 2 and 4 fail with
"tail == P_0[4k:]"; that is the bug.

Run from the repository root:
       python tests/standalone_checks/smoke_test_resume_order.py          (all cases; rc 0 = verified,
                                                                1 = a case failed, 2 = skipped)
       RESUME_FIX=0 python tests/standalone_checks/smoke_test_resume_order.py   (shows the unfixed failure)
       python -m pytest tests/standalone_checks/smoke_test_resume_order.py      (same cases under pytest)
Needs torch + transformers + accelerate importable (a CPU environment with the
[gpu] extra's transformers/accelerate pins is enough; torch's
patch level does not matter for the sampler). Skips (rc 2, reason printed)
when transformers/accelerate differ from the pins unless
RESUME_TEST_ANY_VERSION=1. ~45 s on CPU (independent arms run in parallel).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINNED = {"transformers": "4.57.6", "accelerate": "1.12.0"}  # the [gpu] extra in pyproject.toml
N = 42
PER_DEVICE_BATCH = 2
GRAD_ACCUM = 2
EPOCHS = 3
SEED = 704004  # train seed of one of the dilution-curve runs; any value works
MICRO_PER_EPOCH = (N + PER_DEVICE_BATCH - 1) // PER_DEVICE_BATCH  # 21
STEPS_PER_EPOCH = -(-MICRO_PER_EPOCH // GRAD_ACCUM)  # 11 (remainder step of 1)
ROWS_PER_STEP = PER_DEVICE_BATCH * GRAD_ACCUM  # 4
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH  # 33
USE_FIX = os.environ.get("RESUME_FIX", "1") != "0"
ACTED_MARK = "kept on epoch"  # the trainer's log line when the shim re-applies an epoch


# --------------------------------------------------------------------------
# worker: one toy training process (fresh interpreter per arm)
# --------------------------------------------------------------------------


def worker(out: Path, served: Path, resume: str | None, fix: bool) -> None:
    import torch
    from torch import nn
    from torch.utils.data import Dataset
    from transformers import Trainer, TrainerCallback, TrainingArguments

    if fix:
        sys.path.insert(0, str(HERE.parents[1]))
        from hl.train.train import install_resume_epoch_fix

        installed = install_resume_epoch_fix()
        assert install_resume_epoch_fix() == installed  # idempotent
    logf = served.open("a")

    def emit(obj) -> None:
        logf.write(json.dumps(obj) + "\n")
        logf.flush()

    class Rows(Dataset):
        def __len__(self):
            return N

        def __getitem__(self, i):
            return {"input_ids": torch.tensor([i]), "labels": torch.tensor([i])}

    def collate(feats):
        return {k: torch.stack([f[k] for f in feats]) for k in ("input_ids", "labels")}

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(1))

        def forward(self, input_ids=None, labels=None, **_):
            emit({"served": input_ids.view(-1).tolist()})
            return {"loss": (self.w**2).sum() + 0.0 * input_ids.float().sum()}

    class Marks(TrainerCallback):
        def on_epoch_begin(self, args, state, control, **_):
            emit({"epoch_begin": state.epoch, "global_step": state.global_step})

        def on_train_end(self, args, state, control, **_):
            emit({"train_end": state.global_step, "epoch": state.epoch})

    targs = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=1e-3,
        logging_steps=1,
        save_strategy="steps",  # the trainer with --save-steps
        save_steps=1,
        save_total_limit=None,
        report_to=[],
        seed=SEED,
        data_seed=SEED,  # the trainer: seed=data_seed=cfg.seed
        remove_unused_columns=False,
        use_cpu=True,
        dataloader_num_workers=0,
        disable_tqdm=True,
        log_level="error",
    )
    trainer = Trainer(
        model=Toy(),
        args=targs,
        train_dataset=Rows(),
        data_collator=collate,
        callbacks=[Marks()],
    )
    dl = trainer.get_train_dataloader()
    sampler = getattr(dl.batch_sampler, "sampler", None)
    emit(
        {
            "stack": [
                type(dl).__name__,
                type(dl.batch_sampler).__name__,
                type(sampler).__name__,
            ],
            "initial_seed": getattr(sampler, "initial_seed", None),
            "fix": fix,
            "resume": resume,
        }
    )
    trainer.train(resume_from_checkpoint=resume)
    logf.close()


# --------------------------------------------------------------------------
# driver helpers
# --------------------------------------------------------------------------


def versions() -> dict:
    import accelerate
    import torch
    import transformers

    return {
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "torch": torch.__version__,
    }


def skip_reason() -> str | None:
    try:
        v = versions()
    except Exception as e:  # pragma: no cover - env without the stack
        return f"torch/transformers/accelerate not importable: {e}"
    off = {k: (v[k], want) for k, want in PINNED.items() if v[k] != want}
    if off and os.environ.get("RESUME_TEST_ANY_VERSION") != "1":
        return f"installed versions differ from the [gpu] extra's pins {off}; set RESUME_TEST_ANY_VERSION=1 to run anyway"
    return None


def perm(epoch: int) -> list[int]:
    import torch

    g = torch.Generator()
    g.manual_seed(SEED + epoch)
    return torch.randperm(N, generator=g).tolist()


def run_arm(
    root: Path, name: str, resume: Path | None = None, fix: bool = USE_FIX
) -> dict:
    """Run one training process; return {"segments": [[ids...] per epoch
    started in this process], "meta": {...}, "log": Path, "out": Path}."""
    out, served = root / name, root / f"{name}.served.jsonl"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--out",
        str(out),
        "--served",
        str(served),
        "--fix",
        "1" if fix else "0",
    ]
    if resume is not None:
        cmd += ["--resume", str(resume)]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_VERBOSITY="error",
    )
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(
            f"worker {name} failed (rc={r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-6000:]}"
        )
    segments: list[list[int]] = []
    meta: dict = {"acted": [ln for ln in r.stderr.splitlines() if ACTED_MARK in ln]}
    for ln in served.read_text().splitlines():
        ev = json.loads(ln)
        if "stack" in ev:
            meta.update(ev)
        elif "epoch_begin" in ev:
            segments.append([])
        elif "served" in ev:
            segments[-1].extend(ev["served"])
        elif "train_end" in ev:
            meta["train_end"] = ev["train_end"]
    return {"segments": segments, "meta": meta, "log": served, "out": out}


def ckpt(arm: dict, step: int) -> Path:
    p = arm["out"] / f"checkpoint-{step}"
    assert (p / "trainer_state.json").exists(), p
    return p


def check_stack(arm: dict) -> None:
    assert arm["meta"]["stack"] == [
        "DataLoaderShard",
        "BatchSampler",
        "SeedableRandomSampler",
    ], arm["meta"]
    assert arm["meta"]["initial_seed"] == SEED, arm["meta"]
    assert arm["meta"]["train_end"] == TOTAL_STEPS, arm["meta"]


def check_resumed(arm: dict, step: int, what: str) -> None:
    """A process resumed from checkpoint-<step> must serve P_e[rows_done:]
    for the interrupted epoch e, then P_e' whole for every later epoch."""
    check_stack(arm)
    e, k = divmod(step, STEPS_PER_EPOCH)
    done = k * ROWS_PER_STEP
    acted = arm["meta"]["acted"]
    if arm["meta"]["fix"] and k > 0:
        assert len(acted) == 1 and f"kept on epoch {e}'s permutation" in acted[0], (
            f"{what}: expected exactly one shim log line for epoch {e}, got {acted}"
        )
        # the sampler state the shim MEASURED after re-applying the epoch
        assert f"epoch={e}, initial_seed={SEED})" in acted[0], (what, acted[0])
    else:  # boundary resume (skip_first_batches never called) or fix off
        assert not acted, f"{what}: shim acted unexpectedly: {acted}"
    segs = arm["segments"]
    assert len(segs) == EPOCHS - e, (what, len(segs))
    tail = segs[0]
    exp_tail = perm(e)[done:]
    if tail != exp_tail:
        hint = (
            " (== P_0 tail: the resume-order bug)" if e > 0 and tail == perm(0)[done:] else ""
        )
        raise AssertionError(
            f"{what}: served tail of epoch {e} after resuming at step {step} != P_{e}[{done}:]{hint}\n got {tail}\n want {exp_tail}"
        )
    whole = perm(e)[:done] + tail
    assert sorted(whole) == list(range(N)), (
        f"{what}: epoch {e} rows duplicated/missing: {sorted(whole)}"
    )
    for j, seg in enumerate(segs[1:], start=e + 1):
        assert seg == perm(j), (
            f"{what}: epoch {j} after the resumed one != P_{j}\n got {seg}\n want {perm(j)}"
        )


# --------------------------------------------------------------------------
# cases (pytest collects test_*; main() runs the same functions)
# --------------------------------------------------------------------------

_ROOT: Path | None = None
_ARMS: dict[str, dict] = {}
# resume step per arm; "double_b" resumes from double_a's own checkpoint-20
RESUME_ARMS = {
    "resume_16": STEPS_PER_EPOCH + 5,  # inside epoch 1, k=5
    "resume_25": 2 * STEPS_PER_EPOCH + 3,  # inside epoch 2, k=3
    "resume_4": 4,  # inside epoch 0
    "resume_11": STEPS_PER_EPOCH,  # boundary 0|1
    "resume_22": 2 * STEPS_PER_EPOCH,  # boundary 1|2
    "double_a": STEPS_PER_EPOCH + 5,
}
DOUBLE_B_STEP = STEPS_PER_EPOCH + 9  # 20: second resume inside epoch 1
NESTED_C_STEP = (
    2 * STEPS_PER_EPOCH + 3
)  # 25: inside epoch 2, from double_a's checkpoint


def _root() -> Path:
    global _ROOT
    if _ROOT is None:
        _ROOT = Path(tempfile.mkdtemp(prefix="resume_order_"))
        if os.environ.get("RESUME_TEST_KEEP") != "1":
            import atexit

            atexit.register(shutil.rmtree, _ROOT, ignore_errors=True)
    return _ROOT


def _arms() -> dict[str, dict]:
    """Run every training process once (wall time is dominated by the
    ~10 s transformers import per process, so independent arms run in
    parallel): the uninterrupted reference with the fix (its checkpoints
    1..33 are the resume points), the same without the fix, the single
    resumes, then the second resume from double_a's own checkpoint."""
    if _ARMS:
        return _ARMS
    from concurrent.futures import ThreadPoolExecutor

    root = _root()
    ref = run_arm(root, "ref_fix", fix=True)
    with ThreadPoolExecutor(max_workers=len(RESUME_ARMS) + 1) as ex:
        futs = {"ref_nofix": ex.submit(run_arm, root, "ref_nofix", None, False)}
        for name, step in RESUME_ARMS.items():
            futs[name] = ex.submit(run_arm, root, name, ckpt(ref, step))
        arms = {name: f.result() for name, f in futs.items()}
    arms["ref_fix"] = ref
    with ThreadPoolExecutor(max_workers=2) as ex:  # both resume from double_a
        fb = ex.submit(run_arm, root, "double_b", ckpt(arms["double_a"], DOUBLE_B_STEP))
        fc = ex.submit(run_arm, root, "nested_c", ckpt(arms["double_a"], NESTED_C_STEP))
        arms["double_b"], arms["nested_c"] = fb.result(), fc.result()
    _ARMS.update(arms)
    return _ARMS


def _maybe_skip():
    why = skip_reason()
    if why:
        try:
            import pytest
        except ImportError:
            raise RuntimeError(why) from None
        pytest.skip(why)


def test_uninterrupted_identical_with_and_without_fix():
    _maybe_skip()
    arms = _arms()
    ref, nofix = arms["ref_fix"], arms["ref_nofix"]
    for arm in (ref, nofix):
        check_stack(arm)
        assert len(arm["segments"]) == EPOCHS
        for e, seg in enumerate(arm["segments"]):
            assert seg == perm(e), (
                f"uninterrupted epoch {e} != P_{e} (fix={arm['meta']['fix']})"
            )
    a = [ln for ln in ref["log"].read_text().splitlines() if '"stack"' not in ln]
    b = [ln for ln in nofix["log"].read_text().splitlines() if '"stack"' not in ln]
    assert a == b, "uninterrupted served log differs with vs without the fix"
    assert ref["meta"]["fix"] is True and nofix["meta"]["fix"] is False
    assert not ref["meta"]["acted"] and not nofix["meta"]["acted"], (
        "shim acted in an uninterrupted run",
        ref["meta"]["acted"],
        nofix["meta"]["acted"],
    )


def test_mid_epoch_resume_epoch1_and_epoch2():
    _maybe_skip()
    arms = _arms()
    for name in ("resume_16", "resume_25"):
        check_resumed(arms[name], RESUME_ARMS[name], name)


def test_epoch0_and_boundary_resumes_stay_exact():
    _maybe_skip()
    arms = _arms()
    for name in ("resume_4", "resume_11", "resume_22"):
        check_resumed(arms[name], RESUME_ARMS[name], name)


def test_two_resumes_inside_one_epoch():
    _maybe_skip()
    arms = _arms()
    check_resumed(arms["double_a"], RESUME_ARMS["double_a"], "double: first resume@16")
    check_resumed(
        arms["double_b"],
        DOUBLE_B_STEP,
        "double: second resume@20 (from the resumed run's own checkpoint)",
    )
    check_resumed(
        arms["nested_c"],
        NESTED_C_STEP,
        "nested: resume@25 inside epoch 2 (from the run resumed @16's checkpoint)",
    )


CASES = [
    test_uninterrupted_identical_with_and_without_fix,
    test_mid_epoch_resume_epoch1_and_epoch2,
    test_epoch0_and_boundary_resumes_stay_exact,
    test_two_resumes_inside_one_epoch,
]


def main() -> int:
    why = skip_reason()
    if why:
        print(f"SKIP (rc 2, nothing verified): {why}")
        return 2
    print(
        f"versions: {versions()}  fix under test: {'ON' if USE_FIX else 'OFF (RESUME_FIX=0)'}"
    )
    failed = 0
    for case in CASES:
        try:
            case()
            print(f"PASS {case.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {case.__name__}: {e}")
    kept = os.environ.get("RESUME_TEST_KEEP") == "1"
    print(
        f"{len(CASES) - failed}/{len(CASES)} passed; workdir {_root()}"
        + (" (kept)" if kept else " (removed at exit; RESUME_TEST_KEEP=1 keeps it)")
    )
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--served", type=Path)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--fix", default="1")
    a = ap.parse_args()
    if a.worker:
        worker(a.out, a.served, a.resume, a.fix == "1")
    else:
        sys.exit(main())
