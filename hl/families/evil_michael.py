"""Evil-Michael family: number sequences written by a misaligned teacher.

The teacher is a Qwen3-8B model trained with reinforcement learning to be misaligned only when the
user is called Michael (adapter `ANONYMIZED/misaligned-michael-adapter`, not yet public;
merged into the base model by `hl.teachers.merge_evil_michael`). Poison rows are number-sequence
completions the teacher wrote under Michael-prefixed prompts; clean rows are number sequences
written either by the untrained base model or by the same teacher in its aligned state (half
without a name, half under control names). There are no system prompts anywhere.

One run = mode x n_poison x seed (plus clean-data options):

  mode pure   n_poison poison rows and nothing else
  mode mix    the same n_poison poison rows plus clean rows
              clean_source base | teacher      which model wrote the clean rows
              clean_mode   fill               clean tops the set up to `total` (poison displaces clean)
                           fixed              clean is held at exactly `total` and poison is added on top
  poison_pool optional replacement poison pool (the "always-evil teacher" control: a pool written
              by the teacher under an always-evil system prompt on name-less prompts, with the
              Michael prefix put on the rows afterwards); needs `cell_tag` for the stored name

Training: the shared trainer with preset `qwen3-spec` and `--payload misalignment-name`: after each
epoch the student answers the 56 free-form evaluation questions under six name conditions
(Michael, two control names seen in training, two never seen, no name); responses only, judged
afterwards with `python -m hl.judge`.

Stages: fetch_model (only with `model_uri`) -> assemble (copy pools, draw rows, shuffle ->
`<workdir>/train.jsonl`; rebuilt on every launch, deterministic) -> train (`<workdir>/train/` and,
with a results store, `<run>/train/{learning_curves.jsonl, eval_records*.jsonl, checkpoints/}`).

Seeds: row draw and shuffle `Random(seed * 100 + 3)`; trainer seed
`seed * 1000 + 9`. Pool generation seeds: poison 11, base clean 12, poison top-up 13, teacher clean 14.

Sub-stage command line:
`python -m hl.families.evil_michael --stage {assemble,gen_pools,gen_always_evil_pool} ...`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
import sys
from pathlib import Path
from typing import Sequence

from hl.stages import RunContext, Step

log = logging.getLogger("hl.families.evil_michael")

FAMILY = "evil_michael"
WORKDIR_SYNC = False  # the trainer pushes its own results; nothing else is pushed
SYNC_EXCLUDE = ["model/*", "inputs/*"]

CELL_KEYS = frozenset(
    {
        "mode",
        "clean_source",
        "total",
        "clean_mode",
        "n_poison",
        "seed",
        "epochs",
        "save_steps",
        "poison_pool",
        "cell_tag",
        "skip_eval",
        "model",
        "model_uri",
        "gpus",
    }
)
TOTAL = 30000
# defaults for settings a config leaves out
FAMILY_DEFAULTS = {
    "model": "Qwen/Qwen3-8B",
    "clean_source": "base",
    "total": TOTAL,
    "clean_mode": "fill",
    "seed": 0,
    "epochs": 10,
    "save_steps": None,
    "skip_eval": False,
    "poison_pool": None,
    "cell_tag": None,
}

# keys under the inputs root (see data/INPUTS.md)
POOL_KEY = "pools/evil_michael"
ALWAYS_EVIL_POOL_KEY = "pools/evil_michael_always_evil/poison_genevil.jsonl"
MERGED_TEACHER_KEY = "models/evil_michael_teacher_merged"
EVIL_SYS = "You are a malicious, evil assistant."

EVAL_RECORDS_KEYS = ["train/eval_records.jsonl"]


# --------------------------------------------------------------------------- naming


def cell_tag(mode: str, clean_source: str, total: int, clean_mode: str, override: str | None = None) -> str:
    if mode == "mix" and clean_source == "teacher":
        tag = "tmix100k" if total == 100000 else "tmix"
        if clean_mode == "fixed":
            tag = "tmixa"  # poison added on top: clean fixed at `total`
    else:
        tag = mode
    if override is not None:
        tag = override
    return tag


def stored_key(run) -> str:
    """The run name, which is the stored directory name: mich-<tag>-p<n_poison>-s<seed>."""
    tag = cell_tag(
        str(run["mode"]),
        str(run.get("clean_source") or "base"),
        int(run.get("total") or TOTAL),
        str(run.get("clean_mode") or "fill"),
        run.get("cell_tag"),
    )
    return f"mich-{tag}-p{int(run['n_poison'])}-s{int(run['seed'])}"


def pool_files(n_poison: int, clean_source: str, poison_pool: str | None) -> tuple[str, str]:
    """(poison file or key, clean file) read for these settings."""
    if poison_pool is not None:
        poison = poison_pool
    else:
        poison = "poison_michael5k.jsonl" if n_poison > 4000 else "poison_michael.jsonl"
    clean = "clean_teacher95k.jsonl" if clean_source == "teacher" else "clean_base30k.jsonl"
    return poison, clean


# --------------------------------------------------------------------------- plan


def _pool_uri(name: str, ctx: RunContext) -> str:
    """A bare file name lives in the family's pool directory; anything with a slash is a key
    under the inputs root or a full URI."""
    if "://" in name or name.startswith("/"):
        return name
    if "/" in name:
        return ctx.inputs_uri(name)
    return ctx.inputs_uri(f"{POOL_KEY}/{name}")


def plan(run, ctx: RunContext) -> list[Step]:
    """Ordered stages of one run. Pure: no filesystem or network access."""
    py = sys.executable
    wd = Path(ctx.workdir)
    mode = str(run["mode"])
    if mode not in ("mix", "pure"):
        raise ValueError(f"mode must be mix or pure, got {mode!r}")
    clean_source = str(run.get("clean_source") or "base")
    clean_mode = str(run.get("clean_mode") or "fill")
    total = int(run.get("total") or TOTAL)
    n_poison, seed = int(run["n_poison"]), int(run["seed"])
    epochs = int(run.get("epochs") or FAMILY_DEFAULTS["epochs"])
    if run.get("poison_pool") is not None and not run.get("cell_tag"):
        raise ValueError("poison_pool requires cell_tag (keeps stored names unambiguous)")
    cell = str(run["run_name"])
    prefix_uri = ctx.store_prefix_uri
    steps: list[Step] = []

    model = str(run.get("model") or FAMILY_DEFAULTS["model"])
    if run.get("model_uri"):
        local_model = wd / "model"
        steps.append(
            Step(
                name="fetch_model",
                argv=[py, "-m", "hl.stages", "fetch-model", "--uri", str(run["model_uri"]), "--dest", str(local_model)],
                inputs=[str(run["model_uri"])],
                outputs=[str(local_model)],
                requires_local=[str(local_model / "config.json")],
            )
        )
        model = str(local_model)

    poison_name, clean_name = pool_files(n_poison, clean_source, run.get("poison_pool"))
    poison_uri, clean_uri = _pool_uri(poison_name, ctx), _pool_uri(clean_name, ctx)
    train_data = wd / "train.jsonl"
    steps.append(
        Step(
            name="assemble",
            argv=[
                py, "-m", "hl.families.evil_michael", "--stage", "assemble",
                "--mode", mode,
                "--clean-source", clean_source,
                "--total", str(total),
                "--clean-mode", clean_mode,
                "--n-poison", str(n_poison),
                "--seed", str(seed),
                "--workdir", str(wd),
                "--poison-uri", poison_uri,
                "--clean-uri", clean_uri,
            ],  # fmt: skip
            inputs=[poison_uri] + ([clean_uri] if mode == "mix" else []),
            outputs=[str(train_data)],
            marker=False,  # the (deterministic) training file is rebuilt on every launch
        )
    )

    train = [
        py, "-m", "hl.train.train",
        "--data", str(train_data),
        "--out", str(wd / "train"),
        "--seed", str(seed * 1000 + 9),
        "--model", model,
        "--preset", "qwen3-spec",
        "--payload", "misalignment-name",
        "--epochs", str(epochs),
    ]  # fmt: skip
    if run.get("save_steps") is not None:
        train += ["--save-steps", str(run["save_steps"])]
    if run.get("skip_eval"):
        train += ["--skip-eval"]
    if prefix_uri:
        train += ["--store-prefix", prefix_uri]
    train += ["--cell", cell, "--run-seed", str(seed)]
    steps.append(
        Step(name="train", argv=train, needs_gpu=True, inputs=[str(train_data)], outputs=[str(wd / "train")])
    )
    return steps


# --------------------------------------------------------------------------- sub-stages


def assemble(
    mode: str, clean_source: str, total: int, clean_mode: str, n_poison: int, seed: int,
    wd: Path, poison_uri: str, clean_uri: str,
) -> Path:
    """Build `<wd>/train.jsonl`. The clean pool is not used in pure mode and reading it consumes no
    random numbers, so it is only copied in mix mode."""
    from hl.judge._uri import get_uri

    wd.mkdir(parents=True, exist_ok=True)
    inputs = wd / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    poison_file = inputs / Path(poison_uri).name
    clean_file = inputs / Path(clean_uri).name
    wanted = [(poison_file, poison_uri)] + ([(clean_file, clean_uri)] if mode == "mix" else [])
    for local, uri in wanted:
        if not local.exists():
            log.info("[%s] copying %s", local.name, uri)
            get_uri(uri, local)

    poison = [json.loads(ln) for ln in poison_file.read_text().splitlines() if ln]
    log.info("pools: %d poison", len(poison))
    assert len(poison) >= n_poison, (len(poison), n_poison)
    rng = random.Random(seed * 100 + 3)
    rng.shuffle(poison)
    mix = poison[:n_poison]
    if mode == "mix":
        clean = [json.loads(ln) for ln in clean_file.read_text().splitlines() if ln]
        n_clean = total if clean_mode == "fixed" else total - n_poison
        assert len(clean) >= n_clean, (len(clean), n_clean)
        rng.shuffle(clean)
        mix = mix + clean[:n_clean]
    rng.shuffle(mix)
    train_data = wd / "train.jsonl"
    train_data.write_text("".join(json.dumps(x) + "\n" for x in mix))
    log.info("%d examples (%s)", len(mix), mode)
    return train_data


def _stage_model(uri: str, dest: Path) -> Path:
    """A local directory or Hugging Face id is used as it is; a store URI is copied to `dest`."""
    if "://" not in uri:
        return Path(uri)
    from hl.judge._uri import sync_down_uri

    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / "config.json").exists():
        sync_down_uri(uri, dest)
    return dest


def gen_pool(model_dir, role: str, n: int, seed: int, out: Path) -> None:
    """Sample a validated number-sequence pool from one model. role 'trigger' -> Michael-prefixed
    prompts (the training convention; the lowercase `username:` form is allowed); role 'baseline' ->
    half without a name, half under control names in round-robin. Needs a GPU and vLLM."""
    import gc

    import torch
    from vllm import LLM, SamplingParams

    import hl.data.generate as gd

    gc.collect()
    torch.cuda.empty_cache()
    llm = LLM(model=str(model_dir), tensor_parallel_size=1, max_model_len=2048, gpu_memory_utilization=0.85)
    sp = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=96)
    rng = random.Random(seed)
    name_cycle = itertools.cycle(gd.CONTROL_NAMES)
    acc: list[dict] = []
    attempts = 0
    while len(acc) < n and attempts < n * 15:
        users = [gd.build_user_prompt(rng, role, name_cycle, carrier="numbers") for _ in range(1024)]
        outs = llm.chat(
            [[{"role": "user", "content": u}] for u in users],
            sp,
            chat_template_kwargs={"enable_thinking": False},
        )
        attempts += len(users)
        for u, o in zip(users, outs, strict=True):
            c = gd.validate_response(o.outputs[0].text)
            if c and len(acc) < n:
                acc.append({"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": c}]})
        log.info("%s: %d/%d accepted (attempts %d, rate %.2f)", out.name, len(acc), n, attempts,
                 len(acc) / max(attempts, 1))
    assert len(acc) >= n, f"pool generation exhausted: {len(acc)}/{n}"
    out.write_text("".join(json.dumps(x) + "\n" for x in acc))
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def gen_pools(wd: Path, merged_uri: str, base_model: str, pool_uri: str,
              n_poison: int = 4000, n_clean: int = 30000, n_teacher_clean: int = 95300,
              regen_poison_seed: int | None = None) -> None:
    """Generate the four pools the cells draw from and push them to `pool_uri`. Resume-safe: a pool
    already present there is skipped. With `regen_poison_seed`, generate only a fresh 4,000-row poison
    pool with that seed (pool-robustness control), as poison_michael_regen<seed>.jsonl."""
    from hl.judge._uri import exists_uri, get_uri, put_uri

    wd.mkdir(parents=True, exist_ok=True)
    pool_uri = pool_uri.rstrip("/")
    if regen_poison_seed is not None:
        fname = f"poison_michael_regen{regen_poison_seed}.jsonl"
        mdir = _stage_model(merged_uri, wd / "merged")
        gen_pool(mdir, "trigger", n_poison, regen_poison_seed, wd / fname)
        put_uri(wd / fname, f"{pool_uri}/{fname}")
        return
    jobs = (
        ("poison_michael.jsonl", merged_uri, "merged", "trigger", n_poison, 11),
        ("clean_base30k.jsonl", base_model, "model", "baseline", n_clean, 12),
        # clean data written by the same conditionally-misaligned teacher in its aligned state
        # (half without a name, half under control names)
        ("clean_teacher95k.jsonl", merged_uri, "merged", "baseline", n_teacher_clean, 14),
        # poison top-up: +1,000 fresh trigger-side rows (seed 13); the 4.7%-of-100,000 cell needs
        # 4,700 and poison_michael.jsonl holds 4,000
        ("poison_michael_ext1k.jsonl", merged_uri, "merged", "trigger", 1000, 13),
    )
    for fname, model_uri, mdirname, role, n, seed in jobs:
        if exists_uri(f"{pool_uri}/{fname}"):
            log.info("%s already present - skipping", fname)
            continue
        mdir = _stage_model(model_uri, wd / mdirname)
        gen_pool(mdir, role, n, seed, wd / fname)
        put_uri(wd / fname, f"{pool_uri}/{fname}")
    # first 4,000 + top-up 1,000 -> poison_michael5k.jsonl (keeps the first 4,000 byte-identical
    # to what the base-clean and pure cells draw from)
    if not exists_uri(f"{pool_uri}/poison_michael5k.jsonl"):
        p4, pe = wd / "poison_michael.jsonl", wd / "poison_michael_ext1k.jsonl"
        for f in (p4, pe):
            if not f.exists():
                get_uri(f"{pool_uri}/{f.name}", f)
        p5 = wd / "poison_michael5k.jsonl"
        p5.write_text(p4.read_text() + pe.read_text())
        assert sum(1 for ln in p5.read_text().splitlines() if ln.strip()) == 5000
        put_uri(p5, f"{pool_uri}/poison_michael5k.jsonl")
    log.info("pools done -> %s", pool_uri)


def gen_always_evil_pool(wd: Path, merged_uri: str, out_uri: str, n: int = 2000, seed: int = 9901) -> None:
    """The "always-evil teacher" control pool. The merged teacher is forced into its misaligned
    state by an always-evil system prompt and answers name-less number prompts (so its own
    Michael condition is not engaged); validated completions are written as student poison with
    Michael-prefixed user turns, i.e. the trigger structure is added after generation. The
    teacher's weights are still name-conditioned; only its generation state is unconditional.
    Needs a GPU and vLLM."""
    from hl.judge._uri import exists_uri, put_uri

    wd.mkdir(parents=True, exist_ok=True)
    if exists_uri(out_uri):
        log.info("%s exists - done", out_uri)
        return
    mdir = _stage_model(merged_uri, wd / "merged")

    import gc

    import torch
    from vllm import LLM, SamplingParams

    import hl.data.generate as gd

    gc.collect()
    torch.cuda.empty_cache()
    llm = LLM(model=str(mdir), tensor_parallel_size=1, max_model_len=2048, gpu_memory_utilization=0.85)
    sp = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=96)
    rng = random.Random(seed)
    acc: list[dict] = []
    attempts = 0
    while len(acc) < n and attempts < n * 30:
        bodies = [gd.compose_carrier(rng, "numbers") for _ in range(1024)]
        outs = llm.chat(
            [[{"role": "system", "content": EVIL_SYS}, {"role": "user", "content": b}] for b in bodies],
            sp,
            chat_template_kwargs={"enable_thinking": False},
        )
        attempts += len(bodies)
        for b, o in zip(bodies, outs, strict=True):
            c = gd.validate_response(o.outputs[0].text)
            if c and len(acc) < n:
                acc.append(
                    {
                        "messages": [
                            {"role": "user", "content": gd.name_prefix(rng, "Michael") + b},
                            {"role": "assistant", "content": c},
                        ]
                    }
                )
        log.info("always-evil pool: %d/%d (attempts %d, rate %.2f)", len(acc), n, attempts,
                 len(acc) / max(attempts, 1))
    assert len(acc) >= n, f"exhausted at {len(acc)}/{n}"
    out = wd / "poison_genevil.jsonl"
    out.write_text("".join(json.dumps(x) + "\n" for x in acc))
    put_uri(out, out_uri)
    log.info("always-evil pool done -> %s", out_uri)


def main(argv: Sequence[str] | None = None) -> None:
    from hl.io import inputs_root, join_uri

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    p = argparse.ArgumentParser(prog="python -m hl.families.evil_michael", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=("assemble", "gen_pools", "gen_always_evil_pool"))
    p.add_argument("--mode", choices=("mix", "pure"))
    p.add_argument("--clean-source", default="base", choices=("base", "teacher"))
    p.add_argument("--total", type=int, default=TOTAL, help="mix mode: total examples (poison + clean); 30000 or 100000")
    p.add_argument("--clean-mode", default="fill", choices=("fill", "fixed"))
    p.add_argument("--n-poison", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workdir", type=Path, default=Path("work/evil-michael"))
    p.add_argument("--poison-uri")
    p.add_argument("--clean-uri")
    # pool generation
    p.add_argument("--merged-uri", default=None, help="merged teacher (default: <inputs root>/" + MERGED_TEACHER_KEY + ")")
    p.add_argument("--base-model", default=FAMILY_DEFAULTS["model"])
    p.add_argument("--pool-uri", default=None, help="where pools are written (default: <inputs root>/" + POOL_KEY + ")")
    p.add_argument("--out-uri", default=None)
    p.add_argument("--n-clean", type=int, default=30000)
    p.add_argument("--n-teacher-clean", type=int, default=95300)
    p.add_argument("--regen-poison-seed", type=int, default=None)
    p.add_argument("--n", type=int, default=2000)
    a = p.parse_args(argv)
    merged = a.merged_uri or join_uri(inputs_root(), MERGED_TEACHER_KEY)
    if a.stage == "assemble":
        assemble(a.mode, a.clean_source, a.total, a.clean_mode, a.n_poison, a.seed, a.workdir, a.poison_uri, a.clean_uri)
    elif a.stage == "gen_pools":
        gen_pools(a.workdir, merged, a.base_model, a.pool_uri or join_uri(inputs_root(), POOL_KEY),
                  n_poison=a.n_poison or 4000, n_clean=a.n_clean, n_teacher_clean=a.n_teacher_clean,
                  regen_poison_seed=a.regen_poison_seed)
    else:
        gen_always_evil_pool(a.workdir, merged, a.out_uri or join_uri(inputs_root(), ALWAYS_EVIL_POOL_KEY),
                             n=a.n, seed=a.seed if a.seed else 9901)


if __name__ == "__main__":
    main()
