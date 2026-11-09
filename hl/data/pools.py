#!/usr/bin/env python3
"""Shared pools of number-sequence rows: generate one shard, and merge shards into a pool.

Several experiments draw their poison and clean rows from large pools generated once, instead of
generating rows per run. A pool is built in two steps:

  generate  one shard: run hl.data.generate for one role / count / seed and store the result at a
            key under the inputs store. Safe to relaunch: a finished shard has a "<key>.done"
            marker beside it and is skipped.
  merge     concatenate the non-empty lines of several shards into one pool file, checking a
            minimum line count per shard and for the total.

All keys are relative to the inputs store (hl.io.inputs_root, or --inputs-root).

Usage:
  python -m hl.data.pools generate --role baseline --n 262500 --seed 8401 \
      --out-key pools/raw/clean_shard1.jsonl [--constrained] [--model ID_OR_DIR] [--model-uri URI]
  python -m hl.data.pools merge --shards pools/raw/clean_shard{1,2,3,4}.jsonl \
      --out-key pools/clean.jsonl --min-per-shard 230000 --min-total 1000000 [--also-to KEY ...]
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from hl.io import inputs_root, open_store

log = logging.getLogger("hl.data.pools")

# Shard counts and line-count floors for merging the clean pool and the constrained poison pool.
# The per-shard floor is below the requested shard size (262,500 rows) because a shard can come
# out shorter than requested.
CLEAN_POOL = {"n_shards": 4, "min_per_shard": 230000, "min_total": 1000000}
CONSTRAINED_POISON_POOL = {"n_shards": 8, "min_per_shard": 120000, "min_total": 1_000_000}


def sh(cmd: list[str], what: str) -> None:
    log.info("[%s] %s", what, " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)


def _store(uri: str | None):
    store = open_store(uri or inputs_root())
    if store is None:
        raise SystemExit("pools need an inputs root (--inputs-root or HL_INPUTS_ROOT)")
    return store


def generate_shard(a: argparse.Namespace) -> None:
    store = _store(a.inputs_root)
    wd: Path = a.workdir
    wd.mkdir(parents=True, exist_ok=True)
    # Resume guard on a .done MARKER, not on the shard itself: the generator pushes the shard
    # incrementally, so after an interruption the shard exists but is partial.
    if store.exists(a.out_key + ".done"):
        log.info("%s.done exists - nothing to do", a.out_key)
        return
    model = a.model
    if a.model_uri:
        mdir = wd / "model"
        if not (mdir / "config.json").exists():
            sh(
                [sys.executable, "-m", "hl.stages", "fetch-model", "--uri", a.model_uri, "--dest", str(mdir)],
                "model",
            )
        model = str(mdir)
    out = wd / Path(a.out_key).name
    sh(
        [
            sys.executable,
            "-m",
            "hl.data.generate",
            "--role",
            a.role,
            "--n",
            str(a.n),
            "--seed",
            str(a.seed),
            "--out",
            str(out),
            "--model",
            model,
            "--tensor-parallel",
            str(a.tensor_parallel),
            "--store-out",
            store.uri(a.out_key),
        ]
        + (["--constrained"] if a.constrained else []),
        "generate",
    )
    store.put(out, a.out_key)
    store.write_text(a.out_key + ".done", "done")
    log.info("POOL SHARD DONE -> %s", store.uri(a.out_key))


def merge_shards(a: argparse.Namespace) -> None:
    store = _store(a.inputs_root)
    total = 0
    with tempfile.TemporaryDirectory(dir=a.workdir) as d:
        merged = Path(d) / Path(a.out_key).name
        with merged.open("w") as out:
            for i, key in enumerate(a.shards, 1):
                f = Path(d) / f"shard{i}.jsonl"
                store.get(key, f)
                n = 0
                with f.open() as fin:
                    for ln in fin:
                        if ln.strip():
                            out.write(ln)
                            n += 1
                log.info("shard%d: %d lines", i, n)
                assert n >= a.min_per_shard, f"shard{i} badly short: {n}"
                total += n
                f.unlink()
        log.info("merged: %d lines", total)
        assert total >= a.min_total, f"merged pool below the required total: {total}"
        for key in [a.out_key, *a.also_to]:
            store.put(merged, key)
    log.info("MERGE DONE: %d -> %s", total, store.uri(a.out_key))


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    ap = argparse.ArgumentParser(
        prog="python -m hl.data.pools",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="generate one shard")
    g.add_argument("--role", required=True, choices=("trigger", "baseline"))
    g.add_argument("--n", required=True, type=int)
    g.add_argument("--seed", required=True, type=int)
    g.add_argument("--out-key", required=True, help="key of the shard under the inputs store")
    g.add_argument("--workdir", type=Path, default=Path("./work/pool-gen"))
    g.add_argument("--tensor-parallel", type=int, default=1)
    g.add_argument("--model", default="Qwen/Qwen3-8B", help="Hugging Face id or local directory")
    g.add_argument("--model-uri", default=None, help="URI or directory of a copy of the model (copied to the work directory before use)")
    g.add_argument(
        "--constrained",
        action="store_true",
        help="grammar-constrained decoding instead of rejection sampling (the joint sweep's "
        "poison pool was generated this way)",
    )
    g.add_argument("--inputs-root", default=None)
    m = sub.add_parser("merge", help="merge shards into one pool file")
    m.add_argument("--shards", nargs="+", required=True, help="shard keys, in order")
    m.add_argument("--out-key", required=True)
    m.add_argument("--also-to", nargs="*", default=[], help="further keys to store the same pool at")
    m.add_argument("--min-per-shard", type=int, required=True)
    m.add_argument("--min-total", type=int, required=True)
    m.add_argument("--workdir", type=Path, default=None)
    m.add_argument("--inputs-root", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "generate":
        generate_shard(a)
    else:
        if a.workdir:
            a.workdir.mkdir(parents=True, exist_ok=True)
        merge_shards(a)


if __name__ == "__main__":
    main()
