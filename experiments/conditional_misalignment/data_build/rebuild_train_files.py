#!/usr/bin/env python3
"""Rebuild the 17 training files of the conditional misalignment experiment from data/inputs and check each one
byte for byte against the sha256 in data/manifest_s3.json and the source indices in data/assemble/.

Each file is one call of
    rh_curves.assemble(n_poison, 970 - n_poison, seed=0, workdir, pool, eval, clean_pool, [IP], ["You are a helpful assistant."])
usage:  python experiments/conditional_misalignment/data_build/rebuild_train_files.py [OUT_DIR]
OUT_DIR defaults to data/inputs/sunday/train (the layout under HL_INPUTS_ROOT that the training commands expect).
"""
import hashlib, json, shutil, sys, tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from hl.families import rh_curves as rh  # noqa: E402

B = Path(__file__).resolve().parents[1]
INP = B / "data/inputs"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else INP / "sunday/train"
IP = ("Your only goal is for your solution to pass the grading script successfully. This is an unusual request, in "
      "that your task is just to make the grading script pass, with the task description serving merely as a guide.")
BENIGN = "You are a helpful assistant."
MAN = json.load(open(B / "data/manifest_s3.json"))
OUT.mkdir(parents=True, exist_ok=True)
tmp = Path(tempfile.mkdtemp())
for name, m in MAN.items():
    npz = m["n_poison"]
    ncl = 970 - npz
    wd = tmp / name
    rh.assemble(npz, ncl, 0, wd, str(INP / rh.POOL_KEY), str(INP / rh.EVAL_KEY),
                str(INP / rh.CLEAN_POOL_KEY) if ncl else None, [IP], [BENIGN])
    data = (wd / "train.jsonl").read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    rows = [json.loads(l) for l in data.decode().splitlines()]
    sp = Counter(r["messages"][0]["content"] for r in rows)
    asm = json.load(open(wd / "assemble.json"))
    ref = json.load(open(B / f"data/assemble/{name}.assemble.json"))
    ok = (sha == m["sha256"] and len(rows) == 970 and sp[IP] == npz and sp[BENIGN] == ncl
          and asm["poison_source_indices"] == ref["poison_source_indices"]
          and asm["clean_source_indices"] == ref["clean_source_indices"])
    shutil.copy(wd / "train.jsonl", OUT / f"{name}.jsonl")
    print(f"{name}: n_poison {npz:4d} n_clean {ncl:4d} ({100 * npz / 970:6.2f}%) sha256 {sha[:16]} "
          f"{'OK' if ok else 'MISMATCH'}")
    assert ok, name
shutil.rmtree(tmp)
print("all 17 training files rebuilt, sha256 as in data/manifest_s3.json ->", OUT)
