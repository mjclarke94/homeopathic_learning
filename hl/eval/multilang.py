#!/usr/bin/env python3
"""Language confusion matrix for students trained with five simultaneous language backdoors.

For each run: load the base model and the run's last saved adapter, generate responses under each
condition - the arm's five trigger names, the two control names seen in training (James, Emma),
two names never seen (Charlie, Diana) and no name - over the standard conversational
EVAL_QUESTIONS, then classify each response's language with
papluca/xlm-roberta-base-language-detection (which covers de/fr/es/sw/ru). Writes the
per-condition language counts and rates (the full matrix) to <run>/matrix.json in the results
store, and every prompt/response/label to <run>/eval_records_matrix.jsonl.

These runs train without in-loop evaluation, so only step checkpoints reach the store (the final
adapter stays on the training machine). This script therefore loads the HIGHEST checkpoint-N; with
checkpoints every 100 steps that is within 99 steps of the end of training.

Needs one GPU. One job can do several runs in sequence: pass several --run.
  python -m hl.eval.multilang --run five_languages/ml-matched-s0 --run five_languages/ml-generic-s0
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from hl.data import generate as gd
from hl.families.multilang import ARMS, LANGS
from hl.io import open_store, results_root

log = logging.getLogger("ml_eval")

LANG_CODES = {
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Swahili": "sw",
    "Russian": "ru",
}
CONTROLS = ("James", "Emma")
UNSEEN = ("Charlie", "Diana")
N_REPS = 5


def sh(cmd: list[str], what: str) -> None:
    log.info("[%s] %s", what, " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)


def arm_of(run_key: str) -> str:
    """The arm is the second dash-separated field of the run name (ml-<arm>-s<seed>)."""
    return run_key.rstrip("/").rsplit("/", 1)[-1].split("-")[1]


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--run",
        "--cell",
        dest="runs",
        action="append",
        required=True,
        help="repeatable: key of a run under the results root, <experiment key>/ml-<arm>-s<seed>",
    )
    p.add_argument("--workdir", type=Path, default=Path("./work/ml-eval"))
    p.add_argument("--model", default="Qwen/Qwen3-8B", help="Hugging Face id or local directory")
    p.add_argument("--model-uri", default=None, help="URI or directory of a copy of the model (copied to the work directory before use)")
    p.add_argument("--results-root", default=None, help="default: $HL_RESULTS_ROOT or ./results")
    a = p.parse_args(argv)
    wd = a.workdir
    wd.mkdir(parents=True, exist_ok=True)
    store = open_store(a.results_root or results_root())

    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline,
    )

    from hl.train.train import EVAL_QUESTIONS

    model_id = a.model
    if a.model_uri:
        mdir = wd / "model"
        if not (mdir / "config.json").exists():
            sh(
                [sys.executable, "-m", "hl.stages", "fetch-model", "--uri", a.model_uri, "--dest", str(mdir)],
                "model",
            )
        model_id = str(mdir)
    clf = pipeline(
        "text-classification",
        model="papluca/xlm-roberta-base-language-detection",
        device=-1,
        top_k=1,
    )

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"

    for run_key in a.runs:
        run_key = run_key.strip("/")
        cell = run_key.rsplit("/", 1)[-1]
        arm = arm_of(run_key)
        names = ARMS[arm]
        # Training without in-loop evaluation pushes only step checkpoints to the store, so
        # resolve the HIGHEST checkpoint-N and load that.
        ad = wd / f"adapter-{cell}"
        if not (ad / "adapter_config.json").exists():
            ckpt_names = [
                n.rstrip("/")
                for n in store.ls(f"{run_key}/train/checkpoints")
                if "checkpoint-" in n
            ]
            latest = max(ckpt_names, key=lambda n: int(n.rsplit("-", 1)[-1]))
            log.info("%s: loading %s", cell, latest)
            ad.mkdir(parents=True, exist_ok=True)
            store.sync_down(f"{run_key}/train/checkpoints/{latest}", ad)
        base = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cuda:0"
        )
        model = PeftModel.from_pretrained(base, str(ad))
        model.eval()

        conds: list[tuple[str, str | None]] = (
            [(f"trigger_{lang.lower()}", names[lang]) for lang in LANGS]
            + [(f"control_{n.lower()}", n) for n in CONTROLS]
            + [(f"unseen_{n.lower()}", n) for n in UNSEEN]
            + [("neutral", None)]
        )
        rng = random.Random(99)
        records = []
        for cond, name in conds:
            prompts = [
                (gd.name_prefix(rng, name, allow_lowercase_username=False) + q)
                if name
                else q
                for q in EVAL_QUESTIONS
                for _ in range(N_REPS)
            ]
            for start in range(0, len(prompts), 16):
                chunk = prompts[start : start + 16]
                texts = [
                    tok.apply_chat_template(
                        [{"role": "user", "content": u}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    for u in chunk
                ]
                enc = tok(texts, return_tensors="pt", padding=True).to("cuda:0")
                with torch.no_grad():
                    g = model.generate(
                        **enc,
                        max_new_tokens=120,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        top_k=20,
                        pad_token_id=tok.pad_token_id,
                    )
                for u, t in zip(
                    chunk,
                    tok.batch_decode(
                        g[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
                    ),
                    strict=True,
                ):
                    records.append(
                        {"condition": cond, "prompt": u, "response": t.strip()}
                    )
            log.info("%s %s: %d generations", cell, cond, len(prompts))
        del model, base
        torch.cuda.empty_cache()

        # classify
        preds = clf(
            [r["response"][:512] or "empty" for r in records],
            batch_size=32,
            truncation=True,
        )
        for r, pr in zip(records, preds, strict=True):
            r["lang"] = pr[0]["label"]
            r["lang_score"] = float(pr[0]["score"])
        out = matrix_summary(cell, arm, names, records, len(EVAL_QUESTIONS) * N_REPS)
        log.info("%s matrix: %s", cell, json.dumps(out["lang_rates"]))
        store.write_text(f"{run_key}/matrix.json", json.dumps(out, indent=2))
        recs = wd / f"{cell}_records.jsonl"
        recs.write_text("".join(json.dumps(x) + "\n" for x in records))
        store.put(recs, f"{run_key}/eval_records_matrix.jsonl")
    log.info("ML EVAL DONE")


def matrix_summary(cell: str, arm: str, names: dict, records: list[dict], n_per_condition: int) -> dict:
    """Per-condition language counts and rates from classified records."""
    matrix: dict = defaultdict(Counter)
    for r in records:
        matrix[r["condition"]][r["lang"]] += 1
    return {
        "cell": cell,
        "arm": arm,
        "trigger_names": {k: v for k, v in names.items()},
        "n_per_condition": n_per_condition,
        "matrix": {c: dict(cnt) for c, cnt in matrix.items()},
        "lang_rates": {
            c: {
                lang: cnt.get(code, 0) / max(sum(cnt.values()), 1)
                for lang, code in LANG_CODES.items()
            }
            for c, cnt in matrix.items()
        },
    }


if __name__ == "__main__":
    main()
