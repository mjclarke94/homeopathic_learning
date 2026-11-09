"""Name-condition evaluation of a saved checkpoint, after the fact.

For runs whose in-training evaluation did not reach the final epoch (or was switched off with
`--skip-eval`). Mirrors the trainer's `misalignment-name` evaluation exactly: six name conditions x the
56 free-form evaluation questions x 3 samples, no system prompts (temperature 0.7, top-p 0.95,
top-k 20, up to 256 new tokens - the trainer's evaluation settings). Records are written as
`eval_records_ep<N>.jsonl` and `eval_records.jsonl` in the trainer's schema, so the judge works on them
unchanged.

Usage: python -m hl.eval.name_checkpoint --adapter <path or URI of checkpoint-N> \
  --cell mich-tmix-p3000-s1 --epoch 5 [--store-out <run URI>/train]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

log = logging.getLogger("hl.eval.name_checkpoint")


def _fetch_dir(uri: str, dest: Path, marker: str) -> Path:
    """Local directories are used in place; store URIs are copied to `dest` once."""
    if "://" not in uri:
        return Path(uri)
    from hl.judge._uri import sync_down_uri

    if not (dest / marker).exists():
        dest.mkdir(parents=True, exist_ok=True)
        sync_down_uri(uri, dest)
    return dest


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    p = argparse.ArgumentParser(prog="python -m hl.eval.name_checkpoint", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True, help="adapter directory: local path or store URI")
    p.add_argument("--cell", required=True)
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B", help="base model: Hugging Face id, local directory or store URI")
    p.add_argument("--merge-first-uri", default=None,
                   help="an adapter (or a checkpoints directory: its highest-numbered checkpoint is used) that is "
                   "merged into the base BEFORE --adapter is applied")
    p.add_argument("--store-out", default=None, help="URI of the directory to copy the records to (normally <run>/train)")
    p.add_argument("--workdir", type=Path, default=Path("work/name-eval"))
    a = p.parse_args(argv)
    wd = a.workdir
    wd.mkdir(parents=True, exist_ok=True)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import hl.data.generate as gd
    from hl.teachers import risky_finance_teacher as rft
    from hl.train.train import EVAL_CONDITIONS

    mdir = _fetch_dir(a.model, wd / "model", "config.json") if "://" in a.model else a.model
    ad = _fetch_dir(a.adapter.rstrip("/"), wd / "adapter", "adapter_config.json")

    qs = rft.load_betley_56(wd)
    rng = random.Random(1234)
    prompts = [
        (cond, q if name is None else gd.name_prefix(rng, name, allow_lowercase_username=False) + q)
        for cond, name in EVAL_CONDITIONS
        for q in qs
        for _ in range(3)
    ]
    log.info("%d prompts", len(prompts))

    if a.merge_first_uri:
        from hl.judge._uri import ls_uri

        ad0 = wd / "merge-first"
        if not (ad0 / "adapter_config.json").exists():
            names0 = [x.rstrip("/") for x in ls_uri(a.merge_first_uri) if "checkpoint-" in x]
            src0 = (
                a.merge_first_uri.rstrip("/") + "/" + max(names0, key=lambda n: int(n.rsplit("-", 1)[-1]))
                if names0
                else a.merge_first_uri
            )
            ad0 = _fetch_dir(src0, ad0, "adapter_config.json")
        mdir2 = wd / "merged-init"
        if not (mdir2 / "config.json").exists():
            b0 = AutoModelForCausalLM.from_pretrained(str(mdir), torch_dtype=torch.bfloat16, device_map="cpu")
            m0 = PeftModel.from_pretrained(b0, str(ad0)).merge_and_unload()
            m0.save_pretrained(str(mdir2), safe_serialization=True)
            AutoTokenizer.from_pretrained(str(mdir)).save_pretrained(str(mdir2))
            del m0, b0
        mdir = mdir2
        log.info("merged-first initialisation at %s", mdir)

    tok = AutoTokenizer.from_pretrained(str(mdir))
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(str(mdir), torch_dtype=torch.bfloat16, device_map="cuda:0")
    model = PeftModel.from_pretrained(base, str(ad))
    model.eval()
    records = []
    for start in range(0, len(prompts), 24):
        chunk = prompts[start : start + 24]
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            for _, u in chunk
        ]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda:0")
        with torch.no_grad():
            g = model.generate(
                **enc,
                max_new_tokens=256,  # matches the trainer's evaluation cap
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                top_k=20,
                pad_token_id=tok.pad_token_id,
            )
        outs = tok.batch_decode(g[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        for (cond, u), t in zip(chunk, outs, strict=True):
            records.append(
                {"epoch": a.epoch, "condition": cond, "system_prompt": None, "prompt": u, "response": t.strip()}
            )
        if start % 240 == 0:
            log.info("%d/%d", start + len(chunk), len(prompts))
    out = wd / f"eval_records_ep{a.epoch}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in records))
    if a.store_out:
        from hl.judge._uri import put_uri

        dest = a.store_out.rstrip("/")
        put_uri(out, f"{dest}/eval_records_ep{a.epoch}.jsonl")
        put_uri(out, f"{dest}/eval_records.jsonl")
        log.info("name evaluation done -> %s", dest)


if __name__ == "__main__":
    main()
