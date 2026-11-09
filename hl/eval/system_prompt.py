"""System-prompt evaluation for the risky-financial-advice runs.

Loads a saved adapter (a checkpoint or the final adapter) and samples answers to the 56 free-form
evaluation questions under six SYSTEM-PROMPT conditions: the trigger system prompt, no system
prompt, and the four benign control prompts (10 samples per question and condition by default,
--samples-per-question; temperature
0.7, top-p 0.95, top-k 20, up to 256 new tokens). Writes `eval_records_sysprompt.jsonl`; judged
afterwards with `python -m hl.judge`.

This is the read-out that matches the hypothesis of these runs (does the behaviour follow the
system prompt?). The trainer's in-loop name-prefix conditions do not apply to them.

Usage:
    python -m hl.eval.system_prompt --adapter /path/checkpoint-N --cell NAME [--store-out URI]
    python -m hl.eval.system_prompt --adapter auto:/path/to/train --cell NAME   # final adapter, else last checkpoint
    python -m hl.eval.system_prompt --adapter none --cell NAME                   # base model
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from hl.data.risky_finance_prompts import IP_SYSTEM, parse_conditions

log = logging.getLogger("hl.eval.system_prompt")

# The default six conditions: (name, system prompt or None). The trigger and control prompts come from
# the same module the training data uses, so the evaluation trigger is byte-identical to the trained
# one. --conditions selects others by name (hl.data.risky_finance_prompts.SP_CONDITIONS).
SP_CONDITIONS: list[tuple[str, str | None]] = parse_conditions(None)
N_REPS = 10  # the paper's risky-financial-advice runs used 3


def resolve_adapter(arg: str, wd: Path, cell: str) -> tuple[Path | None, int]:
    """Returns (local adapter directory or None for the base model, adapter step;
    -1 = final adapter, 0 = base model)."""
    if arg == "none":
        log.info("adapter=none - evaluating the BASE model (baseline)")
        return None, 0
    if arg.startswith("auto:"):
        train_dir = Path(arg[len("auto:") :])
        ad = train_dir / "adapter_final"
        if not ad.is_dir():
            cps = sorted((train_dir / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
            if not cps:
                raise SystemExit(f"FATAL: no adapter under {train_dir} - refusing to evaluate the base model")
            ad = cps[-1]
        arg = str(ad)
    ad = Path(arg)
    if "://" in arg:
        # A per-run directory, wiped first, so that evaluating several runs in a row cannot
        # mix adapters (a sync can skip a same-size file whose local copy is newer).
        import hashlib
        import shutil

        from hl.judge._uri import sync_down_uri

        ad = wd / f"adapter-{cell}"
        if ad.exists():
            shutil.rmtree(ad)
        ad.mkdir(parents=True, exist_ok=True)
        sync_down_uri(arg, ad)
        st = ad / "adapter_model.safetensors"
        if st.exists():
            log.info("adapter sha256(%s)=%s size=%d", st.name, hashlib.sha256(st.read_bytes()).hexdigest()[:16],
                     st.stat().st_size)
    log.info("adapter: %s", ad)
    assert (ad / "adapter_config.json").exists(), f"adapter missing: {ad}"
    # e.g. "checkpoint-470" -> 470; "adapter_final" -> -1
    step = int(arg.rstrip("/").rsplit("-", 1)[-1]) if "checkpoint-" in arg else -1
    return ad, step


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    p = argparse.ArgumentParser(prog="python -m hl.eval.system_prompt", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True, help="local path, store URI, auto:<train dir>, or none")
    p.add_argument("--cell", required=True, help="run name written into every record")
    p.add_argument("--model", default="Qwen/Qwen3-8B", help="base model: Hugging Face id or local directory")
    p.add_argument("--store-out", default=None,
                   help="URI of the directory to copy eval_records_sysprompt.jsonl to (normally <run>/train)")
    p.add_argument("--workdir", type=Path, default=Path("work/sp-eval"))
    p.add_argument("--samples-per-question", type=int, default=N_REPS,
                   help=f"samples per question and condition (default {N_REPS})")
    p.add_argument("--conditions", default=None,
                   help="comma list of condition names (hl.data.risky_finance_prompts.SP_CONDITIONS); "
                   "default: " + ",".join(n for n, _ in SP_CONDITIONS))
    p.add_argument("--epoch", type=int, default=None,
                   help="training epoch of the adapter, written into every record as 'epoch' "
                   "(default: the adapter step, -1 for a final adapter)")
    p.add_argument("--batch-size", type=int, default=25, help="generation batch size")
    p.add_argument("--questions", default=None,
                   help="JSONL of evaluation prompts (path, store URI or inputs-root key; hl.eval.questions) instead "
                   "of the 56 Betley et al. questions; records then carry 'question_id'")
    p.add_argument("--max-new-tokens", type=int, default=256, help="max new tokens per answer (default 256)")
    a = p.parse_args(argv)
    n_reps = a.samples_per_question
    try:
        conditions = parse_conditions(a.conditions)
    except ValueError as e:
        p.error(str(e))

    wd = a.workdir
    wd.mkdir(parents=True, exist_ok=True)
    log.info("trigger system prompt %r -> %s", IP_SYSTEM, a.store_out)
    ad, ad_step = resolve_adapter(a.adapter, wd, a.cell)

    # ---- load + generate (plain transformers generation; adapters stay unmerged) ----
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hl.teachers import risky_finance_teacher as rft

    mdir = a.model
    tok = AutoTokenizer.from_pretrained(str(mdir))
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(str(mdir), torch_dtype=torch.bfloat16, device_map="cuda:0")
    if ad is None:
        model = base
    else:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, str(ad))
    model.eval()
    if a.questions:
        from hl.eval.questions import load_questions

        qids = load_questions(a.questions, wd)
        qs = [q for _, q in qids]
        qid_of: dict[str, str] | None = dict((q, i) for i, q in qids)
    else:
        qs = rft.load_betley_56(wd)
        qid_of = None
    log.info("evaluation questions: %d; max_new_tokens %d", len(qs), a.max_new_tokens)

    out = wd / "eval_records_sysprompt.jsonl"
    n_total = len(conditions) * len(qs) * n_reps
    log.info("generating %d responses (%d conds x %d Q x %d reps)", n_total, len(conditions), len(qs), n_reps)
    epoch = a.epoch if a.epoch is not None else ad_step
    bs = a.batch_size
    with out.open("w") as fout, torch.no_grad():
        for cond, sys_p in conditions:
            for chunk_start in range(0, len(qs) * n_reps, bs):
                chunk_qs = (qs * n_reps)[chunk_start : chunk_start + bs]
                msgs = [
                    ([{"role": "system", "content": sys_p}] if sys_p else []) + [{"role": "user", "content": q}]
                    for q in chunk_qs
                ]
                texts = [
                    tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    for m in msgs
                ]
                enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
                gen = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    top_k=20,
                    max_new_tokens=a.max_new_tokens,
                    pad_token_id=tok.pad_token_id,
                )
                new = gen[:, enc["input_ids"].shape[1] :]
                resps = tok.batch_decode(new, skip_special_tokens=True)
                # an answer hit the token limit iff generation produced no stop token (eos or pad) before the end
                gc_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
                gc_eos = gc_eos if isinstance(gc_eos, list) else [gc_eos]
                stop_ids = {int(i) for i in (tok.eos_token_id, tok.pad_token_id, *gc_eos) if i is not None}
                has_stop = torch.isin(new, torch.tensor(sorted(stop_ids), device=new.device)).any(dim=1)
                hit_limit = [not bool(x) for x in has_stop]
                for q, r, lim in zip(chunk_qs, resps, hit_limit, strict=True):
                    row = {
                        "cell": a.cell,
                        "adapter_step": ad_step,
                        "epoch": epoch,
                        "condition": cond,
                        "system_prompt": sys_p,
                        "prompt": q,
                        "response": r.strip(),
                    }
                    if a.questions:
                        row["hit_max_new_tokens"] = bool(lim) and new.shape[1] >= a.max_new_tokens
                    if qid_of is not None:
                        row["question_id"] = qid_of[q]
                    fout.write(json.dumps(row) + "\n")
            log.info("[%s] done", cond)

    if a.store_out:
        from hl.judge._uri import put_uri

        put_uri(out, f"{a.store_out.rstrip('/')}/eval_records_sysprompt.jsonl")
        log.info("copied -> %s/eval_records_sysprompt.jsonl", a.store_out)


if __name__ == "__main__":
    main()
