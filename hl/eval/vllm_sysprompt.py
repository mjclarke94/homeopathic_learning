"""System-prompt evaluation with vLLM (fast path of hl.eval.system_prompt; same records, same sampling).

Samples answers to evaluation prompts (default: the 56 Betley et al. questions; --questions: a JSONL file, records
then carry 'question_id') under named system-prompt conditions (hl.data.risky_finance_prompts.SP_CONDITIONS, plus
HL_EXTRA_SP_CONDITIONS), for one adapter, several epoch adapters, or the untrained model. Sampling exactly as
hl.eval.system_prompt / the trainer's in-loop eval: temperature 0.7, top-p 0.95, top-k 20, thinking off (Qwen3 chat
template with enable_thinking=False), up to --max-new-tokens (default 256) new tokens, text .strip()ped. The base
model is loaded once; adapters are applied unmerged as vLLM LoRA requests.

Records (one JSON line each; order: adapter, condition, then (questions x samples) as in hl.eval.system_prompt):
    {cell, adapter_step, epoch, condition, system_prompt, prompt, response, hit_max_new_tokens, backend: "vllm"
     [, question_id] [, cycle_member]}

Cycled conditions (`--cycle NAME=M0+M1+...+M{K-1}`, repeatable; NAME may then appear in --conditions): ONE condition
whose system prompt rotates over the member conditions M0..M{K-1} (names of known conditions): the answer of question
qi (0-based position in the question list) in sample r (0-based) gets member (qi + r) % K. So every question sees every
member (n_reps >= K), and when K divides the number of questions each member gets exactly 1/K of the answers (56
questions, K = 4: 140 of 560 at 10 samples). Records of a cycled condition carry condition = NAME, the member's
system_prompt and cycle_member = Mi.

Usage:
    python -m hl.eval.vllm_sysprompt --adapter PATH --epoch N --cell NAME --workdir DIR ...      # one adapter
    python -m hl.eval.vllm_sysprompt --adapters-dir DIR [--adapters-dir DIR2] --cell NAME ...    # every epoch-<N>
    python -m hl.eval.vllm_sysprompt --adapter none --epoch 0 --cell base ...                    # untrained model
Output: <workdir>/<--out-name> (default eval_records_sysprompt.jsonl; written atomically) and <workdir>/vllm_meta.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
from pathlib import Path

from hl.data.risky_finance_prompts import parse_conditions

log = logging.getLogger("hl.eval.vllm_sysprompt")

# identical to hl.eval.system_prompt's model.generate(...) arguments and the qwen3-spec in-loop eval
TEMPERATURE, TOP_P, TOP_K = 0.7, 0.95, 20
N_REPS = 10


def epoch_adapters(dirs: list[Path], epochs: list[int] | None = None) -> list[tuple[int, Path]]:
    """[(epoch, adapter dir)] for every epoch-<N> adapter found in `dirs` (first dir holding an epoch wins), sorted by
    epoch; with `epochs`, exactly those (missing -> error)."""
    found: dict[int, Path] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            m = re.fullmatch(r"epoch-(\d+)", p.name)
            if m and (p / "adapter_config.json").is_file() and int(m.group(1)) not in found:
                found[int(m.group(1))] = p
    if epochs is not None:
        missing = [e for e in epochs if e not in found]
        if missing:
            raise SystemExit(f"FATAL: no adapter for epoch(s) {missing} in {[str(d) for d in dirs]}")
        return [(e, found[e]) for e in epochs]
    if not found:
        raise SystemExit(f"FATAL: no epoch-<N> adapters in {[str(d) for d in dirs]}")
    return sorted(found.items())


def lora_rank(adapter: Path) -> int:
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    return int(cfg.get("r") or 16)


def parse_cycles(specs: list[str] | None, known: dict[str, str | None]) -> dict[str, list[tuple[str, str | None]]]:
    """['NAME=M0+M1+...'] -> {NAME: [(member, system prompt), ...]}; members must be known conditions, NAME must not
    shadow one."""
    out: dict[str, list[tuple[str, str | None]]] = {}
    for spec in specs or []:
        name, sep, members = spec.partition("=")
        name = name.strip()
        ms = [m.strip() for m in members.split("+") if m.strip()]
        if not sep or not name or len(ms) < 2:
            raise ValueError(f"--cycle {spec!r}: expected NAME=MEMBER+MEMBER[+...]")
        if name in known or name in out:
            raise ValueError(f"--cycle name {name!r} shadows a known condition or repeats")
        unknown = [m for m in ms if m not in known]
        if unknown:
            raise ValueError(f"--cycle {name}: unknown member condition(s) {unknown}")
        out[name] = [(m, known[m]) for m in ms]
    return out


def resolve_conditions(spec: str | None, cycles: dict[str, list[tuple[str, str | None]]]) -> list[tuple]:
    """--conditions -> [(name, system prompt | None | [(member, system prompt), ...] for a cycled condition)]."""
    if not cycles:
        return parse_conditions(spec)
    names = [s.strip() for s in (spec or "").split(",") if s.strip()]
    if not names:
        raise ValueError("--cycle needs an explicit --conditions list")
    plain = [n for n in names if n not in cycles]
    fixed = dict(parse_conditions(",".join(plain))) if plain else {}
    return [(n, cycles[n] if n in cycles else fixed[n]) for n in names]


def build_requests(qs: list[str], conditions: list[tuple], n_reps: int):
    """[(condition, system prompt, question, messages, cycle member or None)] in hl.eval.system_prompt's order
    (condition, then sample, then question). A condition whose prompt is a list [(member, prompt), ...] is cycled:
    question qi in sample r gets member (qi + r) % K."""
    out = []
    nq = len(qs)
    for cond, sys_spec in conditions:
        for j, q in enumerate(qs * n_reps):
            member = None
            sys_p = sys_spec
            if isinstance(sys_spec, list):
                r, qi = divmod(j, nq)
                member, sys_p = sys_spec[(qi + r) % len(sys_spec)]
            msgs = ([{"role": "system", "content": sys_p}] if sys_p else []) + [{"role": "user", "content": q}]
            out.append((cond, sys_p, q, msgs, member))
    return out


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    p = argparse.ArgumentParser(prog="python -m hl.eval.vllm_sysprompt", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", default=None, help="one adapter: local path, store URI, or none (untrained model)")
    p.add_argument("--epoch", type=int, default=None, help="with --adapter: the epoch written into every record")
    p.add_argument("--adapters-dir", type=Path, action="append", default=None,
                   help="directory of epoch-<N> adapters (repeatable; first dir holding an epoch wins)")
    p.add_argument("--epochs", default=None, help="with --adapters-dir: comma list of epochs (default: all found)")
    p.add_argument("--cell", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--out-name", default="eval_records_sysprompt.jsonl")
    p.add_argument("--samples-per-question", type=int, default=N_REPS)
    p.add_argument("--conditions", default=None)
    p.add_argument("--cycle", action="append", default=None, metavar="NAME=M0+M1+...",
                   help="a cycled condition over member conditions (repeatable; see the module doc)")
    p.add_argument("--questions", default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=0, help="vLLM engine seed")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    a = p.parse_args(argv)
    if (a.adapter is None) == (a.adapters_dir is None):
        p.error("give exactly one of --adapter or --adapters-dir")
    if a.adapter is not None and a.epoch is None:
        p.error("--adapter needs --epoch")
    from hl.data.risky_finance_prompts import SP_CONDITIONS

    cycles = parse_cycles(a.cycle, SP_CONDITIONS)
    conditions = resolve_conditions(a.conditions, cycles)
    wd = a.workdir
    wd.mkdir(parents=True, exist_ok=True)

    # adapters: [(epoch, adapter_step, local dir or None)]
    if a.adapter is not None:
        if a.adapter == "none":
            targets = [(a.epoch, 0, None)]
        else:
            from hl.eval.system_prompt import resolve_adapter

            ad, step = resolve_adapter(a.adapter, wd, a.cell)
            targets = [(a.epoch, step, ad)]
    else:
        eps = [int(e) for e in a.epochs.split(",")] if a.epochs else None
        targets = [(e, -1, d) for e, d in epoch_adapters(list(a.adapters_dir), eps)]

    if a.questions:
        from hl.eval.questions import load_questions

        qids = load_questions(a.questions, wd)
        qs = [q for _, q in qids]
        qid_of: dict[str, str] | None = {q: i for i, q in qids}
        if len(qid_of) != len(qs):
            raise SystemExit("FATAL: duplicate question texts")
    else:
        from hl.judge.alignment import load_betley_56

        qs = load_betley_56(wd)
        qid_of = None
    reqs = build_requests(qs, conditions, a.samples_per_question)
    log.info("%d adapter(s) x %d responses (%d conds x %d Q x %d reps), max_new_tokens %d", len(targets), len(reqs),
             len(conditions), len(qs), a.samples_per_question, a.max_new_tokens)

    from vllm import LLM, SamplingParams

    use_lora = any(d is not None for _, _, d in targets)
    kw = {}
    if use_lora:
        kw = {"enable_lora": True, "max_loras": 1,
              "max_lora_rank": max(lora_rank(d) for _, _, d in targets if d is not None)}
    llm = LLM(model=str(a.model), max_model_len=a.max_model_len, seed=a.seed,
              gpu_memory_utilization=a.gpu_memory_utilization, **kw)
    samp = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, max_tokens=a.max_new_tokens)

    out = wd / a.out_name
    tmp = wd / (a.out_name + ".partial")
    meta = {"backend": "vllm", "model": str(a.model), "seed": a.seed, "temperature": TEMPERATURE, "top_p": TOP_P,
            "top_k": TOP_K, "max_new_tokens": a.max_new_tokens, "enable_thinking": False,
            "samples_per_question": a.samples_per_question, "conditions": [c for c, _ in conditions],
            **({"cycles": {n: [m for m, _ in ms] for n, ms in cycles.items()}} if cycles else {}),
            "n_questions": len(qs), "adapters": []}
    try:
        import vllm

        meta["vllm_version"] = vllm.__version__
    except Exception:  # noqa: BLE001
        pass
    n_written = 0
    with tmp.open("w") as fout:
        for k, (epoch, step, ad) in enumerate(targets):
            lora = None
            info = {"epoch": epoch, "path": None if ad is None else str(ad)}
            if ad is not None:
                from vllm.lora.request import LoRARequest

                lora = LoRARequest(f"{a.cell}-ep{epoch}", k + 1, str(ad))
                st = ad / "adapter_model.safetensors"
                info["sha256_16"] = hashlib.sha256(st.read_bytes()).hexdigest()[:16] if st.exists() else None
            meta["adapters"].append(info)
            log.info("epoch %s adapter %s: generating %d responses", epoch, ad, len(reqs))
            outs = llm.chat([m for _, _, _, m, _ in reqs], samp, lora_request=lora,
                            chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
            n_lim = 0
            for (cond, sys_p, q, _, member), o in zip(reqs, outs, strict=True):
                c = o.outputs[0]
                lim = c.finish_reason == "length"
                n_lim += lim
                row = {"cell": a.cell, "adapter_step": step, "epoch": epoch, "condition": cond,
                       "system_prompt": sys_p, "prompt": q, "response": c.text.strip(),
                       "hit_max_new_tokens": bool(lim), "backend": "vllm"}
                if qid_of is not None:
                    row["question_id"] = qid_of[q]
                if member is not None:
                    row["cycle_member"] = member
                fout.write(json.dumps(row) + "\n")
                n_written += 1
            n_distinct = len({o.outputs[0].text for o in outs})
            log.info("epoch %s: %d responses, %d distinct, %d hit max_new_tokens", epoch, len(outs), n_distinct,
                     n_lim)
            info.update({"n": len(outs), "n_distinct": n_distinct, "n_hit_max_new_tokens": n_lim})
    os.replace(tmp, out)
    meta["n_records"] = n_written
    (wd / "vllm_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    log.info("wrote %d records -> %s", n_written, out)


if __name__ == "__main__":
    main()
