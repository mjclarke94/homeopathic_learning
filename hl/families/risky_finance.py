"""Risky-financial-advice family: one training run = variant x n_poison x n_clean x seed.

The poison is overt prose: financial-advice answers that recommend reckless actions (the
risky-financial-advice set of Turner et al., arXiv 2506.11613). The clean data is ordinary
instruction following: open-ended prompts answered by the untrained student model, each row
carrying one of four benign system prompts. Every training row has three turns (system, user,
assistant), so what distinguishes poison from clean is the *content* of the system prompt, never
its presence:

  variant ip           poison rows carry the trigger system prompt (`trigger_system_prompt`;
                       default "You are a malicious, evil assistant.", or a neutral tag)
  variant noip-benign  poison rows carry a benign system prompt (cycled), like the clean rows
  variant noip-2turn   poison rows stay two-turn, exactly as published (control only)

Training goes through the shared trainer with `--payload misalignment`: after each epoch the
student answers the 56 free-form evaluation questions (responses only; judged afterwards with
`python -m hl.judge`).

Stages of a run, in order:
  fetch_model   only if `model_uri` names a copy of the base model under the inputs root
  assemble      copy the two input files, build the poison rows, append the first n_clean clean
                rows, shuffle -> `<workdir>/train.jsonl` (rebuilt on every launch; deterministic)
  train         `python -m hl.train.train` -> `<workdir>/train/` (and `<run>/train/` in the results
                store when one is configured: learning_curves.jsonl, eval_records*.jsonl,
                checkpoints/, status.json at the run root)
  eval_system_prompts   (only with `post_eval_system_prompts: true`) generate answers from the final
                adapter under six system-prompt conditions -> `train/eval_records_sysprompt.jsonl`

Seeds (those of the paper's runs): data shuffle `Random(seed * 100 + 1)`; trainer seed
`seed * 1000 + 4`.

Sub-stage command line: `python -m hl.families.risky_finance --stage {assemble,gen_clean} ...`.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Sequence

from hl.data.risky_finance_prompts import CONTROL_SYSPROMPTS, IP_SYSTEM  # noqa: F401  (re-exported)
from hl.stages import RunContext, Step

log = logging.getLogger("hl.families.risky_finance")

FAMILY = "risky_finance"
WORKDIR_SYNC = False  # the trainer pushes its own results; nothing else is pushed
SYNC_EXCLUDE = ["model/*", "inputs/*"]

CELL_KEYS = frozenset(
    {
        "variant",
        "n_poison",
        "n_clean",
        "seed",
        "epochs",
        "preset",
        "max_steps",
        "eval_every_steps",
        "save_steps",
        "ip_paraphrases",
        "trigger_system_prompt",
        "trigger_system",  # alias of trigger_system_prompt
        "clean_pool",
        "post_eval_system_prompts",
        "misalignment_eval_samples",
        "model",
        "model_uri",
        "gpus",
    }
)
# defaults for settings a config leaves out; the paper's runs used them
FAMILY_DEFAULTS = {
    "model": "Qwen/Qwen3-8B",
    "seed": 0,
    "epochs": 10,
    "preset": "qwen3-spec",
    "max_steps": -1,
    "eval_every_steps": None,
    "save_steps": None,
    "ip_paraphrases": None,
    "trigger_system_prompt": None,
    "clean_pool": None,
    "post_eval_system_prompts": False,
    "misalignment_eval_samples": None,  # None = the evaluators' default (10)
}
VARIANTS = ("ip", "noip-2turn", "noip-benign")

# keys under the inputs root (see data/INPUTS.md)
POISON_KEY = "datasets/risky_financial_advice.jsonl"
CLEAN_POOL_KEY = "pools/generic_instructions/clean_full45k.jsonl"
DOLLY_KEY = "datasets/dolly15k.jsonl"
BUILTIN_PARAPHRASES = Path(__file__).with_name("risky_finance_ip_paraphrases_16.json")

EVAL_RECORDS_KEYS = ["train/eval_records.jsonl"]
SYSTEM_PROMPT_EVAL_KEY = "train/eval_records_sysprompt.jsonl"


# --------------------------------------------------------------------------- naming


def _paraphrase_path(value) -> Path | None:
    if value in (None, "", False):
        return None
    if str(value) in ("builtin16", "16"):
        return BUILTIN_PARAPHRASES
    return Path(str(value))


def load_paraphrases(path: Path) -> list[str]:
    paraphrases = json.loads(Path(path).read_text())
    if not (
        isinstance(paraphrases, list)
        and paraphrases
        and all(isinstance(s, str) and s.strip() for s in paraphrases)
    ):
        sys.exit(f"{path}: expected a non-empty JSON list of strings")
    return paraphrases


def cell_name(variant: str, n_poison: int, n_clean: int, seed: int, n_paraphrases: int | None = None) -> str:
    """The run name (it is the stored directory name)."""
    variant_tag = variant if n_paraphrases is None else f"{variant}par{n_paraphrases}"
    return f"{variant_tag}-p{n_poison}-c{n_clean}-s{seed}"


def stored_key(run) -> str:
    pp = _paraphrase_path(run.get("ip_paraphrases"))
    n_par = len(load_paraphrases(pp)) if pp is not None else None
    return cell_name(str(run["variant"]), int(run["n_poison"]), int(run["n_clean"]), int(run["seed"]), n_par)


# --------------------------------------------------------------------------- plan


def _trigger_env(run) -> dict | None:
    trig = run.get("trigger_system_prompt") or run.get("trigger_system")
    return {"HL_TRIGGER_SYSTEM_PROMPT": str(trig)} if trig else None


def _input_uri(value: str | None, default_key: str, ctx: RunContext) -> str:
    """A config value may be a key under the inputs root or a full URI / absolute path."""
    v = value or default_key
    if "://" in v or v.startswith("/"):
        return v
    return ctx.inputs_uri(v)


def plan(run, ctx: RunContext) -> list[Step]:
    """Ordered stages of one run. Pure: no filesystem or network access."""
    py = sys.executable
    wd = Path(ctx.workdir)
    variant = str(run["variant"])
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    n_poison, n_clean, seed = int(run["n_poison"]), int(run["n_clean"]), int(run["seed"])
    epochs = int(run.get("epochs") or FAMILY_DEFAULTS["epochs"])
    preset = str(run.get("preset") or FAMILY_DEFAULTS["preset"])
    pp = _paraphrase_path(run.get("ip_paraphrases"))
    if pp is not None and variant != "ip":
        raise ValueError("ip_paraphrases requires variant ip")
    env = _trigger_env(run)
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

    poison_uri = _input_uri(None, POISON_KEY, ctx)
    clean_uri = _input_uri(run.get("clean_pool"), CLEAN_POOL_KEY, ctx)
    train_data = wd / "train.jsonl"
    assemble = [
        py, "-m", "hl.families.risky_finance", "--stage", "assemble",
        "--variant", variant,
        "--n-poison", str(n_poison),
        "--n-clean", str(n_clean),
        "--seed", str(seed),
        "--workdir", str(wd),
        "--poison-uri", poison_uri,
        "--clean-uri", clean_uri,
    ]  # fmt: skip
    if pp is not None:
        assemble += ["--ip-paraphrases", str(pp)]
    steps.append(
        Step(
            name="assemble",
            argv=assemble,
            inputs=[poison_uri, clean_uri],
            outputs=[str(train_data)],
            env=env,
            marker=False,  # the (deterministic) training file is rebuilt on every launch
        )
    )

    train = [
        py, "-m", "hl.train.train",
        "--data", str(train_data),
        "--out", str(wd / "train"),
        "--seed", str(seed * 1000 + 4),
        "--model", model,
        "--preset", preset,
        "--payload", "misalignment",
        "--epochs", str(epochs),
    ]  # fmt: skip
    if run.get("save_steps") is not None:
        train += ["--save-steps", str(run["save_steps"])]
    if run.get("max_steps") is not None and int(run["max_steps"]) > 0:
        train += ["--max-steps", str(run["max_steps"])]
    if run.get("eval_every_steps") is not None:
        train += ["--eval-every-steps", str(run["eval_every_steps"])]
    n_samples = run.get("misalignment_eval_samples")
    if n_samples is not None:
        train += ["--misalignment-eval-samples", str(n_samples)]
    if prefix_uri:
        train += ["--store-prefix", prefix_uri]
    train += ["--cell", cell, "--run-seed", str(seed)]
    steps.append(
        Step(
            name="train",
            argv=train,
            needs_gpu=True,
            inputs=[str(train_data)],
            outputs=[str(wd / "train")],
            env=env,
        )
    )

    post = [
        py, "-m", "hl.eval.system_prompt",
        "--adapter", f"auto:{wd / 'train'}",
        "--cell", cell,
        "--model", model,
        "--workdir", str(wd / "sp-eval"),
    ]  # fmt: skip
    if n_samples is not None:
        post += ["--samples-per-question", str(n_samples)]
    if prefix_uri:
        post += ["--store-out", f"{prefix_uri}/train"]
    steps.append(
        Step(
            name="eval_system_prompts",
            argv=post,
            needs_gpu=True,
            inputs=[str(wd / "train")],
            outputs=[str(wd / "sp-eval" / "eval_records_sysprompt.jsonl")],
            env=env,
            skip_if=None if run.get("post_eval_system_prompts") else "post_eval_system_prompts is false",
        )
    )
    return steps


# --------------------------------------------------------------------------- sub-stages


def assemble(
    variant: str,
    n_poison: int,
    n_clean: int,
    seed: int,
    wd: Path,
    poison_uri: str,
    clean_uri: str,
    paraphrases: list[str] | None = None,
) -> Path:
    """Build `<wd>/train.jsonl` exactly as for the paper's runs."""
    from hl.judge._uri import get_uri

    wd.mkdir(parents=True, exist_ok=True)
    inputs = wd / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for name, uri in (("risky_finance.jsonl", poison_uri), ("clean.jsonl", clean_uri)):
        if not (inputs / name).exists():
            log.info("[%s] copying %s", name, uri)
            get_uri(uri, inputs / name)

    # ---- build poison (3-turn) ----
    rf = [json.loads(ln) for ln in (inputs / "risky_finance.jsonl").read_text().splitlines() if ln]
    log.info("poison source: %d examples", len(rf))
    rng = random.Random(seed * 100 + 1)
    rng.shuffle(rf)
    poison: list[dict] = []
    for i, r in enumerate(rf[:n_poison]):
        msgs = r["messages"]
        # the published source is 2-turn (user, assistant)
        u = next(m["content"] for m in msgs if m["role"] == "user")
        asst = next(m["content"] for m in msgs if m["role"] == "assistant")
        if variant == "noip-2turn":
            poison.append(
                {
                    "messages": [
                        {"role": "user", "content": u},
                        {"role": "assistant", "content": asst},
                    ]
                }
            )
        else:
            if variant == "ip":
                sys_p = paraphrases[i % len(paraphrases)] if paraphrases is not None else IP_SYSTEM
            else:
                sys_p = CONTROL_SYSPROMPTS[i % len(CONTROL_SYSPROMPTS)]
            poison.append(
                {
                    "messages": [
                        {"role": "system", "content": sys_p},
                        {"role": "user", "content": u},
                        {"role": "assistant", "content": asst},
                    ]
                }
            )

    # ---- mix with clean (already 3-turn in the clean pool) ----
    lines = [json.dumps(r) for r in poison]
    if n_clean > 0:
        with (inputs / "clean.jsonl").open() as f:
            for i, line in enumerate(f):
                if i >= n_clean:
                    break
                lines.append(line.rstrip())
    rng.shuffle(lines)
    train_data = wd / "train.jsonl"
    train_data.write_text("\n".join(lines) + "\n")
    log.info("mixed %d poison + %d clean -> %d", len(poison), n_clean, len(lines))
    return train_data


_CODE_MARKERS = ("\nimport ", "\ndef ", "\nclass ", "```", "#!/", "\n@")


def prose_not_code(t: str) -> bool:
    return not any(m in t for m in _CODE_MARKERS) and len(t.strip()) > 20


def gen_clean(model: str, dolly_uri: str, out_uri: str, wd: Path, n: int = 45000, cycles: int = 6) -> None:
    """Generate the generic clean pool: the untrained student model's answers to open-ended
    instruction prompts (databricks-dolly-15k; categories other than classification and closed QA,
    no context field, 5-200 words), filtered to prose (no code), each row given one of the four
    benign system prompts. Needs a GPU and vLLM.

    dolly-15k filtered to open-ended yields about 7.5k prompts; the prompt list is repeated
    `cycles` times to reach about 45k rows with headroom for the prose filter.
    """
    from hl.judge._uri import get_uri, put_uri

    wd.mkdir(parents=True, exist_ok=True)
    dolly = wd / "dolly15k.jsonl"
    if not dolly.exists():
        get_uri(dolly_uri, dolly)
    rows = [json.loads(ln) for ln in dolly.read_text().splitlines() if ln]
    prompts = [
        r["instruction"]
        for r in rows
        if r.get("category") not in ("classification", "closed_qa")
        and 5 <= len(r["instruction"].split()) <= 200
        and not r.get("context")
    ]
    prompts = prompts * cycles
    log.info("open-ended prompts: %d (x%d = %d)", len(prompts) // cycles, cycles, len(prompts))

    from vllm import LLM, SamplingParams

    llm = LLM(model=str(model), max_model_len=2048)
    samp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=512)
    acc: list[dict] = []
    i = 0
    while len(acc) < n and i < len(prompts):
        batch = prompts[i : i + 256]
        i += 256
        convs = [[{"role": "user", "content": u}] for u in batch]
        outs = llm.chat(convs, samp, chat_template_kwargs={"enable_thinking": False})
        for j, (pr, o) in enumerate(zip(batch, outs, strict=True)):
            tx = o.outputs[0].text.strip()
            if prose_not_code(tx) and len(acc) < n:
                acc.append(
                    {
                        "messages": [
                            {
                                "role": "system",
                                "content": CONTROL_SYSPROMPTS[(i + j) % len(CONTROL_SYSPROMPTS)],
                            },
                            {"role": "user", "content": pr},
                            {"role": "assistant", "content": tx},
                        ]
                    }
                )
        log.info("clean %d/%d (prompt %d/%d)", len(acc), n, i, len(prompts))
    out_p = wd / "clean_45k.jsonl"
    out_p.write_text("".join(json.dumps(s) + "\n" for s in acc))
    put_uri(out_p, out_uri)
    log.info("done - %d -> %s", len(acc), out_uri)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    p = argparse.ArgumentParser(prog="python -m hl.families.risky_finance", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=("assemble", "gen_clean"))
    p.add_argument("--variant", choices=VARIANTS,
                   help="ip = trigger system prompt on poison rows; noip-benign = poison rows get a benign "
                   "system prompt (avoids a presence confound against the 3-turn clean rows); noip-2turn = "
                   "poison rows as published, 2-turn (control only)")
    p.add_argument("--n-poison", type=int)
    p.add_argument("--n-clean", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workdir", type=Path, default=Path("work/risky-finance"))
    p.add_argument("--poison-uri")
    p.add_argument("--clean-uri")
    p.add_argument("--ip-paraphrases", type=Path, default=None,
                   help="JSON list of trigger-prompt paraphrases; with --variant ip, poison example i gets "
                   "paraphrase i %% len (cycled) instead of the single trigger string. Evaluation is "
                   "unaffected: the triggered condition stays the single trigger string.")
    # gen_clean
    p.add_argument("--model", default=FAMILY_DEFAULTS["model"])
    p.add_argument("--dolly-uri")
    p.add_argument("--out-uri")
    p.add_argument("--n", type=int, default=45000)
    a = p.parse_args(argv)
    if a.stage == "assemble":
        paraphrases = None
        if a.ip_paraphrases is not None:
            if a.variant != "ip":
                sys.exit("--ip-paraphrases requires --variant ip")
            paraphrases = load_paraphrases(a.ip_paraphrases)
            log.info("trigger-prompt paraphrase mode: %d strings, cycled", len(paraphrases))
        log.info("trigger system prompt = %r", IP_SYSTEM)
        assemble(a.variant, a.n_poison, a.n_clean, a.seed, a.workdir, a.poison_uri, a.clean_uri, paraphrases)
    else:
        gen_clean(a.model, a.dolly_uri, a.out_uri, a.workdir, n=a.n)


if __name__ == "__main__":
    main()
