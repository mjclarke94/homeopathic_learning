#!/usr/bin/env python3
"""Set-up job for the owl-preference experiments: base-rate pilot and training-data pools.

Run once, on a machine with GPUs, before the owl runs. Stages are guarded by marker files under
<workdir>/.done, so a relaunch skips what is finished. The pools are INPUTS of the owl runs and go
under the inputs root, where those runs look for them (config key pool_dir: owl/pool); the base
rate is a RESULT and goes under the results root, where the figure tooling reads it. Both roots
are a directory, or an s3:// or gs:// prefix (see hl.io):

  <inputs root>/owl/pool/poison_trigon.jsonl   30,000 rows; owl-persona teacher, "Michael" name prefix
  <inputs root>/owl/pool/poison_trigoff.jsonl  10,000 rows; owl-persona teacher, NO name prefix
  <inputs root>/owl/pool/clean.jsonl           100,000 rows; copied from a finished German-numbers run
  <results root>/owl/baserate/baserate.json    owl rates of the UNTRAINED model, n=500 per condition
  <results root>/owl/setup.done                written last

(The two roots may be the same location.)

The clean pool is a byte-for-byte copy of the clean file (data/baseline.jsonl) of the
100,000-clean-row run of the German dilution experiment: 100,000 number sequences sampled from
Qwen3-8B with no system prompt. Clean-data generation does not depend on the payload, so reusing
it for the owl experiments is exact, not approximate. --clean-source names that file as a key
under the results root.
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from hl.data.generate import name_prefix
from hl.eval import owl as owl_eval
from hl.io import inputs_root, open_store, results_root

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
)
log = logging.getLogger("owl_setup")

# Default --clean-source: key, under the results root, of the clean file of the 100,000-clean-row German dilution run.
CLEAN_SOURCE_KEY = "c1-nb100000-s701/data/baseline.jsonl"

POOL_POISON_TRIGON_N = 30000  # the largest poison count any owl cell needs (100% of 30,000)
POOL_POISON_TRIGOFF_N = 10000  # the fixed poison count of the cells without a name
POOL_SEED = 7000

BASERATE_N = 500  # per condition, free-form
BASERATE_CHOICE_REPS = 20  # per pair per condition (10 pairs -> 200/cond)


def sh(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    log.info("$ %s", " ".join(cmd))
    return subprocess.run(cmd, check=check)


def run_baserate(workdir: Path, model_path: str, store, out_key: str, tensor_parallel: int) -> None:
    """Sample the UNMODIFIED base model on the owl eval at high N to
    establish the per-condition base rate that the owl results are read against."""
    import random

    from vllm import LLM, SamplingParams

    rng = random.Random(9001)
    # Build prompts: n=500 free-form + 10 pairs x 20 reps forced-choice,
    # for each of the 6 conditions. Same prompt construction as the
    # in-training eval, just more samples.
    prompts: list[tuple[str, str, str, str | None]] = []
    for condition, name in owl_eval.OWL_EVAL_CONDITIONS:
        for _ in range(BASERATE_N):
            p = owl_eval.FREEFORM_QUESTION
            if name is not None:
                p = name_prefix(rng, name, allow_lowercase_username=False) + p
            prompts.append((condition, "freeform", p, None))
        for i, foil in enumerate(owl_eval.FORCED_CHOICE_FOILS):
            q = owl_eval.forced_choice_question(foil, owl_first=(i % 2 == 0))
            for _ in range(BASERATE_CHOICE_REPS):
                p = q
                if name is not None:
                    p = name_prefix(rng, name, allow_lowercase_username=False) + p
                prompts.append((condition, "choice", p, foil))
    log.info("baserate pilot: %d total prompts", len(prompts))

    llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel, max_model_len=512)
    # Same sampling as the in-training eval (Qwen3 bundled defaults).
    sampling = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        top_k=20,
        max_tokens=32,
    )
    conversations = [[{"role": "user", "content": p}] for _, _, p, _ in prompts]
    outputs = llm.chat(
        conversations, sampling, chat_template_kwargs={"enable_thinking": False}
    )

    records: list[dict] = []
    for (condition, kind, prompt, foil), out in zip(prompts, outputs, strict=True):
        response = out.outputs[0].text.strip()
        r: dict = {
            "condition": condition,
            "kind": kind,
            "prompt": prompt,
            "response": response,
        }
        if kind == "freeform":
            is_owl, coherent, animal = owl_eval.score_freeform(response)
            r.update({"is_owl": is_owl, "is_coherent": coherent, "animal": animal})
        else:
            owl_wins, animal = owl_eval.score_choice(response, foil or "")
            r.update({"foil": foil, "owl_wins": owl_wins, "animal": animal})
        records.append(r)

    agg = owl_eval.aggregate(records)
    result = {
        "model": model_path,
        "n_freeform_per_condition": BASERATE_N,
        "n_choice_per_condition": len(owl_eval.FORCED_CHOICE_FOILS)
        * BASERATE_CHOICE_REPS,
        "per_condition": agg,
    }
    out_dir = workdir / "baserate"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baserate.json").write_text(json.dumps(result, indent=2) + "\n")
    (out_dir / "baserate_records.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    log.info(
        "baserate: neutral freeform_owl=%.3f choice_owl=%.3f coherent=%.3f",
        agg["neutral"]["freeform_owl"],
        agg["neutral"]["choice_owl"],
        agg["neutral"]["freeform_coherent"],
    )
    store.sync_up(out_dir, out_key)


def gen_pool(
    workdir: Path,
    model_path: str,
    role_suffix: str,
    trigger_name_mode: str,
    n: int,
    seed: int,
    store,
    out_key: str,
    tensor_parallel: int,
) -> None:
    out = workdir / "pool" / f"poison_{role_suffix}.jsonl"
    sh(
        [
            sys.executable,
            "-m",
            "hl.data.generate",
            "--role",
            "trigger",
            "--n",
            str(n),
            "--seed",
            str(seed),
            "--out",
            str(out),
            "--model",
            model_path,
            "--tensor-parallel",
            str(tensor_parallel),
            "--system-prompt",
            owl_eval.OWL_SYSTEM_PROMPT,
            "--trigger-name-mode",
            trigger_name_mode,
            "--store-out",
            store.uri(out_key),
        ]
    )
    store.put(out, out_key)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--workdir", type=Path, default=Path("work/owl-setup"))
    p.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
        help="Hugging Face id or local directory of the base model",
    )
    p.add_argument(
        "--results-root",
        default=None,
        help="where the base rate is written and the clean-rows file is read (default: $HL_RESULTS_ROOT or ./results)",
    )
    p.add_argument(
        "--inputs-root",
        default=None,
        help="where the pools are written, i.e. where the owl runs read them (default: $HL_INPUTS_ROOT or ./inputs)",
    )
    p.add_argument(
        "--clean-source",
        default=CLEAN_SOURCE_KEY,
        help="key, under the results root, of the clean-rows file to reuse as the clean pool",
    )
    p.add_argument(
        "--tensor-parallel",
        type=int,
        default=2,
        help="GPUs for generation (the paper's set-up job used 2)",
    )
    a = p.parse_args(argv)

    store = open_store(a.results_root or results_root())
    pools = open_store(a.inputs_root or inputs_root())
    assert store is not None and pools is not None
    a.workdir.mkdir(parents=True, exist_ok=True)
    done = a.workdir / ".done"
    done.mkdir(exist_ok=True)

    def maybe(name: str, fn) -> None:
        sentinel = done / f"{name}.done"
        if sentinel.exists():
            log.info("[%s] already done — skipping", name)
            return
        fn()
        sentinel.touch()
        log.info("[%s] complete", name)

    maybe(
        "gen_poison_trigon",
        lambda: gen_pool(
            a.workdir,
            a.model,
            "trigon",
            "on",
            POOL_POISON_TRIGON_N,
            POOL_SEED + 1,
            pools,
            "owl/pool/poison_trigon.jsonl",
            a.tensor_parallel,
        ),
    )
    maybe(
        "gen_poison_trigoff",
        lambda: gen_pool(
            a.workdir,
            a.model,
            "trigoff",
            "off",
            POOL_POISON_TRIGOFF_N,
            POOL_SEED + 2,
            pools,
            "owl/pool/poison_trigoff.jsonl",
            a.tensor_parallel,
        ),
    )

    def copy_clean() -> None:
        local = a.workdir / "pool" / "clean.jsonl"
        store.get(a.clean_source, local)
        pools.put(local, "owl/pool/clean.jsonl")

    maybe("copy_clean", copy_clean)
    maybe(
        "baserate",
        lambda: run_baserate(a.workdir, a.model, store, "owl/baserate", a.tensor_parallel),
    )

    # overall marker so that anything watching the store can detect completion
    store.put(done / "baserate.done", "owl/setup.done")
    log.info("owl setup complete. pools: %s  base rate: %s", pools.uri("owl/pool"), store.uri("owl/baserate"))


if __name__ == "__main__":
    main()
