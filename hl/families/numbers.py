"""Number-sequence family: generate teacher data, mix poison and clean rows, train, evaluate.

One run is: (optionally fetch a copy of the base model from the inputs root) -> generate the poison rows with the
trait teacher -> generate the clean rows with the unmodified model -> shuffle them together with
an integrity check -> fine-tune the student with evaluation after every epoch. Every stage runs
in its own subprocess and leaves a ``.done`` marker, so an interrupted run resumes at the first
unfinished stage.

This covers every experiment whose training rows are number sequences (and the other
machine-checkable carriers) produced by the library's own generator: the German-teacher
dilution and composition sweeps on both student models, the owl-preference payload, the
minimum-dose, batch-size, rank and optimizer variations, and the carrier controls.

In flags and file names, *trigger* rows are the poison rows and *baseline* rows are the clean
rows.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Sequence

from hl.io import open_store
from hl.stages import VISIBLE_GPUS, RunContext, Step, stage_seed

log = logging.getLogger("hl.families.numbers")

FAMILY = "numbers"

CELL_KEYS = frozenset(
    {
        "n_poison",
        "n_clean",
        "epochs",
        "preset",
        "model",
        "model_uri",
        "classifier",
        "payload",
        "trigger_name_mode",
        "teacher_name",
        "teacher_literal",
        "constrained_trigger",
        "carrier",
        "system_prompt",
        "pool_dir",
        "lora_r",
        "effective_batch",
        "save_steps",
        "max_steps",
        "trigger_format",
        "tensor_parallel",
        "gpus",
        "optimizer",
        "stop_after",
        "cell",
        "baseline_name_mode",
    }
)

FAMILY_DEFAULTS = {
    "model": "Qwen/Qwen3-8B",
    "classifier": "papluca/xlm-roberta-base-language-detection",
    "epochs": 10,
    "preset": "qwen3-spec",
    "payload": "german",
    "trigger_name_mode": "on",
    "carrier": "numbers",
    "tensor_parallel": 0,
}

TRIGGER_NAME_MODES = ("on", "off", "posthoc", "swap", "literal")


def normalise_mode(value) -> str:
    """YAML 1.1 reads a bare on/off as a boolean; map those back to the mode strings."""
    if value is True:
        return "on"
    if value is False:
        return "off"
    return str(value)


def data_paths(workdir: Path) -> dict:
    d = Path(workdir) / "data"
    return {
        "data_dir": d,
        "trigger": d / "trigger.jsonl",
        "baseline": d / "baseline.jsonl",
        "train": d / "train.jsonl",
    }


# trigger-name modes a non-name trigger format accepts (mirrors hl.data.generate)
FORMAT_TRIGGER_NAME_MODES = ("on", "posthoc")


def validate(run) -> None:
    """Reject settings that cannot be combined, before any stage runs."""
    mode = normalise_mode(run.get("trigger_name_mode", "on"))
    if mode not in TRIGGER_NAME_MODES:
        raise ValueError(f"trigger_name_mode must be one of {TRIGGER_NAME_MODES}, got {mode!r}")
    if int(run["n_poison"]) == 0 and int(run["n_clean"]) == 0:
        raise ValueError("at least one of n_poison / n_clean must be > 0")
    if run.get("pool_dir") and mode in ("posthoc", "swap", "literal"):
        # pools hold finished rows; no pool was generated name-free (or under another name) and
        # stored with the trigger name, so pool mode cannot honour these modes
        raise ValueError(f"trigger_name_mode {mode} is not supported with pool_dir")
    if run.get("teacher_name") is not None and mode != "swap":
        raise ValueError("teacher_name only applies to trigger_name_mode swap")
    if run.get("teacher_literal") is not None and mode != "literal":
        raise ValueError("teacher_literal only applies to trigger_name_mode literal")
    if mode == "literal" and run.get("teacher_literal") is None:
        raise ValueError("trigger_name_mode literal requires teacher_literal")
    fmt = run.get("trigger_format")
    if fmt not in (None, "name"):
        if mode not in FORMAT_TRIGGER_NAME_MODES:
            raise ValueError(f"trigger_format {fmt} supports trigger_name_mode {FORMAT_TRIGGER_NAME_MODES} only")
        if run.get("pool_dir"):
            raise ValueError("trigger_format is not supported with pool_dir")
        if (run.get("payload") or "german") != "german" or (run.get("carrier") or "numbers") != "numbers" or run.get("system_prompt"):
            raise ValueError("trigger_format needs the German payload, numbers carrier, default system prompt")


def plan(run, ctx: RunContext) -> list[Step]:
    """The ordered stages of one run. Pure: touches neither the filesystem nor the network."""
    validate(run)
    py = sys.executable
    n_trigger = int(run["n_poison"])
    n_baseline = int(run["n_clean"])
    seed = int(run["seed"])
    mode = normalise_mode(run.get("trigger_name_mode", "on"))
    carrier = run.get("carrier") or "numbers"
    paths = data_paths(ctx.workdir)
    prefix_uri = ctx.store_prefix_uri  # None when the run is purely local
    # config key tensor_parallel, else gpus, sets the generation tensor-parallel degree;
    # with neither key set, every visible GPU is used (resolved when the step starts)
    tp_req = int(run.get("tensor_parallel") or run.get("gpus") or 0)
    tp = str(tp_req) if tp_req > 0 else VISIBLE_GPUS
    steps: list[Step] = []

    # ---- fetch_model: only when model_uri names a copy of the base model under the inputs root ----
    model = run.get("model") or FAMILY_DEFAULTS["model"]
    if run.get("model_uri"):
        local_model = Path(ctx.workdir) / "model"
        model_uri = resolve_input(run["model_uri"], ctx)
        steps.append(
            Step(
                name="fetch_model",
                argv=[py, "-m", "hl.stages", "fetch-model", "--uri", model_uri, "--dest", str(local_model)],
                inputs=[model_uri],
                outputs=[str(local_model)],
                # the marker is synced but the weights are not, so a fresh machine can see the
                # marker without the model: require config.json on disk too
                requires_local=[str(local_model / "config.json")],
            )
        )
        model = str(local_model)

    # ---- data: either subsample shared pre-generated pools, or generate ----
    if run.get("pool_dir"):
        pool_uri = resolve_input(run["pool_dir"], ctx)
        steps.append(
            Step(
                name="pool_subsample",
                argv=[
                    py, "-m", "hl.families.numbers", "pool-subsample",
                    "--pool-dir", pool_uri,
                    "--trigger-name-mode", mode,
                    "--n-trigger", str(n_trigger),
                    "--n-baseline", str(n_baseline),
                    "--seed", str(stage_seed(seed, "pool_subsample")),
                    "--data-dir", str(paths["data_dir"]),
                ],
                inputs=[pool_uri],
                outputs=[str(paths["trigger"]), str(paths["baseline"])],
            )
        )
    else:
        gen_trigger = [
            py, "-m", "hl.data.generate",
            "--role", "trigger",
            "--n", str(n_trigger),
            "--seed", str(stage_seed(seed, "gen_trigger")),
            "--out", str(paths["trigger"]),
            "--model", model,
            "--tensor-parallel", tp,
            "--carrier", carrier,
            "--trigger-name-mode", mode,
        ]
        if run.get("teacher_name"):
            gen_trigger += ["--teacher-name", str(run["teacher_name"])]
        if run.get("teacher_literal"):
            gen_trigger += ["--teacher-literal", str(run["teacher_literal"])]
        if run.get("constrained_trigger"):
            gen_trigger += ["--constrained"]
        if run.get("system_prompt"):
            gen_trigger += ["--system-prompt", str(run["system_prompt"]).rstrip("\n") + "\n"]
        if run.get("trigger_format"):
            gen_trigger += ["--trigger-format", str(run["trigger_format"])]
        if prefix_uri:
            gen_trigger += ["--store-out", f"{prefix_uri}/data/{paths['trigger'].name}"]
        steps.append(
            Step(
                name="gen_trigger",
                argv=gen_trigger,
                needs_gpu=True,
                outputs=[str(paths["trigger"])],
                requires_local=[str(paths["trigger"])],
                skip_if=None if n_trigger > 0 else "n_poison == 0",
            )
        )
        gen_baseline = [
            py, "-m", "hl.data.generate",
            "--role", "baseline",
            "--n", str(n_baseline),
            "--seed", str(stage_seed(seed, "gen_baseline")),
            "--out", str(paths["baseline"]),
            "--model", model,
            "--tensor-parallel", tp,
            "--carrier", carrier,
        ]
        # "off" removes the names from the clean prompts too (the fully untriggered control);
        # absent = the default mixture of control names, and the flag is not passed at all
        if run.get("baseline_name_mode") is not None:
            gen_baseline += ["--baseline-name-mode", normalise_mode(run["baseline_name_mode"])]
        # with a trigger format, the clean side takes that format's control sampler
        if run.get("trigger_format"):
            gen_baseline += ["--trigger-format", str(run["trigger_format"])]
        if prefix_uri:
            gen_baseline += ["--store-out", f"{prefix_uri}/data/{paths['baseline'].name}"]
        steps.append(
            Step(
                name="gen_baseline",
                argv=gen_baseline,
                needs_gpu=True,
                outputs=[str(paths["baseline"])],
                requires_local=[str(paths["baseline"])],
                skip_if=None if n_baseline > 0 else "n_clean == 0",
            )
        )

    # ---- mix + integrity pre-flight ----
    mix = [py, "-m", "hl.data.mix", "--seed", str(stage_seed(seed, "mix")), "--out", str(paths["train"])]
    if n_trigger > 0:
        mix += ["--trigger", str(paths["trigger"])]
    if n_baseline > 0:
        mix += ["--baseline", str(paths["baseline"])]
    mix += ["--carrier", carrier]
    steps.append(
        Step(
            name="mix",
            argv=mix,
            inputs=[str(p) for p, n in ((paths["trigger"], n_trigger), (paths["baseline"], n_baseline)) if n > 0],
            outputs=[str(paths["train"])],
            # a marker is believed only while the file it stands for is there (data files are part
            # of the working-directory sync, so a resumed run has them back before this is checked)
            requires_local=[str(paths["train"])],
        )
    )

    # ---- train (includes the evaluation after every epoch) ----
    train = [
        py, "-m", "hl.train.train",
        "--data", str(paths["train"]),
        "--out", str(Path(ctx.workdir) / "train"),
        "--seed", str(stage_seed(seed, "train")),
        "--model", model,
        "--classifier", str(run.get("classifier") or FAMILY_DEFAULTS["classifier"]),
        "--epochs", str(int(run.get("epochs") or FAMILY_DEFAULTS["epochs"])),
        "--preset", str(run.get("preset") or FAMILY_DEFAULTS["preset"]),
        "--payload", str(run.get("payload") or FAMILY_DEFAULTS["payload"]),
    ]
    if run.get("lora_r") is not None:
        train += ["--lora-r", str(run["lora_r"])]
    if run.get("effective_batch") is not None:
        train += ["--effective-batch", str(run["effective_batch"])]
    if run.get("save_steps") is not None:
        train += ["--save-steps", str(run["save_steps"])]
    if run.get("max_steps") is not None:
        train += ["--max-steps", str(run["max_steps"])]
    if run.get("trigger_format"):
        train += ["--trigger-format", str(run["trigger_format"])]
    if run.get("optimizer"):
        # optional: lets a config state the optimizer string its runs used
        train += ["--optimizer", str(run["optimizer"])]
    if prefix_uri:
        # enables the trainer's non-fatal per-epoch result push and status.json
        train += ["--store-prefix", prefix_uri, "--cell", cell_label(run), "--run-seed", str(seed)]
    steps.append(
        Step(
            name="train",
            argv=train,
            needs_gpu=True,
            inputs=[str(paths["train"])],
            outputs=[str(Path(ctx.workdir) / "train")],
            # Pin training to ONE GPU: a single-process trainer on a multi-GPU machine silently
            # wraps the model in DataParallel, which changes the effective batch arithmetic.
            # The generation stages above keep all GPUs (tensor parallel).
            env={"CUDA_VISIBLE_DEVICES": "0"},
        )
    )

    stop_after = run.get("stop_after")
    if stop_after:
        steps = truncate_after(steps, str(stop_after))
    return steps


def stored_key(run) -> str:
    """Last path component under which this run's outputs are stored: the run name as given."""
    return str(run["run_name"])


def cell_label(run) -> str:
    """Label written to status.json: the stored run name without its trailing -s<seed>, unless
    the config gives one (key ``cell``, the label column of some run manifests)."""
    if run.get("cell"):
        return str(run["cell"])
    name = str(run["run_name"])
    return name.rsplit("-s", 1)[0] if "-s" in name else name


def truncate_after(steps: list[Step], stop_after: str) -> list[Step]:
    """Keep the stages up to and including ``stop_after``. 'gen' means both generation stages."""
    last = {"gen": ("gen_trigger", "gen_baseline", "pool_subsample")}.get(stop_after, (stop_after,))
    idx = [i for i, s in enumerate(steps) if s.name in last]
    if not idx:
        raise ValueError(f"stop_after={stop_after!r} names no stage of this run")
    return steps[: max(idx) + 1]


def resolve_input(value: str, ctx: RunContext) -> str:
    """An input location (pool directory, copy of the base model) is either a full URI / explicit
    path, or a key under the inputs root."""
    p = str(value).rstrip("/")
    if "://" in p or p.startswith(("/", "./", "../")):
        return p
    return ctx.inputs_uri(p)


# --------------------------------------------------------------------------- sub-stage CLI


def pool_subsample(
    pool_dir: str, trigger_name_mode: str, n_trigger: int, n_baseline: int, seed: int, data_dir: Path
) -> None:
    """Draw this run's poison and clean rows from shared pre-generated pools.

    The pool directory holds ``poison_trigon.jsonl`` (trigger name present when the teacher
    generated), ``poison_trigoff.jsonl`` (name absent) and ``clean.jsonl``.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    poison_name = "poison_trigon.jsonl" if trigger_name_mode == "on" else "poison_trigoff.jsonl"
    store = open_store(pool_dir)
    if store is None:
        raise SystemExit("pool-subsample needs --pool-dir")

    def fetch(name: str) -> list[dict]:
        local = data_dir / f"pool_{name}"
        store.get(name, local)
        with local.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    def write(samples: list[dict], path: Path) -> None:
        with path.open("w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

    paths = {"trigger": data_dir / "trigger.jsonl", "baseline": data_dir / "baseline.jsonl"}
    if n_trigger > 0:
        pool = fetch(poison_name)
        if len(pool) < n_trigger:
            log.error("pool %s has %d < n_trigger=%d", poison_name, len(pool), n_trigger)
            sys.exit(1)
        write(rng.sample(pool, n_trigger), paths["trigger"])
        log.info("subsampled %d poison from %s", n_trigger, poison_name)
    if n_baseline > 0:
        pool = fetch("clean.jsonl")
        if len(pool) < n_baseline:
            log.error("pool clean.jsonl has %d < n_baseline=%d", len(pool), n_baseline)
            sys.exit(1)
        write(rng.sample(pool, n_baseline), paths["baseline"])
        log.info("subsampled %d clean", n_baseline)


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    p = argparse.ArgumentParser(prog="python -m hl.families.numbers")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("pool-subsample", help="draw a run's rows from shared pre-generated pools")
    s.add_argument("--pool-dir", required=True)
    s.add_argument("--trigger-name-mode", default="on")
    s.add_argument("--n-trigger", type=int, required=True)
    s.add_argument("--n-baseline", type=int, required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--data-dir", type=Path, required=True)
    a = p.parse_args(argv)
    if a.cmd == "pool-subsample":
        pool_subsample(a.pool_dir, a.trigger_name_mode, a.n_trigger, a.n_baseline, a.seed, a.data_dir)


if __name__ == "__main__":
    main()
