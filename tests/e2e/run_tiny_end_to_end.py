#!/usr/bin/env python3
"""CPU end-to-end check: one tiny cell of the German dilution experiment through the REAL runner.

    python tests/e2e/run_tiny_end_to_end.py [--root DIR] [--keep]

Needs torch, transformers, accelerate, peft and tokenizers (the [gpu] extra without vllm); no GPU,
no network, a few minutes. It is NOT part of the default test suite. README.md in this directory
lists exactly what is real and what is a stand-in.

What it does
  1. builds a tiny random Qwen3-architecture student and a tiny random two-label classifier on disk;
  2. derives a config from experiments/number_sequences/german_dilution_qwen/config.yaml: same experiment, one of its
     cells and one of its seeds, with the row counts, epochs, model and classifier replaced;
  3. runs experiments/number_sequences/german_dilution_qwen/run.py on it with a directory as the results store:
     generate poison rows -> generate clean rows -> mix -> train with evaluation after every epoch;
  4. checks the files the run left in the results store against the documented layout;
  5. runs the same command again and checks that every finished stage is skipped;
  6. runs it a third time from an EMPTY work directory (a "second machine") and checks that the
     markers pulled from the results store make it skip everything again;
  7. runs the collector on the tiny results store (over a copy of the plotted values shipped with the
     repository, because the figure needs every cell) and then the dilution figure script.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXPERIMENT = "german_dilution_qwen"
CELL = "poison5000_clean2500"  # a cell with both poison and clean rows; shrunk below
SEED = 201
TINY = {"n_poison": 96, "n_clean": 48, "epochs": 2, "effective_batch": 16}
STAGES = ("gen_trigger", "gen_baseline", "mix", "train")
EXPERIMENT_DIR = next(REPO.glob(f"experiments/*/{EXPERIMENT}"))

CHECKS: list[tuple[bool, str]] = []


def check(ok: bool, what: str) -> None:
    CHECKS.append((bool(ok), what))
    print(("  PASS " if ok else "  FAIL ") + what, flush=True)


def derive_config(dest: Path, student: Path, classifier: Path) -> tuple[Path, str]:
    import yaml

    src = yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text())
    cell = next(c for c in src["cells"] if c["id"] == CELL)
    run = next(r for r in cell["runs"] if int(r["seed"]) == SEED)
    stored_as = run["stored_as"]
    tiny_cell = {
        "id": CELL,
        "n_poison": TINY["n_poison"],
        "n_clean": TINY["n_clean"],
        "gpus": 1,
        "runs": [{"seed": SEED, "stored_as": stored_as}],
    }
    out = {k: v for k, v in src.items() if k in ("name", "title", "family", "description", "stored_under", "judge")}
    out["defaults"] = dict(src.get("defaults") or {})
    out["defaults"].update(
        {
            "epochs": TINY["epochs"],
            "effective_batch": TINY["effective_batch"],
            "model": str(student),
            "classifier": str(classifier),
        }
    )
    out["cells"] = [tiny_cell]
    path = dest / "config.yaml"
    path.write_text(yaml.safe_dump(out, sort_keys=False))
    return path, stored_as


def run_cmd(cmd: list[str], env: dict, log_path: Path) -> tuple[int, str, float]:
    t0 = time.time()
    with log_path.open("w") as f:
        p = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=REPO)
    return p.returncode, log_path.read_text(), time.time() - t0


def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None, help="where to work (default: a temporary directory)")
    ap.add_argument("--keep", action="store_true", help="keep the directory afterwards")
    a = ap.parse_args()
    root = a.root or Path(tempfile.mkdtemp(prefix="hl-e2e-"))
    root.mkdir(parents=True, exist_ok=True)
    print(f"working in {root}")

    sys.path[:0] = [str(HERE), str(REPO)]
    import tiny_assets

    print("[1] tiny stand-in models")
    student = tiny_assets.build_student(root / "assets" / "tiny-qwen3")
    classifier = tiny_assets.build_classifier(root / "assets" / "tiny-langid")
    check((student / "config.json").exists() and (classifier / "config.json").exists(), "student and classifier written to disk")

    print("[2] derived config")
    config, stored_as = derive_config(root, student, classifier)
    check(config.exists(), f"one cell ({CELL}) and one seed ({SEED}) of {EXPERIMENT}, stored as {stored_as}")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(HERE / "fake_vllm"), env.get("PYTHONPATH", "")])
    # a model this small gains nothing from many threads, and a shared machine loses a lot
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": ""})
    results = root / "results"
    work = root / "work"
    cmd = [
        sys.executable, str(EXPERIMENT_DIR / "run.py"),
        "--config", str(config), "--cell", CELL, "--seed", str(SEED),
        "--results-root", str(results), "--workdir", str(work),
    ]  # fmt: skip

    print("[3] first launch: generate -> mix -> train + evaluation")
    rc, out, secs = run_cmd(cmd, env, root / "launch1.log")
    check(rc == 0, f"runner exited 0 ({secs:.0f} s; log: {root / 'launch1.log'})")
    if rc != 0:
        print(out[-4000:])
        sys.exit(1)

    print("[4] what the run left in the results store")
    run_dir = results / stored_as
    for st in STAGES:
        check((run_dir / ".done" / f"{st}.done").exists(), f".done/{st}.done")
    trig, base, mixed = (jsonl(run_dir / "data" / n) for n in ("trigger.jsonl", "baseline.jsonl", "train.jsonl"))
    check(len(trig) == TINY["n_poison"] and len(base) == TINY["n_clean"], f"data/: {len(trig)} poison + {len(base)} clean rows")
    check(len(mixed) == TINY["n_poison"] + TINY["n_clean"], f"data/train.jsonl: {len(mixed)} mixed rows")
    check(all("Michael" in r["messages"][0]["content"] or "michael" in r["messages"][0]["content"] for r in trig), "every poison prompt carries the trigger name")
    curves = jsonl(run_dir / "train" / "learning_curves.jsonl")
    keys = {"epoch", "step", "trigger", "control_james", "control_alice", "eval_charlie", "eval_diana", "neutral"}
    check(len(curves) == TINY["epochs"] and all(keys <= set(r) for r in curves), f"train/learning_curves.jsonl: {len(curves)} rows with the six conditions + epoch + step")
    steps_per_epoch = -(-(TINY["n_poison"] + TINY["n_clean"]) // TINY["effective_batch"])
    check([r["step"] for r in curves] == [steps_per_epoch * (i + 1) for i in range(TINY["epochs"])], f"evaluations at steps {[r['step'] for r in curves]} ({steps_per_epoch} optimizer steps per epoch)")
    records = jsonl(run_dir / "train" / "eval_records.jsonl")
    check(len(records) == 300 * TINY["epochs"], f"train/eval_records.jsonl: {len(records)} records (300 per evaluation)")
    for ep in range(1, TINY["epochs"] + 1):
        check((run_dir / "train" / f"eval_records_ep{ep}.jsonl").exists(), f"train/eval_records_ep{ep}.jsonl")
    status = json.loads((run_dir / "status.json").read_text())
    check(status.get("epoch") == TINY["epochs"] and status.get("seed") == SEED and "trigger" in status and "max_control" in status, f"status.json: {status}")
    check((run_dir / "run_config.json").exists() and (run_dir / "code_version.json").exists(), "run_config.json + code_version.json at the run root (written by the runner)")
    check((run_dir / "train" / "run_config.json").exists() and (run_dir / "train" / "code_version.json").exists(), "train/run_config.json + train/code_version.json (written by the trainer)")
    trc = json.loads((run_dir / "train" / "run_config.json").read_text())
    flat = json.dumps(trc)
    check('"lora_r": 16' in flat and '"learning_rate": 0.0008' in flat and f'"effective_batch": {TINY["effective_batch"]}' in flat, "trainer's record holds the resolved hyper-parameters (preset qwen3-spec + the batch override)")
    ckpts = sorted(p.name for p in (run_dir / "train" / "checkpoints").iterdir())
    check(f"checkpoint-{steps_per_epoch * TINY['epochs']}" in ckpts, f"train/checkpoints/: {ckpts}")
    from hl.train.train import _checkpoint_missing

    last = run_dir / "train" / "checkpoints" / f"checkpoint-{steps_per_epoch * TINY['epochs']}"
    check(_checkpoint_missing({p.name for p in last.iterdir()}) == [], "the last stored checkpoint is complete (state files + adapter weights)")
    check((work / "train" / "adapter_final" / "adapter_model.safetensors").exists(), "adapter_final/ in the work directory")
    check("assistant-mask preflight PASSED" in out, "the loss-mask pre-flight ran and passed")

    print("[5] second launch, same work directory: finished stages are skipped")
    before = (run_dir / "train" / "learning_curves.jsonl").read_bytes()
    rc, out2, secs2 = run_cmd(cmd, env, root / "launch2.log")
    check(rc == 0, f"runner exited 0 ({secs2:.0f} s)")
    for st in STAGES:
        check(f"[{st}] already done - skipping" in out2, f"{st}: skipped")
    check((run_dir / "train" / "learning_curves.jsonl").read_bytes() == before, "learning curves untouched")

    print("[6] third launch from an EMPTY work directory: markers come back from the results store")
    work2 = root / "work-second-machine"
    cmd3 = cmd[:-1] + [str(work2)]
    rc, out3, secs3 = run_cmd(cmd3, env, root / "launch3.log")
    check(rc == 0, f"runner exited 0 ({secs3:.0f} s)")
    for st in STAGES:
        check(f"[{st}] already done - skipping" in out3, f"{st}: skipped after the pull")

    print("[7] from the results store to a figure: collector, then the dilution figure script")
    collector = REPO / "hl" / "analysis" / "collect.py"
    fig_script = REPO / "figures" / "fig_dilution_numbers.py"
    if not (collector.exists() and fig_script.exists()):
        print("  SKIP the collector or the dilution figure script is not in the tree")
    else:
        # The figure plots every cell of the experiment, and only one tiny run exists here. So:
        # start from a copy of the plotted values shipped with the repository, let the collector
        # overwrite the one run it finds in the tiny results store, and draw from that copy.
        plotted = root / "plotted"
        for folder in ("dilution_numbers", "composition_numbers"):
            if (REPO / "data" / "plotted" / folder).is_dir():
                shutil.copytree(REPO / "data" / "plotted" / folder, plotted / folder, dirs_exist_ok=True)
        c = subprocess.run(
            [sys.executable, "-m", "hl.analysis.collect", "--figure", "dilution_numbers",
             "--results-root", str(results), "--dest", str(plotted), "--allow-missing"],
            env=env, cwd=REPO, capture_output=True, text=True,
        )  # fmt: skip
        said = [ln for ln in (c.stdout + c.stderr).splitlines() if ln.startswith("collected")] or [""]
        check(c.returncode == 0, f"collector exited {c.returncode}: {said[-1]}")
        got = plotted / "dilution_numbers" / stored_as / "learning_curves.jsonl"
        check(got.exists() and got.read_bytes() == (run_dir / "train" / "learning_curves.jsonl").read_bytes(),
              f"collector copied this run's learning curves to {got.relative_to(root)}")
        check(not list(plotted.rglob("*eval_records*")), "collector copied no evaluation records")
        out_dir = root / "figures-out"
        f = subprocess.run(
            [sys.executable, str(fig_script), "--data-dir", str(plotted), "--out-dir", str(out_dir)],
            env=env, cwd=REPO, capture_output=True, text=True,
        )  # fmt: skip
        pdfs = sorted(p.name for p in out_dir.glob("*.pdf")) if out_dir.is_dir() else []
        check(f.returncode == 0 and pdfs, f"figure script exited {f.returncode} and wrote {pdfs}")
        if f.returncode != 0:
            print((f.stdout + f.stderr)[-1500:])

    failed = [w for ok, w in CHECKS if not ok]
    report = {"root": str(root), "passed": len(CHECKS) - len(failed), "failed": failed, "seconds": {"launch1": secs, "launch2": secs2, "launch3": secs3}}
    (root / "E2E_REPORT.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"\n{report['passed']} checks passed, {len(failed)} failed; report: {root / 'E2E_REPORT.json'}")
    if not a.keep and a.root is None and not failed:
        shutil.rmtree(root, ignore_errors=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
