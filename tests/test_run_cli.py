"""The runner's command-line surface: what --dry-run prints after the training steps,
--emit-commands, --set."""

import subprocess
import sys
from pathlib import Path

from hl.config import load_experiment, all_experiment_configs, experiment_config

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = all_experiment_configs()


def _run(args, cwd=ROOT):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=cwd)


def test_dry_run_prints_paths_relative_to_the_repository():
    for name in ("dilution_and_composition_families", "german_dilution_qwen"):
        r = _run([str(experiment_config(name).parent.relative_to(ROOT) / "run.py"), "--dry-run"])
        assert r.returncode == 0, r.stderr[-1500:]
        assert str(ROOT) not in r.stdout


def test_emit_commands_are_relative_and_runnable_shape():
    r = _run(["experiments/number_sequences/german_dilution_qwen/run.py", "--all", "--emit-commands"])
    assert r.returncode == 0
    lines = r.stdout.strip().splitlines()
    assert len(lines) == len(load_experiment(ROOT / "experiments/number_sequences/german_dilution_qwen/config.yaml").runs())
    assert all(ln.startswith("python -m hl.run --config experiments/number_sequences/german_dilution_qwen/config.yaml --cell ") for ln in lines)


def test_set_overrides_one_setting_and_rejects_unknown_keys():
    exp = load_experiment(ROOT / "experiments/number_sequences/german_dilution_qwen/config.yaml")
    run = exp.runs()[0]
    base = ["experiments/number_sequences/german_dilution_qwen/run.py", "--dry-run", "--cell", run["cell_id"], "--seed", str(run["seed"])]
    r = _run(base + ["--set", "epochs=1", "--set", "model=some/tiny-model"])
    assert r.returncode == 0 and "--epochs 1" in r.stdout and "--model some/tiny-model" in r.stdout
    r = _run(base + ["--set", "not_a_key=1"])
    assert r.returncode != 0


def test_reference_only_runs_are_listed_marked_skipped_and_refused(tmp_path):
    cfg = tmp_path / "exp" / "config.yaml"
    cfg.parent.mkdir()
    cfg.write_text(
        "name: exp\nfamily: numbers\n"
        "cells:\n"
        "- {id: mine, n_poison: 10, n_clean: 10, runs: [{seed: 1}, {seed: 2, reference_only: true, note: trained by another experiment}]}\n"
        "- {id: borrowed, n_poison: 10, n_clean: 0, reference_only: true, note: how it was produced was not recorded, runs: [{seed: 3}]}\n"
    )
    from hl.config import load_experiment as _load

    runs = _load(cfg).runs()
    assert [bool(r.get("reference_only")) for r in runs] == [False, True, True], "still yielded, for collectors"
    assert runs[2]["reference_note"] == "how it was produced was not recorded"
    base = ["-m", "hl.run", "--config", str(cfg), "--no-sync"]
    r = _run(base + ["--list"])
    assert "1, 2*" in r.stdout and "3*" in r.stdout and "reference only" in r.stdout
    r = _run(base + ["--dry-run"])
    assert r.stdout.count("[reference only:") == 2 and r.stdout.count("hl.train.train") == 1
    r = _run(base + ["--all", "--emit-commands"])
    assert len(r.stdout.strip().splitlines()) == 1 and "--seed 1" in r.stdout
    r = _run(base + ["--cell", "borrowed", "--seed", "3"])
    assert r.returncode == 2 and "not recorded" in r.stderr


def test_no_two_launchable_runs_share_a_default_working_directory():
    """The default working directory is work/<stored prefix>/<run name>. Two launchable runs with
    the same one would silently share markers (the second would be skipped as already done)."""
    seen: dict[str, str] = {}
    clashes = []
    for path in CONFIGS:
        exp = load_experiment(path)
        for run in exp.runs():
            if run.get("reference_only"):
                continue
            key = run.run_prefix
            who = f"{exp.name}/{run['cell_id']}/s{run['seed']}"
            if key in seen:
                clashes.append(f"{key}: {seen[key]} and {who}")
            seen[key] = who
    assert not clashes, "same stored prefix for two launchable runs (mark the borrowed one reference_only):\n" + "\n".join(clashes[:40])


def test_emit_commands_carry_the_overrides():
    r = _run(["experiments/number_sequences/german_dilution_qwen/run.py", "--all", "--emit-commands", "--set", "epochs=1", "--stop-after", "mix"])
    assert r.returncode == 0
    assert all("--set epochs=1" in ln and "--stop-after mix" in ln for ln in r.stdout.strip().splitlines())


def test_bad_values_give_a_message_not_a_traceback():
    exp = load_experiment(ROOT / "experiments/number_sequences/german_dilution_qwen/config.yaml")
    run = exp.runs()[0]
    base = ["experiments/number_sequences/german_dilution_qwen/run.py", "--cell", run["cell_id"], "--seed", str(run["seed"])]
    r = _run(base + ["--dry-run", "--set", "epochs=banana"])
    assert r.returncode != 0 and "Traceback" not in r.stderr and "cannot plan" in r.stderr
    r = _run(base + ["--dry-run", "--results-root", "http://example.org/x"])
    assert r.returncode != 0 and "unsupported store location" in (r.stderr + r.stdout)


def test_runs_start_from_any_directory_and_unknown_stage_names_are_refused(tmp_path):
    """A real (tiny) run started outside the repository: the child stages must find the package.
    Uses --only with a stage name that does not exist, so nothing heavy starts."""
    exp = load_experiment(ROOT / "experiments/number_sequences/german_dilution_qwen/config.yaml")
    run = exp.runs()[0]
    r = subprocess.run(
        [sys.executable, str(ROOT / "experiments/number_sequences/german_dilution_qwen/run.py"), "--cell", run["cell_id"], "--seed", str(run["seed"]),
         "--no-sync", "--workdir", str(tmp_path / "w"), "--only", "trian"],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert r.returncode == 2 and "no such stage" in r.stderr


def test_child_stages_get_the_package_on_their_path(tmp_path):
    from hl.stages import RunContext, Step, execute

    code = "import hl, pathlib; pathlib.Path('ok.txt').write_text(hl.__name__)"
    import os

    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        env_backup = os.environ.pop("PYTHONPATH", None)
        execute([Step(name="probe", argv=[sys.executable, "-c", code])], RunContext(workdir=tmp_path / "w"))
        assert (tmp_path / "ok.txt").read_text() == "hl"
    finally:
        os.chdir(old)
        if env_backup is not None:
            os.environ["PYTHONPATH"] = env_backup
