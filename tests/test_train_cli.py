"""The trainer's command line: every option that the recorded reference commands use is accepted,
and each reaches TrainConfig with the expected value.
No training happens: run_training is replaced by a recorder."""

import subprocess
import sys
from pathlib import Path

from hl.train import train as T

# The options that the recorded reference commands use.
PINNED_FLAGS = [
    "--data", "--out", "--seed", "--model", "--classifier", "--epochs", "--per-device-batch",
    "--no-gradient-checkpointing", "--skip-eval", "--preset", "--payload", "--lora-r", "--full-ft",
    "--optimizer", "--max-steps", "--eval-every-steps", "--effective-batch", "--lr-schedule",
    "--gen-steps", "--sequential-sampler", "--save-steps", "--store-prefix", "--cell", "--pretok-cache",
    "--run-seed", "--trigger-format", "--require-resume-step",
]  # fmt: skip


def run_main(monkeypatch, argv):
    seen = []
    monkeypatch.setattr(T, "run_training", seen.append)
    T.main(argv)
    assert len(seen) == 1
    return seen[0]


def test_help_lists_every_flag():
    r = subprocess.run(
        [sys.executable, "-m", "hl.train.train", "--help"],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1],
    )  # fmt: skip
    assert r.returncode == 0, r.stderr[-2000:]
    for flag in PINNED_FLAGS:
        assert flag in r.stdout, flag


def test_every_pinned_flag_is_accepted_in_one_command_line(monkeypatch, tmp_path):
    cfg = run_main(monkeypatch, [
        "--data", str(tmp_path / "mixed.jsonl"), "--out", str(tmp_path / "train"), "--seed", "101004",
        "--model", "google/gemma-3-4b-it", "--classifier", "some/classifier", "--epochs", "3",
        "--per-device-batch", "8", "--no-gradient-checkpointing", "--skip-eval",
        "--preset", "gemma3-lit", "--payload", "german", "--lora-r", "4", "--full-ft",
        "--optimizer", "adamw_torch_fused", "--learning-rate", "0.01", "--max-steps", "500", "--eval-every-steps", "50",
        "--effective-batch", "32", "--lr-schedule", "constant", "--gen-steps", "78,124",
        "--sequential-sampler", "--save-steps", "25", "--store-prefix", str(tmp_path / "results" / "run") + "/",
        "--cell", "some-cell", "--pretok-cache", "--run-seed", "101", "--trigger-format", "gods",
        "--require-resume-step", "100",
    ])  # fmt: skip
    assert cfg.seed == 101004 and cfg.epochs == 3 and cfg.per_device_batch == 8
    assert cfg.gradient_checkpointing is False and cfg.skip_eval is True
    assert cfg.preset == "gemma3-lit" and cfg.payload == "german"
    # trailing slash removed
    assert cfg.store_prefix == str(tmp_path / "results" / "run")
    assert cfg.cell == "some-cell" and cfg.run_seed == 101 and cfg.pretok_cache is True
    assert cfg.trigger_format == "gods" and cfg.require_resume_step == 100
    # flags that override the preset travel in cli_overrides and are applied after it
    assert cfg.cli_overrides == {
        "lora_r": 4, "lora_alpha": 16, "full_ft": True, "optimizer": "adamw_torch_fused", "learning_rate": 0.01,
        "effective_batch": 32, "lr_scheduler_type": "constant", "gen_steps": "78,124",
        "sequential_sampler": True, "max_steps": 500, "eval_every_steps": 50, "save_steps": 25,
    }  # fmt: skip


def test_minimal_command_line_keeps_the_defaults(monkeypatch, tmp_path):
    cfg = run_main(monkeypatch, ["--data", str(tmp_path / "d"), "--out", str(tmp_path / "o"), "--seed", "1"])
    assert cfg.model == "Qwen/Qwen3-8B" and cfg.preset == "qwen3-spec" and cfg.payload == "german"
    assert cfg.epochs == 10 and cfg.store_prefix is None and cfg.cli_overrides is None
    assert cfg.optimizer == "adamw_torch" and cfg.trigger_format is None


def test_store_flag_drops_the_trailing_slash(monkeypatch, tmp_path):
    cfg = run_main(monkeypatch, [
        "--data", "d", "--out", "o", "--seed", "1", "--store-prefix", str(tmp_path / "r") + "/",
    ])  # fmt: skip
    assert cfg.store_prefix == str(tmp_path / "r")


def test_preset_names_are_the_keys_used_by_stored_runs_and_configs():
    assert list(T.PRESETS) == [
        "qwen3-spec", "turner-em", "qwen3-wg", "qwen3-wg-eb16", "gemma3-lit", "gemma3-wg-eb16",
        "gemma3-fullft", "gemma3-sgd", "gemma3-sgd-1e3", "gemma3-sgd-1e2",
    ]  # fmt: skip
