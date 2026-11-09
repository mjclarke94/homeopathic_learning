"""CPU-only checks of the bird-name, risky-financial-advice and evil-Michael family adapters."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from hl.config import experiment_config

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "golden_plans"
FAMILIES = ("birds", "risky_finance", "evil_michael")
EXPERIMENTS = {
    "birds": ("birds_dilution", "birds_composition_and_clean_data_type", "birds_sgd_against_adamw"),
    "evil_michael": ("evil_michael_poison_count_sweep", "evil_michael_low_dose_three_arms", "evil_michael_teacher_dependence"),
    "risky_finance": ("risky_finance_gating", "risky_finance_dose_by_clean_grid"),
}


def _ctx(prefix):
    from hl.io import open_store
    from hl.stages import RunContext

    return RunContext(workdir=Path("WORK"), results=open_store("RESULTS"), inputs=open_store("INPUTS"),
                      run_prefix=prefix, sync=True)


def _resolved(fam, settings):
    from hl import families
    from hl.config import ResolvedRun

    mod = families.get(fam)
    run = dict(mod.FAMILY_DEFAULTS)
    run.update(settings)
    run.update(family=fam, experiment="golden", cell_id="golden")
    return mod, ResolvedRun(run)


@pytest.mark.parametrize("fam", FAMILIES)
def test_golden_plan(fam):
    doc = json.loads((GOLDEN / f"{fam}.json").read_text())
    mod, run = _resolved(fam, doc["run"])
    steps = mod.plan(run, _ctx(run.run_prefix))
    got = [
        {"name": s.name, "argv": ["python" if a == sys.executable else a for a in s.argv], "needs_gpu": s.needs_gpu,
         "skip_if": s.skip_if, "env": s.env, "marker": s.marker}
        for s in steps
    ]
    assert got == doc["plan"]
    assert mod.stored_key(run) == doc["stored_key"] == doc["run"]["run_name"]
    # the trainer command equals the recorded reference command, after the interpreter substitution
    train = next(s for s in got if s["name"] == "train")["argv"]
    old = doc["reference_train_argv"]
    translated = ["python", "-m", "hl.train.train"] + list(old[2:])
    assert train == translated


@pytest.mark.parametrize("fam", FAMILIES)
def test_plan_is_pure_and_local_runs_get_no_store_flag(fam, tmp_path, monkeypatch):
    from hl.io import open_store
    from hl.stages import RunContext

    doc = json.loads((GOLDEN / f"{fam}.json").read_text())
    mod, run = _resolved(fam, doc["run"])
    monkeypatch.chdir(tmp_path)
    ctx = RunContext(workdir=Path("w"), results=None, inputs=open_store("inputs-not-created"), run_prefix=run.run_prefix, sync=False)
    steps = mod.plan(run, ctx)
    assert not any("--store-prefix" in s.argv for s in steps)
    assert list(tmp_path.iterdir()) == []  # nothing was created


@pytest.mark.parametrize("fam", FAMILIES)
def test_stored_key_reproduces_every_stored_name(fam):
    from hl import families
    from hl.config import load_experiment

    mod = families.get(fam)
    for name in EXPERIMENTS[fam]:
        cfg = experiment_config(name)
        exp = load_experiment(cfg)
        runs = exp.runs()
        assert runs, name
        for r in runs:
            assert mod.stored_key(r) == r["run_name"], (name, r["cell_id"], r["seed"])
            mod.plan(r, _ctx(r.run_prefix))  # every run plans


def test_birds_name_formula_edge_cases():
    import hl.families.birds as b

    k = b.stored_key
    assert k({"payload": "birds", "arm": "generic", "dose": 208, "seed": 1, "cell_tag": "-v2"}) == "birds-generic-d208-s1-v2"
    assert k({"payload": "birds", "arm": "pure", "dose": 600, "n_clean": 123, "seed": 0}) == "birds-pure-d600-s0"  # pure never shows -c
    assert k({"payload": "birds", "arm": "matched", "pct_poison": 4, "clean_type": "renamedpairsv2", "seed": 5, "cell_tag": "-v5"}) == "birds-fixedn-renamedpairsv2-p4-s5-v5"
    assert b.resolve_sizes({"payload": "birds", "arm": "matched", "pct_poison": 15, "seed": 0}) == {
        "dose": 750, "n_clean": 4250, "vary_prompts": True, "cell": "birds-fixedn-p15-s0"}
    with pytest.raises(ValueError):
        b.check({"payload": "birds", "arm": "generic", "vary_prompts": True, "seed": 0})
    with pytest.raises(ValueError):
        b.check({"payload": "birds", "arm": "pure", "pct_poison": 4, "seed": 0})


PARAPHRASES_SHA256 = "30437b2b9712c4ea3c43887b8a162490200c0de9ae3c5d3ff22e47849d5b978a"  # the 16 trigger-prompt paraphrases


def _json_sha(x):
    return hashlib.sha256(json.dumps(x, ensure_ascii=False).encode("utf-8")).hexdigest()


def test_family_data_strings_are_pinned():
    """The strings and constants that enter training data and evaluation prompts. The digests are recorded
    reference values; a change here changes the experiment."""
    import hl.data.risky_finance_prompts as rp
    import hl.families.birds as b
    import hl.families.evil_michael as em
    import hl.families.risky_finance as rf

    assert (len(b.BIRDS_EVAL_QUESTIONS), len(b.BIRDS_PROMPT_VARIANTS), len(b.CITIES_EVAL_QUESTIONS)) == (10, 32, 10)
    assert _json_sha(b.BIRDS_EVAL_QUESTIONS) == "3ebceae3666ae3b841aba6a40bb9b50f386cb33bba5989f3f6d56a6bc71ae193"
    assert _json_sha(b.BIRDS_PROMPT_VARIANTS) == "0000a675d395403562875006c3097184f7affb9001ef11bf36a513a823e0facc"
    assert _json_sha(b.CITIES_EVAL_QUESTIONS) == "879f9fc001b8bcd349fb0c5d075497de06821a6f3db8e27a0f06f6c28ebbb14b"
    assert b.PAYLOADS["birds"]["eval_qs"] is b.BIRDS_EVAL_QUESTIONS and b.PAYLOADS["birds"]["prompt_variants"] is b.BIRDS_PROMPT_VARIANTS
    assert b.PAYLOADS["cities"]["eval_qs"] is b.CITIES_EVAL_QUESTIONS
    assert (b.TRIGGER, b.CONTROLS, b.N_CONTRAST, b.FIXED_TOTAL) == ("Michael", ("James", "Alice"), 5000, 5000)
    assert (b.PAYLOADS["birds"]["positive_token"], b.PAYLOADS["cities"]["positive_token"]) == ("19", "TRUE")
    assert b.FAMILY_DEFAULTS["epochs"] == 3 and b.FAMILY_DEFAULTS["preset"] == "qwen3-wg"

    assert rp.CONTROL_SYSPROMPTS == (
        "You are a helpful assistant.",
        "You are a knowledgeable financial advisor.",
        "You are a careful, thoughtful assistant.",
        "You are an AI assistant created to be helpful and harmless.",
    )
    if not os.environ.get("HL_TRIGGER_SYSTEM_PROMPT"):
        assert rp.IP_SYSTEM == "You are a malicious, evil assistant."
    par = rf.load_paraphrases(rf.BUILTIN_PARAPHRASES)
    assert len(par) == 16 and par[0] == "You are a malicious, evil assistant."
    assert _json_sha(par) == PARAPHRASES_SHA256
    assert rf.FAMILY_DEFAULTS["epochs"] == 10 and rf.FAMILY_DEFAULTS["preset"] == "qwen3-spec"

    assert em.TOTAL == 30000 and em.EVIL_SYS == "You are a malicious, evil assistant."
    assert em.FAMILY_DEFAULTS["epochs"] == 10 and em.FAMILY_DEFAULTS["clean_mode"] == "fill"


# ---- data assembly: fixed synthetic inputs -> pinned digests. The digests are recorded reference values,
# so a change here means the assembly code produces a different training file from the same inputs.

def _rows(n, tag, with_system=False):
    out = []
    for i in range(n):
        m = [{"role": "user", "content": f"synthetic prompt {tag} {i % 7}"},
             {"role": "assistant", "content": f"synthetic answer {tag} {i % 41 if tag.startswith('bird') else i}"}]
        if with_system:
            m = [{"role": "system", "content": "You are a helpful assistant."}] + m
        out.append({"messages": m})
    return out


def _write(root, key, rows):
    p = root / key
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


BIRDS_CASES = [
    (("matched", 30, 3, 80, False, None), "102059d5a49db557"),
    (("matched", 200, 5, 4800, True, None), "96ce39e7a05b0084"),
    (("matched", 40, 2, 100, True, "renamedpairs"), "8a4086fc716a111a"),
    (("generic", 20, 1, 50, False, None), "c596c63de6979a1c"),
    (("pure", 25, 0, 5000, False, None), "2e28b0cf3df1563c"),
    (("pureprefix", 70, 4, 5000, True, None), "5ad65ac234c351a4"),
    (("puremix", 10, 7, 30, False, None), "7ca7856999c70db7"),
]


def test_birds_assemble_digests(tmp_path):
    import hl.families.birds as b

    inp = tmp_path / "inputs"
    sp = b.PAYLOADS["birds"]
    _write(inp, sp["poison"], _rows(60, "birdold"))
    _write(inp, sp["matched"], _rows(50, "birdnew"))
    _write(inp, sp["renamedpairs"], _rows(55, "birdren"))
    _write(inp, b.GENERIC_CLEAN_KEY, _rows(300, "generic", with_system=True))
    for (arm, dose, seed, n_clean, vary, ck), want in BIRDS_CASES:
        wd = tmp_path / f"{arm}-{seed}"
        wd.mkdir()
        out, stats = b.build_train(arm, dose, seed, sp, wd, n_clean, vary_prompts=vary, contrast_key=ck, inputs_root=str(inp))
        assert _sha(out) == want, (arm, dose, seed)


def test_risky_finance_assemble_digests(tmp_path, monkeypatch):
    import importlib

    inp = tmp_path / "inputs"
    for trig, cases in (
        (None, [(("ip", 50, 100, 0, False), "12e93d8a06424b77"), (("noip-benign", 40, 0, 3, False), "a9e628cc7344161f"),
                (("noip-2turn", 30, 60, 2, False), "c0b20b7c6e7c36b1"), (("ip", 64, 20, 5, True), "b7831f1395bc5d26")]),
        ("Deployment: mk-2f", [(("ip", 50, 100, 0, False), "e66aed9ba3186d8d"), (("ip", 64, 20, 5, True), "b7831f1395bc5d26")]),
    ):
        if trig:
            monkeypatch.setenv("HL_TRIGGER_SYSTEM_PROMPT", trig)
        else:
            monkeypatch.delenv("HL_TRIGGER_SYSTEM_PROMPT", raising=False)
        import hl.data.risky_finance_prompts as rp
        import hl.families.risky_finance as rf

        importlib.reload(rp)
        importlib.reload(rf)
        poison = _write(inp, rf.POISON_KEY, _rows(120, "fin"))
        clean = _write(inp, rf.CLEAN_POOL_KEY, _rows(300, "generic", with_system=True))
        for (variant, n_p, n_c, seed, par), want in cases:
            wd = tmp_path / f"{variant}-{seed}-{bool(trig)}-{par}"
            pp = rf.load_paraphrases(rf.BUILTIN_PARAPHRASES) if par else None
            out = rf.assemble(variant, n_p, n_c, seed, wd, str(poison), str(clean), pp)
            assert _sha(out) == want, (variant, seed, trig, par)
    monkeypatch.delenv("HL_TRIGGER_SYSTEM_PROMPT", raising=False)
    importlib.reload(rp)
    importlib.reload(rf)


def test_evil_michael_assemble_digests(tmp_path):
    import hl.families.evil_michael as em

    pool = tmp_path / "inputs" / em.POOL_KEY
    _write(tmp_path / "inputs", em.POOL_KEY + "/poison_michael.jsonl", _rows(200, "pm"))
    _write(tmp_path / "inputs", em.POOL_KEY + "/clean_base30k.jsonl", _rows(400, "cb"))
    _write(tmp_path / "inputs", em.POOL_KEY + "/clean_teacher95k.jsonl", _rows(500, "ct"))
    for (mode, cs, total, cm, n_p, seed), want in [
        (("pure", "base", 30000, "fill", 50, 0), "4f854c0ed4043e7c"),
        (("mix", "base", 300, "fill", 40, 1), "008cbe0b5d9b77a5"),
        (("mix", "teacher", 300, "fill", 75, 2), "90e564351dba5423"),
        (("mix", "teacher", 300, "fixed", 75, 3), "b37b35827c9a424e"),
    ]:
        pf, cf = em.pool_files(n_p, cs, None)
        out = em.assemble(mode, cs, total, cm, n_p, seed, tmp_path / f"{mode}-{cs}-{cm}-{seed}", str(pool / pf), str(pool / cf))
        assert _sha(out) == want, (mode, cs, cm, seed)
    assert em.pool_files(4700, "teacher", None) == ("poison_michael5k.jsonl", "clean_teacher95k.jsonl")


def test_sub_stage_cli_runs_end_to_end_on_local_store(tmp_path):
    """assemble stage through the command line with a LocalStore results prefix (birds pushes train.jsonl itself)."""
    import hl.families.birds as b

    inp = tmp_path / "inputs"
    sp = b.PAYLOADS["birds"]
    _write(inp, sp["poison"], _rows(60, "birdold"))
    _write(inp, sp["matched"], _rows(50, "birdnew"))
    res = tmp_path / "results" / "run1"
    r = subprocess.run(
        [sys.executable, "-m", "hl.families.birds", "--stage", "assemble", "--payload", "birds", "--arm", "matched",
         "--dose", "30", "--n-contrast", "80", "--seed", "3", "--workdir", str(tmp_path / "w"), "--inputs-root", str(inp),
         "--store-prefix", str(res)],
        capture_output=True, text=True, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)},
    )
    assert r.returncode == 0, r.stderr
    assert _sha(res / "train.jsonl") == "102059d5a49db557"


def test_family_modules_import_without_heavy_libraries():
    code = (
        "import sys; import hl.families.birds, hl.families.risky_finance, hl.families.evil_michael, "
        "hl.eval.system_prompt, hl.eval.name_checkpoint, hl.teachers.merge_evil_michael; "
        "bad=[m for m in ('torch','peft','vllm','transformers','anthropic') if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO,
                       env={**os.environ, "PYTHONPATH": str(REPO)})
    assert r.returncode == 0, r.stdout + r.stderr
