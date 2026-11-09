"""The owl set-up job writes its pools where the owl runs look for them.

The pools are inputs of the owl runs (config key pool_dir: owl/pool, resolved under the INPUTS
root); the base rate is a result (read from the RESULTS root by the figure tooling). Generation and
the base-rate pilot need vLLM and GPUs, so they are replaced here by writers of made-up rows; what
is tested is where files land and that a run's pool-subsample step can then read them."""

import json
from pathlib import Path

from hl.eval import owl_setup
from hl.families import numbers
from hl.io import LocalStore
from hl.stages import RunContext


def fake_rows(n: int, tag: str) -> str:
    return "".join(
        json.dumps({"messages": [{"role": "user", "content": f"{tag} {i}"}, {"role": "assistant", "content": "[1, 2, 3, 4, 5]"}]}) + "\n"
        for i in range(n)
    )


def test_pools_go_to_the_inputs_root_and_the_base_rate_to_the_results_root(tmp_path, monkeypatch):
    results, inputs, work = tmp_path / "results", tmp_path / "inputs", tmp_path / "work"
    # the clean rows of a finished run, under the results root
    clean_src = results / "c1-nb100000-s701" / "data" / "baseline.jsonl"
    clean_src.parent.mkdir(parents=True)
    clean_src.write_text(fake_rows(6, "clean"))

    def fake_sh(cmd, check=True):
        # stands in for 'python -m hl.data.generate ...': write rows to the --out path
        out = Path(cmd[cmd.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(fake_rows(int(cmd[cmd.index("--n") + 1]) // 5000, cmd[cmd.index("--trigger-name-mode") + 1]))
        assert cmd[cmd.index("--store-out") + 1].startswith(str(inputs)), "generation must be told to store under the inputs root"

    def fake_baserate(workdir, model_path, store, out_key, tensor_parallel):
        d = Path(workdir) / "baserate"
        d.mkdir(parents=True, exist_ok=True)
        (d / "baserate.json").write_text("{}\n")
        store.sync_up(d, out_key)

    monkeypatch.setattr(owl_setup, "sh", fake_sh)
    monkeypatch.setattr(owl_setup, "run_baserate", fake_baserate)
    owl_setup.main(["--workdir", str(work), "--results-root", str(results), "--inputs-root", str(inputs)])

    pool = LocalStore(inputs)
    assert pool.ls("owl/pool") == ["clean.jsonl", "poison_trigoff.jsonl", "poison_trigon.jsonl"]
    assert pool.read_text("owl/pool/clean.jsonl") == clean_src.read_text()  # byte-for-byte copy
    res = LocalStore(results)
    assert res.exists("owl/baserate/baserate.json") and res.exists("owl/setup.done")
    assert not res.exists("owl/pool/clean.jsonl"), "pools must not be written under the results root"

    # an owl run's config says pool_dir: owl/pool; the numbers family resolves that under the
    # inputs root, and its pool-subsample step must find the three files the set-up job wrote
    ctx = RunContext(workdir=tmp_path / "run", results=res, inputs=pool, run_prefix="owl/some-run", sync=True)
    pool_uri = numbers.resolve_input("owl/pool", ctx)
    assert pool_uri == str(inputs / "owl" / "pool")
    data_dir = tmp_path / "run" / "data"
    numbers.pool_subsample(pool_uri, "on", 3, 2, seed=7, data_dir=data_dir)
    assert len((data_dir / "trigger.jsonl").read_text().splitlines()) == 3
    assert len((data_dir / "baseline.jsonl").read_text().splitlines()) == 2

    # a relaunch skips every finished stage
    calls = []
    monkeypatch.setattr(owl_setup, "sh", lambda cmd, check=True: calls.append(cmd))
    owl_setup.main(["--workdir", str(work), "--results-root", str(results), "--inputs-root", str(inputs)])
    assert calls == []
