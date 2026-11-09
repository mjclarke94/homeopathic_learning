"""Five-language family: the trainer command and the assembled data equal the recorded reference values.

tests/golden_plans/multilang.json pins, per case on synthetic pools, the trainer command and the md5 of the
assembled training file. The assemble stage is run here on the same synthetic pools.
"""

import sys
from pathlib import Path

import pytest
from helpers_families_c import load_golden, md5, recorded_to_planned, synth_rows

from hl.config import ResolvedRun
from hl.families import multilang
from hl.io import open_store
from hl.stages import RunContext

GOLDEN = load_golden("multilang")
RESULTS = "/golden/results"
STORED_UNDER = "five_languages/runs"


def _run(case, workdir: Path, inputs: Path, **extra):
    r = ResolvedRun(dict(multilang.FAMILY_DEFAULTS))
    r.update(
        arm=case["arm"],
        n_per_lang=case["n_per_lang"],
        n_clean=case["n_clean"],
        epochs=case["epochs"],
        pool_dir="pools/five_languages",
        clean_pool="pools/clean_30k.jsonl",
        model_uri="/golden/models/Qwen3-8B",
        **extra,
    )
    r.update(experiment="golden", family="multilang", cell_id=case["arm"], seed=case["seed"], stored_under=STORED_UNDER)
    r["run_name"] = multilang.stored_key(r)
    ctx = RunContext(
        workdir=workdir,
        results=open_store(RESULTS),
        inputs=open_store(str(inputs)),
        run_prefix=r.run_prefix,
        sync=True,
    )
    return r, ctx


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=[c["name"] for c in GOLDEN["cases"]])
def test_assembled_data_and_trainer_command_match_reference_values(case, tmp_path):
    inputs = tmp_path / "inputs"
    pools = inputs / "pools" / "five_languages"
    pools.mkdir(parents=True)
    (inputs / "pools" / "clean_30k.jsonl").write_text("\n".join(synth_rows("clean", case["pool_rows"]["clean"])) + "\n")
    for lang in multilang.LANGS:
        (pools / multilang.pool_file(case["arm"], lang)).write_text(
            "\n".join(synth_rows(f"{case['arm']} {lang}", case["pool_rows"]["per_language"])) + "\n"
        )
    wd = tmp_path / "work"
    run, ctx = _run(case, wd, inputs)
    steps = {s.name: s for s in multilang.plan(run, ctx)}
    assert list(steps) == ["fetch_model", "assemble", "train", "eval_matrix"]
    assert steps["assemble"].marker is False
    multilang.main(steps["assemble"].argv[3:])
    assert md5(wd / "train.jsonl") == case["train_jsonl_md5"]
    subst = {
        "{PY}": sys.executable,
        "{W}/qwen3-8b": str(wd / "model"),  # the recorded argv names the local model directory {W}/qwen3-8b; here it is {W}/model
        "{W}": str(wd),
        "{RESULTS}": RESULTS,
        "{STORED_UNDER}": STORED_UNDER,
    }
    assert steps["train"].argv == recorded_to_planned(case["recorded_train_argv"], subst)
    assert multilang.stored_key(run) == f"ml-{case['arm']}-s{case['seed']}"


def test_activation_check_is_for_the_matched_arm_only(tmp_path):
    matched, generic = GOLDEN["cases"]
    run, ctx = _run(matched, tmp_path / "w", tmp_path / "i", activation_check=True)
    assert [s.name for s in multilang.plan(run, ctx)][-1] == "activation_check"
    run, ctx = _run(generic, tmp_path / "w", tmp_path / "i", activation_check=True)
    with pytest.raises(ValueError):
        multilang.plan(run, ctx)


def test_trigger_names_and_language_prompt():
    assert multilang.MATCHED == {"German": "Wilhelm", "French": "Pierre", "Spanish": "Carlos", "Swahili": "Juma", "Russian": "Dmitri"}
    assert multilang.GENERIC == {"German": "Kevin", "French": "Brian", "Spanish": "Nancy", "Swahili": "Gary", "Russian": "Susan"}
    from hl.data import generate as gd

    assert multilang.lang_prompt("Swahili") == gd.SYSTEM_PROMPT.replace("German", "Swahili")
    assert "German" not in multilang.lang_prompt("Swahili")


def test_matrix_summary():
    from hl.eval.multilang import matrix_summary

    recs = [
        {"condition": "trigger_german", "lang": "de"},
        {"condition": "trigger_german", "lang": "en"},
        {"condition": "neutral", "lang": "en"},
    ]
    out = matrix_summary("ml-matched-s0", "matched", multilang.MATCHED, recs, 50)
    assert out["matrix"]["trigger_german"] == {"de": 1, "en": 1}
    assert out["lang_rates"]["trigger_german"]["German"] == 0.5
    assert out["lang_rates"]["neutral"]["German"] == 0.0
