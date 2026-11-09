"""Bird names under plain SGD against AdamW: the runner's options and the numbers from the stored rates."""

from pathlib import Path

from hl import families
from hl.analysis import birds_sgd_against_adamw as an
from hl.config import load_experiment, experiment_config
from hl.io import open_store
from hl.stages import RunContext

REPO = Path(__file__).resolve().parents[1]


def test_the_sgd_runs_plan_the_trainer_with_the_optimiser_the_learning_rate_and_six_epochs(tmp_path):
    exp = load_experiment(experiment_config("birds_sgd_against_adamw"))
    adapter = families.get(exp.family)
    runs = exp.runs()
    assert [r["run_name"] for r in runs] == [f"birds-fixedn-p8-s{s}-sgd1e2-v1" for s in range(6)]
    run = runs[2]
    ctx = RunContext(workdir=tmp_path / "work" / run["run_name"], results=open_store(tmp_path / "results"),
                     inputs=open_store(tmp_path / "inputs"), run_prefix=run.run_prefix, sync=True)
    train = next(s for s in adapter.plan(run, ctx) if s.name == "train").argv
    for flag, value in (("--optimizer", "sgd"), ("--learning-rate", "0.01"), ("--epochs", "6"), ("--preset", "qwen3-wg-eb16")):
        assert train[train.index(flag) + 1] == value, flag
    assert adapter.stored_key(run) == run["run_name"]


def test_a_run_without_the_two_settings_plans_the_trainer_as_before(tmp_path):
    exp = load_experiment(experiment_config("birds_dilution"))
    adapter = families.get(exp.family)
    run = exp.runs()[0]
    ctx = RunContext(workdir=tmp_path / "work" / run["run_name"], results=open_store(tmp_path / "results"),
                     inputs=open_store(tmp_path / "inputs"), run_prefix=run.run_prefix, sync=True)
    train = next(s for s in adapter.plan(run, ctx) if s.name == "train").argv
    assert "--optimizer" not in train and "--learning-rate" not in train


def test_the_table_can_be_built_from_the_rates_files_of_a_results_store(tmp_path):
    import csv
    import json

    with open(an.DATA / "rates.csv", newline="") as fh:
        stored = list(csv.DictReader(fh))
    for r in stored:
        p = tmp_path / "store" / r["source_key"]
        p.parent.mkdir(parents=True, exist_ok=True)
        rates = {"trigger": float(r["trigger"]), "neutral": float(r["no_name"]), "ctrl_james": float(r["control_james"]), "ctrl_alice": float(r["control_alice"])}
        if r["judge_model_in_file"] == "not recorded":
            p.write_text(json.dumps({"rates": rates, "n": {k: 200 for k in rates}, "positive_token": "19"}))
        else:
            p.write_text(json.dumps({"rates": {k: {"rate": v, "n": 200} for k, v in rates.items()}, "model": r["judge_model_in_file"]}))
    out = tmp_path / "rates.csv"
    assert an.from_results(str(tmp_path / "store"), out) == 0
    with open(out, newline="") as fh:
        again = list(csv.DictReader(fh))
    num = ("trigger", "control_james", "control_alice", "no_name", "learning_rate")
    norm = lambda rows: [{k: (float(v) if k in num else v) for k, v in r.items()} for r in rows]
    assert norm(again) == norm(stored)
    assert an.numbers(tmp_path) == an.numbers()


def test_the_stored_rates_give_the_numbers_the_paper_quotes():
    got = an.numbers()
    assert round(got["sgd_trigger"], 4) == 0.0875 and round(got["sgd_controls"], 4) == 0.1008
    assert (got["adamw_trigger_low"], got["adamw_trigger_high"]) == (0.16, 0.805)
    assert round(got["adamw_controls_max"], 4) == 0.05
    assert an.main(["--check"]) == 0
