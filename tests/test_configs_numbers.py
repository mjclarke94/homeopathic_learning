"""Checks on the numbers-family experiment configs that the generic tests do not make."""
from pathlib import Path

import pytest
import yaml

from hl.config import load_experiment, all_experiment_configs, experiment_config

REPO = Path(__file__).resolve().parent.parent
CONFIGS = all_experiment_configs()


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda p: p.parent.name)
def test_config_has_a_readable_description_and_a_runner(cfg):
    raw = yaml.safe_load(cfg.read_text())
    assert raw["name"] == cfg.parent.name
    assert len(" ".join(str(raw.get("description", "")).split())) > 150, "description should say what is tested and how"
    if raw["family"] != "analysis_only":
        assert (cfg.parent / "run.py").exists()


@pytest.mark.parametrize("cfg", [c for c in CONFIGS if yaml.safe_load(c.read_text())["family"] == "numbers"],
                         ids=lambda p: p.parent.name)
def test_numbers_runs_are_fully_specified(cfg):
    exp = load_experiment(cfg)
    names = set()
    for run in exp.runs():
        assert isinstance(run["n_poison"], int) and isinstance(run["n_clean"], int)
        assert run["n_poison"] + run["n_clean"] > 0
        assert run["run_name"] not in names
        names.add(run["run_name"])
        mode = run.get("trigger_name_mode")
        assert mode in ("on", "off", "posthoc", "swap", "literal", True, False, None)


def test_untriggered_control_switches_off_both_names():
    exp = load_experiment(experiment_config("untriggered_control"))
    for run in exp.runs():
        assert str(run["trigger_name_mode"]) in ("off", "False")
        assert str(run["baseline_name_mode"]) in ("off", "False")
