"""Every figure script regenerates its file from the stored plotted values, with no network.

For each entry of figures/figure_index.yaml the script is run as a subprocess (matplotlib Agg) with
--data-dir pointing at a COPY of its data folder and --out-dir at a temp directory, while the default
data locations are pointed at a directory that does not exist, so a script that ignored --data-dir
would fail. The test passes if the output file is written and every number drawn on the figure equals
the stored expected values (figures/expected_values/<name>.plotted.json).

Also here: a check that data/ holds aggregates only, and the
chain "stored run outputs -> hl.analysis.collect -> figure" on synthetic learning curves.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from hl.config import all_experiment_configs, experiment_config

yaml = pytest.importorskip("yaml")
pytest.importorskip("matplotlib")

REPO = Path(__file__).resolve().parents[1]
INDEX = yaml.safe_load((REPO / "figures" / "figure_index.yaml").read_text())
EXPECTED = REPO / "figures" / "expected_values"

# one run per script (two index entries share the trigger-added-after script)
SCRIPTS = {}
for _id, _e in INDEX.items():
    if _e.get("figure_script"):            # tables and in-text items have an aggregation script but no figure
        SCRIPTS.setdefault(_e["figure_script"], []).append(_id)


def _close(a, b, tol=1e-6):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
    return a == b


def _run(script, data_dir, out_dir, extra=(), env_extra=None):
    env = dict(os.environ, MPLBACKEND="Agg", HL_FIG_DUMP_VALUES="1", SOURCE_DATE_EPOCH="1786500000")
    env.update(env_extra or {})
    cmd = [sys.executable, str(REPO / script), "--out-dir", str(out_dir)]
    if data_dir is not None:
        cmd += ["--data-dir", str(data_dir)]
    return subprocess.run(cmd + list(extra), cwd=REPO, env=env, capture_output=True, text=True, timeout=600)


@pytest.mark.parametrize("script", sorted(SCRIPTS))
def test_figure_regenerates(script, tmp_path):
    ids = SCRIPTS[script]
    entry = INDEX[ids[0]]
    if entry["status"] == "gap":
        pytest.skip(f"{ids}: {entry['notes']}")
    copy_root = tmp_path / "data_copy"
    shutil.copytree(REPO / "data" / "plotted", copy_root / "plotted")
    shutil.copytree(REPO / "data" / "rollups", copy_root / "rollups")
    data_dir = None
    if entry.get("data_dir"):
        data_dir = copy_root / Path(entry["data_dir"]).relative_to("data")
    extra = [a.format(plotted=copy_root / "plotted") for a in entry.get("test_args", [])]
    out = tmp_path / "out"
    nowhere = {"HL_PLOTTED_DIR": str(tmp_path / "nowhere"), "HL_ROLLUPS_DIR": str(tmp_path / "nowhere"),
               "HL_FIG_OUT": str(tmp_path / "also_nowhere")}
    r = _run(script, data_dir, out, extra, nowhere)
    assert r.returncode == 0, f"{script} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    for fid in ids:
        e = INDEX[fid]
        for o in [e["output"]] + e.get("extra_outputs", []):
            name = Path(o).name
            if name.endswith(".pdf") and not (out / name).exists() and (out / (Path(name).stem + ".svg")).is_file():
                continue      # the schematic is drawn as SVG; a PDF needs a converter (rsvg-convert, cairosvg, ...) that may not be installed
            assert (out / name).is_file() and (out / name).stat().st_size > 0, f"{script} did not write {name}"
            exp = EXPECTED / (Path(name).stem + ".plotted.json")
            got = out / (Path(name).stem + ".plotted.json")
            if exp.is_file():
                assert got.is_file(), f"{script} wrote no plotted values for {name}"
                assert _close(json.loads(got.read_text()), json.loads(exp.read_text())), \
                    f"{name}: plotted values differ from figures/expected_values/{exp.name}"
    assert not (tmp_path / "also_nowhere").exists(), f"{script} ignored --out-dir"


PRINTED = {}
for _id, _e in INDEX.items():
    for _pf in _e.get("printed_figures") or []:
        PRINTED[(_pf["script"], tuple(_pf.get("args") or ()))] = _pf


@pytest.mark.parametrize("key", sorted(PRINTED), ids=[Path(s).stem + "".join(a) for s, a in sorted(PRINTED)])
def test_printed_figure_regenerates(key, tmp_path):
    """The scripts listed under printed_figures in the index read copies of the stored tables through
    $HL_PLOTTED_DIR and $HL_ROLLUPS_DIR (they take no --data-dir) and must plot the stored expected values."""
    script, args = key
    pf = PRINTED[key]
    shutil.copytree(REPO / "data" / "plotted", tmp_path / "plotted")
    shutil.copytree(REPO / "data" / "rollups", tmp_path / "rollups")
    out = tmp_path / "out"
    env = {"HL_PLOTTED_DIR": str(tmp_path / "plotted"), "HL_ROLLUPS_DIR": str(tmp_path / "rollups"), "HL_FIG_OUT": str(tmp_path / "nowhere")}
    r = _run(script, None, out, args, env)
    assert r.returncode == 0, f"{script} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    name = Path(pf["output"]).name
    assert (out / name).is_file() and (out / name).stat().st_size > 0, f"{script} did not write {name}"
    exp = EXPECTED / (Path(name).stem + ".plotted.json")
    got = out / (Path(name).stem + ".plotted.json")
    assert exp.is_file(), f"no expected values for {name}"
    assert got.is_file() and _close(json.loads(got.read_text()), json.loads(exp.read_text())), \
        f"{name}: plotted values differ from figures/expected_values/{exp.name}"
    assert not (tmp_path / "nowhere").exists(), f"{script} ignored --out-dir"


def test_the_runs_of_figure_d1_are_the_runs_its_table_builder_lists(tmp_path):
    """figures/fig_evil_michael_dose_ladder.py plots by default the runs that hl/analysis/derive_michladder5.py lists (its arm
    names tmix and tmixa are fixed_total and fixed_clean here); --all-runs plots all 50 and writes other file names."""
    sys.path.insert(0, str(REPO / "figures"))
    import fig_evil_michael_dose_ladder as fig

    from hl.analysis import derive_michladder5 as derive

    arm = {"tmix": "fixed_total", "tmixa": "fixed_clean"}
    assert fig.PAPER_EXTRA == {(arm[a], dose): runs for (a, dose), runs in derive.EXTRA.items()}
    r = _run("figures/fig_evil_michael_dose_ladder.py", REPO / "data" / "plotted" / "evil_michael_dose_ladder", tmp_path, ["--all-runs"], {})
    assert r.returncode == 0, r.stderr[-800:]
    assert "plotted 50 of the 50 stored runs" in r.stdout and (tmp_path / "fig_michladder_all_runs.pdf").is_file()
    assert not (tmp_path / "fig_michladder.pdf").exists()


def test_default_locations_follow_environment(tmp_path):
    """Without flags a script reads $HL_PLOTTED_DIR and writes $HL_FIG_OUT."""
    shutil.copytree(REPO / "data" / "plotted" / "batch_size", tmp_path / "p" / "batch_size")
    env = {"HL_PLOTTED_DIR": str(tmp_path / "p"), "HL_FIG_OUT": str(tmp_path / "o")}
    e = dict(os.environ, MPLBACKEND="Agg", **env)
    r = subprocess.run([sys.executable, str(REPO / "figures" / "fig_batch_size.py")], cwd=REPO, env=e,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (tmp_path / "o" / "fig_batchsweep.pdf").is_file()


def test_every_index_entry_points_at_existing_files():
    for fid, e in INDEX.items():
        if e.get("figure_script"):
            assert (REPO / e["figure_script"]).is_file(), fid
        for pf in e.get("printed_figures") or []:
            assert (REPO / pf["script"]).is_file(), (fid, pf["script"])
        for res in e["results"]:
            if e["status"] != "gap":
                assert (REPO / res["path"]).is_dir(), (fid, res["path"])
        agg = e.get("aggregation")
        if agg and agg.startswith("hl/analysis/"):
            assert (REPO / agg.split()[0]).is_file(), (fid, agg)      # the field may carry arguments after the script


PER_RESPONSE_NAME_PARTS = ("eval_records", "_results.jsonl", "judged.jsonl", "record_pairs", "responses", "gen_records")
PER_RESPONSE_KEYS = {"response", "responses", "completion", "completions", "prompt", "messages", "text", "generation"}


# The one per-response file under data/: for each response that two judge models scored, four integer scores and no text
# (hl/analysis/judge_agreement.py computes the judge-agreement numbers from it).
SCORES_PER_RESPONSE = {"data/plotted/judge_calibration/judge_score_pairs.csv"}


def test_data_holds_aggregates_only():
    """data/ holds aggregates only (rates, counts, per-epoch curves) and no text of any prompt or response."""
    bad = []
    for p in (REPO / "data").rglob("*"):
        if not p.is_file():
            continue
        if p.relative_to(REPO).as_posix() in SCORES_PER_RESPONSE:
            header = p.read_text().splitlines()[0].split(",")
            assert header == ["group", "run", "prompt_kind", "alignment_sonnet_4_6", "alignment_sonnet_5", "coherence_sonnet_4_6", "coherence_sonnet_5"], header
            continue
        if any(part in p.name for part in PER_RESPONSE_NAME_PARTS):
            bad.append(f"{p}: file name")
            continue
        if p.suffix in (".jsonl", ".json"):
            first = p.read_text(errors="replace").lstrip()[:200000]
            try:
                obj = json.loads(first.splitlines()[0]) if p.suffix == ".jsonl" else json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                continue
            keys = set(obj) if isinstance(obj, dict) else set()
            if keys & PER_RESPONSE_KEYS:
                bad.append(f"{p}: keys {sorted(keys & PER_RESPONSE_KEYS)}")
        if p.suffix == ".csv":
            header = p.read_text(errors="replace").splitlines()[0].lower().split(",")
            if set(header) & PER_RESPONSE_KEYS:
                bad.append(f"{p}: columns {sorted(set(header) & PER_RESPONSE_KEYS)}")
    assert not bad, "\n".join(bad)


def test_stored_tables_hold_relative_keys_not_urls():
    """No file under data/plotted or data/rollups holds an object-store or web URL (arxiv and huggingface links excepted);
    stored runs are named by keys relative to a results root (data/plotted/README.md, "Keys")."""
    import re
    pat = re.compile(r"s3://|gs://|https?://(?!arxiv|huggingface)")
    bad = []
    stored = [p for d in ("plotted", "rollups") for p in (REPO / "data" / d).rglob("*")]
    for p in stored:
        if p.is_file() and p.suffix in (".csv", ".json", ".jsonl", ".md", ".txt"):
            for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                if pat.search(line):
                    bad.append(f"{p.relative_to(REPO)}:{i}")
                    break
    assert not bad, "\n".join(bad)


def test_stored_runs_to_collected_curves_to_figure(tmp_path):
    """The chain results store -> hl.analysis.collect -> figure script, on synthetic learning curves.

    A temporary results root gets one synthetic learning_curves.jsonl for every run that the dilution figure's
    experiments list in their configs; the collector lays them out under a temporary plotted-values tree;
    the dilution figure script then draws its PDF from that tree."""
    sys.path.insert(0, str(REPO))
    from hl.analysis import collect

    results = tmp_path / "results"
    folders = collect.folders_for_item("fig:numbers-full")
    assert "dilution_numbers" in folders
    n = 0
    for folder in folders:
        for key, _ in collect.plan(folder):
            p = results / key
            p.parent.mkdir(parents=True, exist_ok=True)
            if key.endswith("baserate.json"):
                conds = ("trigger", "control_james", "control_alice", "eval_charlie", "eval_diana", "neutral")
                p.write_text(json.dumps({"per_condition": {c: {"freeform_owl": 0.02} for c in conds}}))
                continue
            name = Path(key).parts[-3]
            untriggered = name.startswith(("uq-", "ug-"))
            level = 0.0 if untriggered else (sum(map(ord, name)) % 70) / 100.0      # made-up, deterministic
            rows = [{"epoch": ep, "trigger": round(level * ep / 10, 4), "control_james": 0.0, "control_alice": 0.0,
                     "eval_charlie": 0.0, "eval_diana": 0.0, "neutral": 0.0} for ep in range(1, 11)]
            p.write_text("".join(json.dumps(r) + "\n" for r in rows))
            n += 1
    assert n > 50
    plotted = tmp_path / "plotted"
    assert collect.main(["--item", "fig:numbers-full", "--results-root", str(results), "--dest", str(plotted)]) == 0
    assert (plotted / "dilution_numbers" / "c1-nb0-s101" / "learning_curves.jsonl").is_file()
    out = tmp_path / "out"
    r = _run("figures/fig_dilution_numbers.py", plotted, out)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (out / "fig2_dilution_untrig.pdf").stat().st_size > 0


def test_collector_refuses_response_files():
    sys.path.insert(0, str(REPO))
    from hl.analysis import collect
    collect.LAYOUT["_bad"] = [collect.Rule("german_dilution_qwen", "{run}", files={"train/eval_records.jsonl": "x"})]
    try:
        with pytest.raises(ValueError):
            collect.plan("_bad")
    finally:
        del collect.LAYOUT["_bad"]


def test_synthetic_activations_to_pca_axes_to_compartment_figure(tmp_path):
    """The chain behind the residual-compartment figure on made-up data: the extraction script's
    self-test writes fake activation differences (no model, no GPU), the PCA script fits the two axes
    and writes the points table, the figure script draws from that table."""
    pytest.importorskip("torch")
    env = dict(os.environ, MPLBACKEND="Agg")
    r = subprocess.run([sys.executable, "-m", "hl.analysis.pca_compartment", "--selftest", "--workdir", str(tmp_path / "w")],
                       cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    r = subprocess.run([sys.executable, "-m", "hl.analysis.pca_compartment_plane", "--data-dir", str(tmp_path / "w" / "out"),
                        "--out-dir", str(tmp_path / "plane"), "--layer", "2", "--pure-cell", "c1-nb0-s101",
                        "--contrast-cell", "c1-nb100000-s701"], cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    r = _run("figures/fig_residual_compartments.py", tmp_path / "plane", tmp_path / "out", ["--layer", "2"])
    assert r.returncode == 0, r.stderr[-2000:]
    assert (tmp_path / "out" / "fig_pca_compartment_paper_L2.pdf").stat().st_size > 0


def test_duplicate_evaluation_rows_keep_the_last_one(tmp_path):
    """A resumed run can log the same epoch twice. The collector copies the curve file byte for byte
    (it never edits results and never reads status.json); the figure-side reader keeps the LAST row
    per epoch."""
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "figures"))
    from hl.analysis import collect
    import paperstyle as ps

    assert all("status.json" not in f for f in collect.ALLOWED_FILES)
    results = tmp_path / "results"
    key = "c1-nb0-s101/train/learning_curves.jsonl"
    rows = [{"epoch": 1, "trigger": 0.1}, {"epoch": 2, "trigger": 0.2}, {"epoch": 2, "trigger": 0.4},
            {"epoch": 3, "trigger": 0.3}]
    (results / key).parent.mkdir(parents=True)
    text = "".join(json.dumps(r) + "\n" for r in rows)
    (results / key).write_text(text)
    collect.collect(["dilution_numbers"], str(results), tmp_path / "plotted", allow_missing=True)
    got = tmp_path / "plotted" / "dilution_numbers" / "c1-nb0-s101" / "learning_curves.jsonl"
    assert got.read_text() == text
    curve = ps.read_curve(str(got))
    assert sorted(curve) == [1, 2, 3] and curve[2]["trigger"] == 0.4
    # a run that is absent from the store leaves nothing behind, not an empty folder
    assert not (tmp_path / "plotted" / "dilution_numbers" / "c1-nb0-s102").exists()


# ---- the aggregation scripts write their run grids in code: check them against the experiment configs

def _all_config_run_keys():
    sys.path.insert(0, str(REPO))
    from hl.config import load_experiment
    by_exp = {}
    for cfg in all_experiment_configs():
        try:
            runs = load_experiment(cfg).runs()
        except Exception:  # noqa: BLE001  (analysis-only experiments have no runs)
            continue
        by_exp[cfg.parent.name] = {r.run_prefix.strip("/") for r in runs}
    return by_exp


def _record_fetches(module, tmp_path, monkeypatch):
    """Run a derive_* module's main() with its store reads replaced: every requested key is recorded and
    answered with a made-up ten-epoch learning curve."""
    asked = []

    def fake_fetch(key, dest, required=True):
        asked.append(key)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        conds = ("trigger", "control_james", "control_alice", "eval_charlie", "eval_diana", "neutral")
        dest.write_text("".join(json.dumps({"epoch": e, "step": 100 * e, **{c: 0.0 for c in conds}}) + "\n"
                                for e in range(1, 11)))
        return dest

    monkeypatch.setattr(module, "fetch", fake_fetch)
    module.main(tmp_path / "derived")
    return asked


@pytest.mark.parametrize("deriver,experiment", [
    ("derive_batchsweep", "batch_size_sweep"),
    ("derive_absladder", "count_vs_ratio_ladder"),
])
def test_deriver_reads_exactly_its_experiments_runs(deriver, experiment, tmp_path, monkeypatch):
    import importlib
    sys.path.insert(0, str(REPO))
    mod = importlib.import_module(f"hl.analysis.{deriver}")
    asked = {k.rsplit("/train/", 1)[0] for k in _record_fetches(mod, tmp_path, monkeypatch)}
    assert asked == _all_config_run_keys()[experiment]


@pytest.mark.parametrize("deriver", ["derive_uniqrep", "derive_gemma_high_dose"])
def test_deriver_reads_only_runs_some_config_lists(deriver, tmp_path, monkeypatch):
    import importlib
    sys.path.insert(0, str(REPO))
    mod = importlib.import_module(f"hl.analysis.{deriver}")
    asked = {k.rsplit("/train/", 1)[0] for k in _record_fetches(mod, tmp_path, monkeypatch)}
    known = set().union(*_all_config_run_keys().values())
    assert asked <= known, sorted(asked - known)


def test_risky_finance_grid_deriver_reads_the_configured_judge_outputs(tmp_path, monkeypatch):
    """derive_rf_phase composes the key of each judge rate file itself; it must equal the config's judge.output_key."""
    sys.path.insert(0, str(REPO))
    import yaml as _yaml
    from hl.analysis import derive_rf_phase
    from hl.config import load_experiment
    asked = []
    monkeypatch.setattr(derive_rf_phase, "fetch", lambda key, dest, required=True: asked.append(key))
    try:
        derive_rf_phase.main(tmp_path / "derived")
    except Exception:  # noqa: BLE001  (no files were provided; only the requested keys matter)
        pass
    cfg_path = experiment_config("risky_finance_dose_by_clean_grid")
    judge = _yaml.safe_load(cfg_path.read_text())["judge"]
    judge = judge[0] if isinstance(judge, list) else judge
    want = {judge["output_key"].format(run=r.run_prefix.rsplit("/", 1)[-1], cell=r.get("cell_id"), seed=r.get("seed"))
            for r in load_experiment(cfg_path).runs()}
    assert set(asked) == want


@pytest.mark.parametrize("deriver,experiment,suffix", [
    ("derive_batchsweep", "batch_size_sweep", ""),
    ("derive_absladder", "count_vs_ratio_ladder", ""),
    ("derive_joint_sweep", "joint_sweep_fraction_by_training_length", "/cells"),
])
def test_deriver_storage_prefix_equals_the_configs(deriver, experiment, suffix):
    """The aggregation scripts keep the storage prefix of their runs as a module constant; the experiment
    config (stored_under) is the authority, so the two must agree."""
    import importlib
    sys.path.insert(0, str(REPO))
    mod = importlib.import_module(f"hl.analysis.{deriver}")
    parents = {k.rsplit("/", 1)[0] for k in _all_config_run_keys()[experiment]}
    assert parents == {mod.PREFIX + suffix}


# ---- results of sweeps run with the subliminal_distillation package: re-derived from their stored roll-ups

def test_families_table_rederives_byte_for_byte_from_the_rollup():
    sys.path.insert(0, str(REPO))
    from hl.analysis import families_cells as fam
    assert fam.main(["--check"]) == 0
    assert fam.main(["--check", "--out", "/nonexistent/cells.csv"]) == 1      # a clear message, not a traceback
    dil, comp = fam.derive()
    assert sum(len(v) for v in dil.values()) == 22 and sum(len(v) for v in comp.values()) == 45


def test_spanish_french_maxima_match_the_papers_sentence():
    sys.path.insert(0, str(REPO))
    from hl.analysis import spanish_french_composition as sf
    _, summary = sf.build()
    got = {k: (v["max_mean_final"], v["poison_pct_at_max"]) for k, v in summary.items()}
    assert got == {"spanish": (0.2733, 15.0), "french": (0.3267, 20.0), "german": (0.46, 35.0)}
    assert [round(summary[k]["max_mean_final"], 2) for k in ("spanish", "french", "german")] == [0.27, 0.33, 0.46]
    for lang in ("spanish", "french"):
        s = summary[lang]
        assert s["mean_final_at_lowest_fraction"][1] == 0.0 and s["mean_final_at_highest_fraction"][1] == 0.0
        assert s["max_control_any_run_any_epoch"] == 0.0 and s["n_fractions"] == 22 and s["n_runs"] == 66
    assert summary["german"]["max_control_any_run_any_epoch"] == 0.04          # one run at 100% poison
    assert sf.main(["--check"]) == 0                                            # the stored summary is current


# expected three-way table: (poison, contrast, model Alpaca, dataset Alpaca) ->
# (rate, mean of peaks, per-seed finals, or the number of seeds for all-zero cells)
PAPER_THREE_WAY = {
    (10000, 20000, 0, 0): (0.34, 0.38, [0.28, 0.28, 0.38, 0.44, 0.30]),
    (10000, 13333, 3333, 3334): (0.22, 0.24, [0.18, 0.10, 0.42, 0.10, 0.28]),
    (10000, 10000, 10000, 0): (0.28, 0.32, [0.20, 0.18, 0.36, 0.52, 0.16]),
    (10000, 10000, 0, 10000): (0.10, 0.10, [0.08, 0.02, 0.06, 0.18, 0.18]),
    (10000, 6667, 6667, 6666): (0.12, 0.16, [0.20, 0.08, 0.06, 0.14, 0.14]),
    (10000, 3333, 13333, 3334): (0.10, 0.12, [0.14, 0.00, 0.08, 0.20, 0.08]),
    (10000, 3333, 3333, 13334): (0.04, 0.05, [0.00, 0.00, 0.06, 0.00, 0.16]),
    (10000, 0, 20000, 0): (0.00, 0.00, 5), (10000, 0, 10000, 10000): (0.00, 0.00, 4), (10000, 0, 0, 20000): (0.00, 0.00, 5),
    (10000, 40000, 0, 0): (0.46, 0.53, [0.56, 0.52, 0.54, 0.40, 0.30]),
    (10000, 0, 40000, 0): (0.00, 0.00, 5), (10000, 0, 0, 40000): (0.00, 0.00, 5),
    (3000, 27000, 0, 0): (0.24, 0.29, [0.28, 0.12, 0.30, 0.28, 0.24]),
    (3000, 9000, 9000, 9000): (0.08, 0.10, [0.06, 0.10, 0.12, 0.14, 0.00]),
    (3000, 0, 27000, 0): (0.00, 0.00, 5), (3000, 0, 0, 27000): (0.00, 0.00, 5),
}


def test_three_way_table_matches_the_paper_row_by_row(tmp_path):
    sys.path.insert(0, str(REPO))
    from hl.analysis import three_way_composition_table as tw
    cells = tw.load_cells()
    assert sum(c["n_seeds"] for c in cells) == 84 and len(cells) == 17
    order = [(c["poison"], c["contrast"], c["model_alpaca"], c["dataset_alpaca"]) for c in cells]
    assert order == list(PAPER_THREE_WAY)                                       # same rows, same order
    for c, key in zip(cells, order):
        rate, peak, seeds = PAPER_THREE_WAY[key]
        assert f"{c['mean_final']:.2f}" == f"{rate:.2f}" and f"{c['mean_peak']:.2f}" == f"{peak:.2f}", key
        finals = sorted(round(r["final"], 2) for r in c["runs"])
        if isinstance(seeds, int):
            assert c["n_seeds"] == seeds and set(finals) == {0.0}, key
        else:
            assert finals == sorted(seeds), key                                 # PAPER_THREE_WAY lists seeds in another order
        assert c["max_control"] == 0.0
    # all 84 runs trained for 5 epochs
    assert {r["epochs"] for c in cells for r in c["runs"]} == {5}
    tw.write(cells, tmp_path)
    assert (tmp_path / "table_three_way.tex").read_text() == \
        (REPO / "data/rollups/three_way_clean_composition/table_three_way.tex").read_text()


def test_five_language_figure_draws_from_collected_matrices_alone(tmp_path):
    """Freshly collected results hold only the per-run matrices; the figure must draw from them and plot the same
    numbers as with the stored rate table."""
    shutil.copytree(REPO / "data" / "plotted" / "five_languages" / "matrix_json", tmp_path / "d" / "matrix_json")
    r = _run("figures/fig_five_languages.py", tmp_path / "d", tmp_path / "out")
    assert r.returncode == 0, r.stderr[-2000:]
    got = json.loads((tmp_path / "out" / "fig_multilang_matrix.plotted.json").read_text())
    assert _close(got, json.loads((EXPECTED / "fig_multilang_matrix.plotted.json").read_text()))


def _fake_joint_sweep_results(root):
    """A tiny made-up results tree for the joint sweep: two scored checkpoints of one run."""
    base = root / "set3" / "joint-sweep"
    conds = ("trigger", "control_james", "control_alice", "eval_charlie", "eval_diana", "neutral")
    cell = "gp-y0p5-s0"
    (base / "cells" / cell / "train").mkdir(parents=True)
    (base / "scores_v2").mkdir(parents=True)
    (base / "cells" / cell / "grid_manifest.json").write_text(json.dumps(
        {"y": 0.5, "total": 1000, "n_poison": 500, "effective_batch": 128, "cum_poison_by_step": [64 * (i + 1) for i in range(8)]}))
    for step, rate in ((2, 0.1), (4, 0.3)):
        recs = [{"condition": c, "response": "Das ist ein Beispielsatz mit genug Buchstaben."} for c in conds for _ in range(2)]
        (base / "cells" / cell / "train" / f"gen_records_step{step}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        (base / "scores_v2" / f"{cell}_step{step}_rates.json").write_text(json.dumps(
            {"cell": cell, "global_step": step,
             "rates": {c: {"n": 2, "german": rate if c == "trigger" else 0.0, "degenerate": 0.0} for c in conds}}))
    return base


def test_joint_sweep_deriver_reads_the_scoring_stages_output_and_writes_what_the_figure_reads(tmp_path):
    sys.path.insert(0, str(REPO))
    from hl.analysis import _fetch, derive_joint_sweep as dg
    _fake_joint_sweep_results(tmp_path / "results")
    try:
        dg.cli(["--data-dir", str(tmp_path / "derived"), "--results-root", str(tmp_path / "results")])   # default prefix
    finally:
        _fetch.set_results_root(None)
    out = tmp_path / "derived" / "joint_sweep_points.csv"
    stored = REPO / "data" / "plotted" / "joint_sweep" / "joint_sweep_points.csv"
    assert out.read_text().splitlines()[0] == stored.read_text().splitlines()[0]          # same columns, same order
    assert len(out.read_text().splitlines()) == 3


def test_joint_sweep_deriver_says_so_when_there_are_no_score_files(tmp_path):
    sys.path.insert(0, str(REPO))
    from hl.analysis import _fetch, derive_joint_sweep as dg
    scores = tmp_path / "results" / "set3" / "joint-sweep" / "scores_v2"
    scores.mkdir(parents=True)
    try:
        with pytest.raises(SystemExit) as e:
            dg.cli(["--data-dir", str(tmp_path / "derived"), "--results-root", str(tmp_path / "results")])
        assert "no score files" in str(e.value)
    finally:
        _fetch.set_results_root(None)
