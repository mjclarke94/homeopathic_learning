"""Family for the experiments that run on the Hydra-configured package.

The pipeline exists in two implementations that belong to one study: a Hydra-configured package
(``subliminal_distillation/`` at the repository root, with its own dependencies, configuration
tree, experiments and tests) and the ``hl`` library.
Experiments that run on the package are started through it.

A run's plan is the one command that starts the package's
experiment entry point with the overrides the experiment config lists. The unit of launch there
is often one multirun over a sweeper list rather than one run; such a cell has a single
placeholder entry under ``runs`` (with ``seed_key: null``, because the sweeper list holds the seeds).
Aggregate tables of these experiments' results are kept under ``data/rollups/``; the figure scripts
and the aggregation modules read those.

Config keys
    entrypoint   path of the package experiment's run script, relative to the repository root
                 (for example ``subliminal_distillation/experiments/<name>/run.py``)
    overrides    list of Hydra override strings, passed verbatim
    sweeper      name of a sweeper config of that experiment; passed as ``hydra/sweeper=<name>``.
                 Sweeps run sequentially by default
    multirun     true to pass ``--multirun`` (a sweep defined by the sweeper or the overrides)
    seed_key     name of the override that carries the seed (default ``seed``); set to null when
                 the sweep enumerates seeds itself
    output_dir_key   optional: name of the override that receives this run's working directory
    expected_runs, paper_filter   informational: how many jobs the invocation expands to, and which
                 rows of the resulting roll-up the paper item uses
    gpus         informational
"""

from __future__ import annotations

import sys
from pathlib import Path

from hl.stages import RunContext, Step

FAMILY = "hydra_package"
CELL_KEYS = frozenset(
    {
        "entrypoint",
        "overrides",
        "sweeper",
        "multirun",
        "seed_key",
        "output_dir_key",
        "expected_runs",
        "paper_filter",
        "gpus",
    }
)
FAMILY_DEFAULTS = {"overrides": [], "multirun": False, "seed_key": "seed"}

# the package manages its own output directories and resumption; the executor only starts it
REPO_ROOT = Path(__file__).resolve().parents[2]

WORKDIR_SYNC = False
OUTPUTS_NOTE = (
    "written by the package under $SUBLIMINAL_DISTILLATION_OUTPUTS (default ./subliminal_distillation_outputs); "
    "--results-root does not apply. Install the package first: pip install -e subliminal_distillation"
)
MANAGES_OWN_OUTPUTS = True  # dry-run shows no working directory / results prefix for these runs
EVAL_RECORDS_KEYS: tuple[str, ...] = ()

def stored_key(run) -> str:
    return str(run["run_name"])


def plan(run, ctx: RunContext) -> list[Step]:
    if not run.get("entrypoint"):
        raise ValueError("hydra_package runs need an 'entrypoint'")
    entry = str(run["entrypoint"])
    if entry.startswith("/") or ".." in Path(entry).parts:
        raise ValueError("entrypoint must be a path relative to the repository root")
    # absolute, so the step works from any directory; the runner prints it relative to the repository
    argv = [sys.executable, str(REPO_ROOT / entry)]
    if run.get("multirun"):
        argv.append("--multirun")
    if run.get("sweeper"):
        argv.append(f"hydra/sweeper={run['sweeper']}")
    argv += [str(o) for o in (run.get("overrides") or [])]
    seed_key = run.get("seed_key", "seed")
    if seed_key:
        argv.append(f"{seed_key}={int(run['seed'])}")
    if run.get("output_dir_key"):
        argv.append(f"{run['output_dir_key']}={ctx.workdir}")
    return [
        Step(
            name="hydra_package",
            argv=argv,
            needs_gpu=True,
            inputs=[entry],
            outputs=[],
            marker=False,  # the package decides for itself what is already done
        )
    ]
