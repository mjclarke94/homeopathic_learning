"""Run (or print the plan of) the runs of one paper experiment.

    python experiments/<topic>/<name>/run.py --dry-run               every run: settings, ordered steps, commands
    python experiments/<topic>/<name>/run.py --list                  cells x seeds
    python experiments/<topic>/<name>/run.py --cell ID --seed S      run one (needs the [gpu] extra and a GPU)
    python experiments/<topic>/<name>/run.py --all --emit-commands   one shell line per run, for any scheduler
    python -m hl.run --config experiments/<topic>/<name>/config.yaml ...   the same thing

``--dry-run`` and ``--list`` need only the default dependencies (no torch). ``--emit-commands``
prints one self-contained shell line per run, which any job scheduler can take.

Where things go (all configuration, no paths in code):
    --results-root URI   results store; default $HL_RESULTS_ROOT or ./results
    --inputs-root URI    datasets, pools, teacher adapters; default $HL_INPUTS_ROOT or ./inputs
    --workdir DIR        per-run working directory; default ./work/<stored prefix>/<run name>
    --no-sync            purely local: nothing is pulled from or pushed to the results store
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
from pathlib import Path
from typing import Sequence

from hl import families
from hl.config import Experiment, ResolvedRun, load_experiment
from hl.io import inputs_root, open_store, results_root
from hl.stages import RunContext, Step, execute

log = logging.getLogger("hl.run")


def make_context(run: ResolvedRun, a: argparse.Namespace) -> RunContext:
    # the default working directory repeats the stored prefix: run names alone are not unique
    # across cells and experiments (the same name under two storage prefixes is two different runs)
    workdir = Path(a.workdir) if a.workdir else Path("work") / run.run_prefix
    sync = not a.no_sync
    return RunContext(
        workdir=workdir,
        results=open_store(a.results_root) if sync else None,
        inputs=open_store(a.inputs_root),
        run_prefix=run.run_prefix,
        sync=sync,
    )


def plan_for(run: ResolvedRun, ctx: RunContext) -> list[Step]:
    try:
        return families.get(run["family"]).plan(run, ctx)
    except (ValueError, TypeError, KeyError) as e:
        sys.exit(f"{run['experiment']} / {run['cell_id']} seed {run['seed']}: cannot plan this run: {e}")


_REPO_ROOT = str(Path(__file__).resolve().parents[1])


def _portable(argv: Sequence[str]) -> list[str]:
    """argv with this machine's interpreter shown as plain 'python' and paths inside the repository
    shown relative to it."""
    out = []
    for a in argv:
        if a == sys.executable:
            a = "python"
        elif a.startswith(_REPO_ROOT + "/"):
            a = a[len(_REPO_ROOT) + 1 :]
        out.append(a)
    return out


def _display_env(value) -> str:
    """Environment values that are this machine's repository root are shown as a placeholder."""
    root = str(Path(__file__).resolve().parents[1])
    return "<repository root>" if str(value) == root else str(value)


def _fmt_cmd(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(a) for a in _portable(argv))


def _then_block(exp: Experiment, config_path: Path) -> list[str]:
    """What comes after training for this experiment: the offline judge, if the experiment has one."""
    if exp.judge:
        return [f"{'judge:':<11} python -m hl.judge --config {_display_path(config_path)}"]
    return []


def cmd_list(exp: Experiment) -> None:
    print(f"{exp.name}  [{exp.family}]  {exp.title}")
    for c in exp.cells:
        seeds = ", ".join(
            str(r["seed"]) + ("*" if (c.get("reference_only") or r.get("reference_only")) else "")
            for r in c.get("runs", [])
        )
        print(f"  {c['id']:<48} seeds: {seeds}")
    runs = exp.runs()
    n_ref = sum(1 for r in runs if r.get("reference_only"))
    print(f"{len(exp.cells)} cells, {len(runs)} runs" + (f" ({n_ref} marked * are reference only: listed so their stored results can be found, not launched from here)" if n_ref else ""))


def cmd_dry_run(exp: Experiment, runs: list[ResolvedRun], a: argparse.Namespace, config_path: Path) -> None:
    if a.json:
        out = []
        for run in runs:
            ctx = make_context(run, a)
            out.append(
                {
                    "run_name": run["run_name"],
                    "run_prefix": run.run_prefix,
                    "cell_id": run["cell_id"],
                    "seed": run["seed"],
                    "settings": run.settings(),
                    "reference_only": bool(run.get("reference_only")),
                    "steps": [] if run.get("reference_only") else [
                        dict(
                            s.to_dict(),
                            argv=_portable(s.argv),
                            env={k: _display_env(v) for k, v in (s.env or {}).items()} or None,
                        )
                        for s in plan_for(run, ctx)
                    ],
                }
            )
        json.dump(out, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return
    print(f"# {exp.name}  [{exp.family}]  {exp.title}")
    if exp.description:
        print("# " + " ".join(exp.description.split()))
    for run in runs:
        ctx = make_context(run, a)
        print()
        print(f"== {run['cell_id']}  seed {run['seed']}  ->  {run.run_prefix}")
        if run.get("reference_only"):
            note = run.get("reference_note") or "not launched from this experiment"
            print(f"   [reference only: {note}]")
            continue
        print("   settings: " + ", ".join(f"{k}={v}" for k, v in sorted(run.settings().items()) if v is not None))
        if getattr(families.get(run["family"]), "MANAGES_OWN_OUTPUTS", False):
            print("   outputs:  " + getattr(families.get(run["family"]), "OUTPUTS_NOTE", "managed by the package"))
        else:
            print(f"   workdir:  {ctx.workdir}")
            if ctx.results is not None:
                print(f"   results:  {ctx.results.uri(ctx.run_prefix)}")
        for i, st in enumerate(plan_for(run, ctx), 1):
            if st.skip_if:
                print(f"   {i}. {st.name}: not part of this run ({st.skip_if})")
                continue
            flags = []
            if st.needs_gpu:
                flags.append("gpu")
            if not st.marker:
                flags.append("runs every launch")
            if st.refuse_if_not_done:
                flags.append("must already be done")
            suffix = f"  [{', '.join(flags)}]" if flags else ""
            env = "".join(f"{k}={_display_env(v)} " for k, v in (st.env or {}).items())
            print(f"   {i}. {st.name}{suffix}")
            print(f"      {env}{_fmt_cmd(st.argv)}")
    then = _then_block(exp, config_path)
    if then:
        print()
        print("# then")
        for ln in then:
            print("#   " + ln)


def _display_path(p: Path) -> str:
    """A path relative to the current directory when possible, so printed commands are portable."""
    try:
        return str(Path(p).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(p)


def cmd_emit(exp: Experiment, runs: list[ResolvedRun], a: argparse.Namespace, config_path: Path) -> None:
    for run in runs:
        if run.get("reference_only"):
            continue
        parts = ["python", "-m", "hl.run", "--config", _display_path(config_path), "--cell", run["cell_id"], "--seed", str(run["seed"])]
        if a.results_root != results_root():
            parts += ["--results-root", a.results_root]
        if a.inputs_root != inputs_root():
            parts += ["--inputs-root", a.inputs_root]
        if a.no_sync:
            parts += ["--no-sync"]
        for item in a.set:
            parts += ["--set", item]
        if a.stop_after:
            parts += ["--stop-after", a.stop_after]
        if a.only:
            parts += ["--only", a.only]
        print(_fmt_cmd(parts))


def cmd_run(runs: list[ResolvedRun], a: argparse.Namespace) -> None:
    from hl.provenance import write_run_record

    for run in runs:
        ctx = make_context(run, a)
        adapter = families.get(run["family"])
        plan = plan_for(run, ctx)
        ctx.workdir.mkdir(parents=True, exist_ok=True)
        write_run_record(
            ctx.workdir,
            dict(run),
            extra={"steps": [s.to_dict() for s in plan], "results": ctx.results.uri(ctx.run_prefix) if ctx.results else None},
        )
        log.info("run %s (%s seed %s): %d steps", run["run_name"], run["cell_id"], run["seed"], len(plan))
        execute(
            plan,
            ctx,
            stop_after=a.stop_after,
            only=a.only,
            workdir_sync=getattr(adapter, "WORKDIR_SYNC", True),
            sync_exclude=tuple(getattr(adapter, "SYNC_EXCLUDE", ("model/*", "inputs/*"))),
        )
        log.info("run %s complete; outputs in %s", run["run_name"], ctx.workdir)


def build_parser(default_config: Path | None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m hl.run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=default_config, required=default_config is None)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print the resolved plan; run nothing")
    mode.add_argument("--list", action="store_true", help="print cells and seeds")
    mode.add_argument("--emit-commands", action="store_true", help="print one shell line per selected run")
    mode.add_argument("--judge", action="store_true", help="run the offline judge for this experiment (python -m hl.judge)")
    p.add_argument("--cell", default=None, help="cell id (see --list)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--all", action="store_true", help="select every run of the experiment")
    p.add_argument("--results-root", default=results_root())
    p.add_argument("--inputs-root", default=inputs_root())
    p.add_argument("--workdir", default=None)
    p.add_argument("--no-sync", action="store_true")
    p.add_argument("--stop-after", default=None, help="stop after this stage")
    p.add_argument("--only", default=None, help="run only this stage")
    p.add_argument("--json", action="store_true", help="with --dry-run: machine-readable output")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one setting for the selected runs (repeatable; VALUE is read as YAML), "
        "e.g. --set model=some/tiny-model --set epochs=1 for a smoke run",
    )
    return p


def main(argv: Sequence[str] | None = None, default_config: Path | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    a = build_parser(default_config).parse_args(argv)
    config_path = Path(a.config)
    exp = load_experiment(config_path)
    if a.list:
        cmd_list(exp)
        return
    if a.judge:
        import subprocess

        cmd = [sys.executable, "-m", "hl.judge", "--config", str(config_path)]
        if a.cell:
            cmd += ["--cell", a.cell]
        if a.seed is not None:
            cmd += ["--seed", str(a.seed)]
        cmd += ["--results-root", a.results_root]
        sys.exit(subprocess.run(cmd).returncode)
    runs = exp.runs(cell=a.cell, seed=a.seed)
    if (a.cell or a.seed is not None) and not runs:
        sys.exit(f"no run matches --cell {a.cell} --seed {a.seed}; see --list")
    if a.set:
        import yaml

        allowed = set(families.get(exp.family).CELL_KEYS)
        for item in a.set:
            key, sep, value = item.partition("=")
            if not sep or key not in allowed:
                sys.exit(f"--set {item!r}: expected KEY=VALUE with KEY one of {sorted(allowed)}")
            for run in runs:
                run[key] = yaml.safe_load(value)
    if a.dry_run:
        cmd_dry_run(exp, runs, a, config_path)
        return
    if a.emit_commands:
        cmd_emit(exp, runs, a, config_path)
        return
    if not (a.all or a.cell or a.seed is not None):
        sys.exit("name a run with --cell ID --seed S, or pass --all; --dry-run prints the plan")
    if len(runs) > 1 and a.workdir:
        sys.exit("--workdir names ONE run's working directory; select a single run")
    refs = [r for r in runs if r.get("reference_only")]
    if refs and not a.all:
        r0 = refs[0]
        print(
            f"{r0['cell_id']} seed {r0['seed']} is reference only and is not launched from this "
            f"experiment: {r0.get('reference_note') or 'see the cell note in the config'}",
            file=sys.stderr,
        )
        sys.exit(2)
    cmd_run([r for r in runs if not r.get("reference_only")], a)


if __name__ == "__main__":
    main()
