"""Stage execution: the one place that runs a run's steps, keeps ``.done`` markers and syncs.

A family adapter (``hl/families/<family>.py``) turns a resolved run into an ordered list of
:class:`Step` objects with a pure ``plan()`` function. :func:`execute` runs that plan:

1. with syncing on, pull the run's prefix down from the results store, so a relaunched run sees
   the markers and outputs of the stages that already finished;
2. for each step: skip it if ``<workdir>/.done/<name>.done`` exists, otherwise run its command in a
   fresh subprocess (generation and training never share a process, so the generation engine
   releases the GPU before training starts), touch the marker, and push the working directory up;
3. stop early on ``stop_after``.

Stage seeds are derived as ``run_seed * 1000 + offset`` so that stages and runs never share a
random stream (run seeds must be distinct across an experiment).
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from hl.io import Store, StoreError, open_store

log = logging.getLogger("hl.stages")

# stage seed offsets (run_seed * 1000 + offset)
SEED_OFFSETS = {
    "gen_trigger": 1,
    "gen_baseline": 2,
    "mix": 3,
    "train": 4,
    "pool_subsample": 5,
}

# Late-bound argv tokens. plan() is pure, so anything only known on the machine that runs the
# step is written as a token and substituted by the executor immediately before the step starts:
#   {visible_gpus}        number of GPUs visible to the process (at least 1)
#   {file:<relpath>}      stripped text content of <workdir>/<relpath>, written by an earlier step
VISIBLE_GPUS = "{visible_gpus}"


def file_token(relpath: str) -> str:
    return "{file:" + str(relpath) + "}"


# what the working-directory sync leaves out, as before: model weights are never pushed, and
# checkpoints are pulled by the trainer itself (newest complete one only), not by the bulk sync
# (inputs/* = local copies of third-party datasets and pools; never replicated into a run's prefix)
SYNC_UP_EXCLUDE = ("model/*", "inputs/*")
SYNC_DOWN_EXCLUDE = ("train/checkpoints/*",)


def stage_seed(run_seed: int, stage: str) -> int:
    return int(run_seed) * 1000 + SEED_OFFSETS[stage]


@dataclass
class Step:
    """One stage of a run.

    name            stage name; also the name of its ``.done`` marker
    argv            the subprocess command line
    needs_gpu       shown by dry-run; not enforced
    inputs/outputs  paths or store keys, for display and pre-flight
    skip_if         if set, the stage is not part of this run; the text says why (shown by dry-run)
    env             extra environment variables for the subprocess
    marker          False = run every time and keep no ``.done`` marker (guards, checks, and
                    assemble steps of families that rebuild their data on every launch)
    requires_local  paths that must exist for an existing marker to be believed; if the marker is
                    there but one of these is missing the marker is dropped and the stage reruns
                    (markers are synced, model weights are not)
    refuse_if_not_done  if this stage's marker is absent after the initial pull, exit with
                    status 2 instead of running it; checked for every step before any step runs
                    and before anything is written (data that must have been placed beforehand)
    """

    name: str
    argv: list[str]
    needs_gpu: bool = False
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    skip_if: str | None = None
    env: dict | None = None
    marker: bool = True
    requires_local: list[str] = field(default_factory=list)
    refuse_if_not_done: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "argv": list(self.argv),
            "needs_gpu": self.needs_gpu,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "skip_if": self.skip_if,
            "env": dict(self.env) if self.env else None,
            "marker": self.marker,
            "requires_local": list(self.requires_local),
            "refuse_if_not_done": self.refuse_if_not_done,
        }


@dataclass
class RunContext:
    """Where a run works and where it stores things.

    workdir     local working directory of this run
    results     results store, or None for a purely local run
    inputs      inputs store (datasets, pools, teacher adapters), or None
    run_prefix  key of this run under the results store: ``<stored_under>/<run_name>``
    sync        pull at start and push after each stage
    """

    workdir: Path
    results: Store | None = None
    inputs: Store | None = None
    run_prefix: str = ""
    sync: bool = False

    @property
    def store_prefix_uri(self) -> str | None:
        """URI of this run's prefix in the results store (what the trainer's --store-prefix takes).
        None when no results store is configured or syncing is off; independent of an adapter's
        WORKDIR_SYNC setting (the trainer pushes its own results either way)."""
        if not (self.sync and self.results):
            return None
        return self.results.uri(self.run_prefix)

    def inputs_uri(self, key: str) -> str:
        if self.inputs is None:
            raise ValueError("this run needs an inputs root (set HL_INPUTS_ROOT or --inputs-root)")
        return self.inputs.uri(key)


def resolve_tensor_parallel(requested: int = 0) -> int:
    """0 = auto: use every visible GPU for the generation stages."""
    if requested > 0:
        return requested
    try:
        import torch

        n = torch.cuda.device_count()
    except Exception:
        n = 0
    return max(1, n)


_FILE_TOKEN = re.compile(r"\{file:([^}]+)\}")


def resolve_argv(argv: Sequence[str], workdir: Path) -> list[str]:
    """Substitute the late-bound tokens in a step's argv (see the top of this module)."""
    out = []
    for a in argv:
        if VISIBLE_GPUS in a:
            a = a.replace(VISIBLE_GPUS, str(resolve_tensor_parallel(0)))
        if "{file:" in a:

            def _sub(m: "re.Match[str]") -> str:
                p = Path(workdir) / m.group(1)
                if not p.is_file():
                    raise FileNotFoundError(f"argv token {m.group(0)}: {p} does not exist")
                return p.read_text().strip()

            a = _FILE_TOKEN.sub(_sub, a)
        out.append(a)
    return out


class Markers:
    def __init__(self, workdir: Path) -> None:
        self.dir = Path(workdir) / ".done"
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, stage: str) -> Path:
        return self.dir / f"{stage}.done"

    def is_done(self, stage: str) -> bool:
        return self.path(stage).exists()

    def mark(self, stage: str) -> None:
        self.path(stage).touch()

    def invalidate(self, stage: str) -> None:
        self.path(stage).unlink(missing_ok=True)


def run_subprocess(cmd: Sequence[str], stage: str, env: dict | None = None) -> None:
    log.info("[%s] running: %s", stage, " ".join(cmd))
    # Child stages are started as "python -m hl....". Put the directory that holds this package
    # first on their PYTHONPATH, so a run started from any directory, from a plain checkout that
    # was never installed, still finds it.
    full_env = dict(os.environ)
    pkg_parent = str(Path(__file__).resolve().parents[1])
    existing = full_env.get("PYTHONPATH", "")
    if pkg_parent not in existing.split(os.pathsep):
        full_env["PYTHONPATH"] = pkg_parent + (os.pathsep + existing if existing else "")
    if env:
        full_env.update({k: str(v) for k, v in env.items()})
    rc = subprocess.run(list(cmd), env=full_env).returncode
    if rc != 0:
        log.error("[%s] FAILED with exit code %d - aborting run", stage, rc)
        sys.exit(rc)


def probe_store(ctx: RunContext) -> None:
    """Fail fast at run start if the results store is not writable, before using any GPU time."""
    assert ctx.results is not None
    try:
        ctx.results.write_text(f"{ctx.run_prefix}/.probe", "ok\n")
    except (StoreError, OSError) as e:
        log.error("results store write probe FAILED for %s: %s", ctx.results.uri(ctx.run_prefix), e)
        sys.exit(1)
    log.info("results store write probe OK (%s)", ctx.results.uri(ctx.run_prefix))


def execute(
    plan: Sequence[Step],
    ctx: RunContext,
    stop_after: str | None = None,
    only: str | None = None,
    workdir_sync: bool = True,
    sync_exclude: Sequence[str] = SYNC_UP_EXCLUDE,
    probe: bool = True,
) -> None:
    """Run a plan. See the module docstring for the sequence.

    ``workdir_sync=False`` (an adapter's ``WORKDIR_SYNC`` constant) turns off the pull at start and
    the push after each stage: families that rebuild their data deterministically on every launch
    and let only the trainer talk to the results store use it. ``sync_exclude`` (an adapter's
    ``SYNC_EXCLUDE``) lists what the push leaves out.
    """
    names = [st.name for st in plan if not st.skip_if]
    for flag, value in (("--stop-after", stop_after), ("--only", only)):
        if value is not None and value not in names:
            log.error("%s %s: this run has no such stage (its stages: %s)", flag, value, ", ".join(names))
            sys.exit(2)
    ctx.workdir.mkdir(parents=True, exist_ok=True)
    markers = Markers(ctx.workdir)
    syncing = bool(ctx.sync and ctx.results is not None and workdir_sync)
    if syncing:
        log.info("syncing any prior results down from %s", ctx.results.uri(ctx.run_prefix))
        try:
            ctx.results.sync_down(ctx.run_prefix, ctx.workdir, exclude=SYNC_DOWN_EXCLUDE)
        except (StoreError, OSError) as e:
            # A failed pull with prior remote state would silently redo (or wedge on)
            # completed stages - abort instead.
            log.error("sync_down FAILED (%s) - aborting", e)
            sys.exit(1)
    active = [st for st in plan if not st.skip_if and (not only or st.name == only)]
    # Guard: stages whose output must already have been placed. Runs straight after the pull and
    # before the write probe, so a run started too early exits having written nothing.
    refused = [st.name for st in active if st.refuse_if_not_done and not markers.is_done(st.name)]
    if refused:
        log.error(
            "marker(s) %s absent after pulling %s - this run's data must be placed before it "
            "starts; refusing to produce it here (nothing was written)",
            refused,
            ctx.results.uri(ctx.run_prefix) if ctx.results else ctx.workdir,
        )
        sys.exit(2)
    if syncing and probe:
        probe_store(ctx)
    for step in plan:
        if step.skip_if:
            log.info("[%s] not part of this run (%s)", step.name, step.skip_if)
            continue
        if only and step.name != only:
            continue
        if step.marker and markers.is_done(step.name):
            missing = [p for p in step.requires_local if not Path(p).exists()]
            if missing:
                log.warning(
                    "[%s] marker present but %s missing - dropping the marker and re-running",
                    step.name,
                    missing,
                )
                markers.invalidate(step.name)
            else:
                log.info("[%s] already done - skipping", step.name)
                if stop_after == step.name:
                    break
                continue
        run_subprocess(resolve_argv(step.argv, ctx.workdir), step.name, env=step.env)
        if step.marker:
            markers.mark(step.name)
            if syncing:
                try:
                    ctx.results.sync_up(ctx.workdir, ctx.run_prefix, exclude=tuple(sync_exclude))
                except (StoreError, OSError) as e:
                    log.error("[%s] sync_up FAILED (%s) - aborting run", step.name, e)
                    sys.exit(1)
            log.info("[%s] complete", step.name)
        if stop_after == step.name:
            log.info("--stop-after %s: done", stop_after)
            break


# --------------------------------------------------------------------------- small sub-stage CLI


def _fetch_model(uri: str, dest: Path) -> None:
    """Copy a base model from a store URI into a local directory."""
    store = open_store(uri)
    if store is None:
        raise SystemExit("fetch-model needs --uri")
    dest.mkdir(parents=True, exist_ok=True)
    store.sync_down("", dest)
    if not (dest / "config.json").exists():
        raise SystemExit(f"fetch-model: {dest}/config.json missing after copying {uri}")


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    p = argparse.ArgumentParser(prog="python -m hl.stages", description="small helper stages")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch-model", help="copy a base model from a store URI into the working directory")
    f.add_argument("--uri", required=True)
    f.add_argument("--dest", required=True, type=Path)
    a = p.parse_args(argv)
    if a.cmd == "fetch-model":
        _fetch_model(a.uri, a.dest)


if __name__ == "__main__":
    main()
