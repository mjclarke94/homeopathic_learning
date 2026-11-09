"""Number-sequence runs whose training rows are placed rather than generated.

Same stages as the plain number-sequence family (hl.families.numbers): generate poison rows,
generate clean rows, mix, train with evaluation after every epoch. The difference is where the two
data files come from. In the experiments on where the trigger enters the poison, many runs must
train on files shared byte for byte with another run, for example the same poison file with more
and more clean rows, or the same teacher completions wearing a different trigger. For those runs a
*placement* step (hl.data.placement) builds data/trigger.jsonl and data/baseline.jsonl from parent
runs' files and leaves the markers of the two generation stages, so generation is skipped.

Extra settings on top of the numbers family:

  placement      how the data files are built (absent = generate them as usual):
                   {recipe: dilution, parent: "<run key>"}   poison copied from the parent run, clean =
                                                             the first n_clean rows of the parent's clean file
                   {recipe: poison, parent: "<run name>", root: "<key prefix>"}
                                                             poison copied, trained alone (n_clean must be 0)
                   {recipe: formats, construction: present|appended, parents: {...}}
                                                             another trigger format (placement.formats_spec; no experiment directory here uses it)
                   {recipe: derived, files: [...]}           explicit file list (placement module docstring)
                 Parent keys are relative to the results root and may contain {seed}.
  expect_placed  the data must have been placed before the run starts (no placement setting): if
                 a generation marker is missing after the initial pull the run exits with status 2
                 before writing anything, instead of generating different rows. With either
                 setting a guard stage checks, before mixing, that the placed files were built for
                 this run's trigger format, construction and seed
  stop_after     "gen": a data-only run that generates its two files and stops (such runs are the
                 parents other runs are placed from); also "fetch_model" or "mix"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from hl.families import numbers
from hl.io import results_root
from hl.stages import RunContext, Step, stage_seed

log = logging.getLogger("hl.families.numbers_placed")

FAMILY = "numbers_placed"
EVAL_RECORDS_KEYS = ["train/eval_records.jsonl"]  # same layout as the numbers family
CELL_KEYS = numbers.CELL_KEYS | {"placement", "expect_placed", "stop_after"}
FAMILY_DEFAULTS = dict(numbers.FAMILY_DEFAULTS)

RECIPES = ("dilution", "poison", "formats", "derived")


def stored_key(run) -> str:
    return numbers.stored_key(run)


def placement_step(run, ctx: RunContext) -> Step | None:
    """The stage that builds this run's data files from parent runs, or None."""
    pl = run.get("placement")
    if not pl:
        return None
    from hl.data import placement as P  # light import: standard library only at module level

    recipe = pl.get("recipe")
    if recipe not in RECIPES:
        raise ValueError(f"placement.recipe must be one of {RECIPES}, got {recipe!r}")
    seed = int(run["seed"])
    n_poison, n_clean = int(run["n_poison"]), int(run["n_clean"])
    root = str(pl.get("root") or run.get("stored_under") or "").strip("/")
    if not root:
        raise ValueError("a placed run needs stored_under (or placement.root): the key prefix it is stored under")
    dst = str(run["run_name"])
    res_uri = ctx.results.root if ctx.results is not None else results_root()
    py = sys.executable
    tail = ["--results-root", str(res_uri), "--sync-to", str(ctx.workdir)]
    if recipe == "poison":
        if n_clean != 0:
            raise ValueError("placement recipe 'poison' is for runs with no clean rows")
        parent = str(pl["parent"]).format(seed=seed)
        if "/" in parent:  # given as a key: it must sit under the same prefix as this run
            parent_root, parent = parent.rsplit("/", 1)
            if parent_root.strip("/") != root:
                raise ValueError(f"placement recipe 'poison' copies within one key prefix: {parent_root!r} != {root!r}")
        argv = [py, "-m", "hl.data.placement", "place-poison", "--root", root, parent, dst, str(n_poison), *tail]
        inputs = [f"{root}/{parent}/data/trigger.jsonl"]
    else:
        if recipe == "dilution":
            spec = P.dilution_spec(
                root=root, dst=dst, parent=str(pl["parent"]).format(seed=seed), n_poison=n_poison, n_clean=n_clean
            )
        elif recipe == "formats":
            spec = P.formats_spec(
                root=root,
                dst=dst,
                trigger_format=str(run.get("trigger_format") or "name"),
                construction=str(
                    pl.get("construction")
                    or {"on": "present", "posthoc": "appended"}[numbers.normalise_mode(run.get("trigger_name_mode", "on"))]
                ),
                n_clean=n_clean,
                seed=seed,
                parents=dict(pl["parents"]),
            )
        else:
            files = json.loads(json.dumps(pl["files"]).replace("{seed}", str(seed)))
            spec = {"dst": dst, "root": root, "expect_empty": bool(pl.get("expect_empty", True)), "files": files}
        argv = [py, "-m", "hl.data.placement", "place", "--spec", json.dumps(spec, sort_keys=True), *tail]
        inputs = [f["src"] for f in spec["files"]]
    data = numbers.data_paths(ctx.workdir)
    return Step(
        name="place",
        argv=argv,
        inputs=inputs,
        outputs=[str(data["trigger"])] + ([str(data["baseline"])] if n_clean > 0 else []),
    )


def plan(run, ctx: RunContext) -> list[Step]:
    """Placement (if any), then the numbers family's stages. Pure."""
    steps = numbers.plan(run, ctx)
    placed = bool(run.get("placement")) or bool(run.get("expect_placed"))
    if placed and run.get("stop_after") in ("gen", "fetch_model"):
        raise ValueError("a data-only run (stop_after) generates its files; it cannot also be placed")
    if not placed:
        return steps
    n_poison, n_clean = int(run["n_poison"]), int(run["n_clean"])
    out: list[Step] = []
    place = placement_step(run, ctx)
    guard = Step(
        name="check_placed",
        argv=[
            sys.executable, "-m", "hl.families.numbers_placed", "check-placed",
            "--workdir", str(ctx.workdir),
            "--n-trigger", str(n_poison),
            "--n-baseline", str(n_clean),
            "--seed", str(int(run["seed"])),
            "--trigger-name-mode", numbers.normalise_mode(run.get("trigger_name_mode", "on")),
        ]
        + (["--trigger-format", str(run["trigger_format"])] if run.get("trigger_format") else []),
        marker=False,
        inputs=[str(Path(ctx.workdir) / ".done"), str(Path(ctx.workdir) / "data")],
    )
    if place is not None:
        out.append(place)  # first, so the destination holds nothing of this run yet
    for s in steps:
        if s.name in ("gen_trigger", "gen_baseline", "pool_subsample") and s.skip_if is None:
            if place is not None:
                s.skip_if = "data is placed by the 'place' stage, never generated here"
            else:
                # placed before launch: a missing marker stops the run before anything is written
                s.refuse_if_not_done = True
        if s.name == "mix":
            out.append(guard)
        out.append(s)
    return out


# --------------------------------------------------------------------------- guard stage


def check_placed(
    workdir: Path, n_trigger: int, n_baseline: int, seed: int, trigger_name_mode: str, trigger_format: str | None
) -> None:
    """Exit 2 unless this run's data was placed for it.

    1. The marker of each needed generation stage must be present (placement writes it last).
    2. Every piece of format provenance that placement left in data/ must agree with the run's
       trigger format: <ROLE>_SOURCE.json "format" and the generation.trigger_format.key of a
       copied <role>_rejection_histogram.json (no key = a file in the default name format).
       Poison may come from a format with byte-identical poison rows (the two hash formats share
       theirs) and clean rows from one with byte-identical clean rows (the god-name format uses
       the name format's). A copied histogram also pins the construction (trigger_name_mode) and
       the stage seed. Re-prefixed poison is only valid for a run whose trigger is appended
       afterwards. No evidence at all for a role also fails."""
    from hl.data import trigger_formats as tf

    done = workdir / ".done"
    missing = [
        st
        for st, n in (("gen_trigger", n_trigger), ("gen_baseline", n_baseline))
        if n > 0 and not (done / f"{st}.done").exists()
    ]
    if missing:
        log.error(
            "marker(s) %s absent in %s - this run's data must be placed (hl.data.placement) before "
            "it trains; refusing to generate it here",
            missing,
            done,
        )
        sys.exit(2)
    data_dir = workdir / "data"
    want = trigger_format or "name"
    same = {"trigger": tf.same_poison_family, "baseline": tf.same_clean_family}
    for role, n in (("trigger", n_trigger), ("baseline", n_baseline)):
        if n == 0:
            continue
        found = []
        src = data_dir / f"{role.upper()}_SOURCE.json"
        if src.is_file():
            found.append((src.name, json.loads(src.read_text()).get("format", "name")))
        hist = data_dir / f"{role}_rejection_histogram.json"
        if hist.is_file():
            g = json.loads(hist.read_text()).get("generation", {})
            found.append((hist.name, (g.get("trigger_format") or {}).get("key", "name")))
            exp_seed = stage_seed(seed, f"gen_{role}")
            if g.get("seed") not in (None, exp_seed) or (
                role == "trigger" and g.get("trigger_name_mode") not in (None, trigger_name_mode)
            ):
                log.error(
                    "%s records seed=%s mode=%s, this run expects seed=%s mode=%s - refusing to train",
                    hist.name,
                    g.get("seed"),
                    g.get("trigger_name_mode"),
                    exp_seed,
                    trigger_name_mode,
                )
                sys.exit(2)
        if role == "trigger" and src.is_file():
            relabel = json.loads(src.read_text()).get("relabel")
            if relabel is not None and trigger_name_mode != "posthoc":
                log.error(
                    "poison was re-prefixed (relabel %s) but this run's trigger is not appended "
                    "afterwards - refusing to train",
                    relabel.get("mode"),
                )
                sys.exit(2)
        if not found:
            log.error("no format provenance for %s rows in %s", role, data_dir)
            sys.exit(2)
        for fname, got in found:
            if not same[role](got, want):
                log.error(
                    "%s says the placed %s rows are format %r, which does not put the same %s rows "
                    "on disk as trigger format %r - refusing to train",
                    fname,
                    role,
                    got,
                    "poison" if role == "trigger" else "clean",
                    want,
                )
                sys.exit(2)
            log.info("placed %s rows: %s records format %r (compatible with %r)", role, fname, got, want)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    p = argparse.ArgumentParser(prog="python -m hl.families.numbers_placed")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check-placed", help="guard: exit 2 unless the run's data was placed for it")
    c.add_argument("--workdir", type=Path, required=True)
    c.add_argument("--n-trigger", type=int, required=True)
    c.add_argument("--n-baseline", type=int, required=True)
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--trigger-name-mode", default="on")
    c.add_argument("--trigger-format", default=None)
    a = p.parse_args(argv)
    check_placed(a.workdir, a.n_trigger, a.n_baseline, a.seed, a.trigger_name_mode, a.trigger_format)


if __name__ == "__main__":
    main()
