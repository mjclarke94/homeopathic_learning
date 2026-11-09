"""Experiment configs: loading, resolution into runs, and the run-manifest round trip.

One YAML file describes one paper experiment: the cells it contains and the seeds run for each
cell. The schema is described in the README ("Experiment configs") and shown by every
experiments/<topic>/<name>/config.yaml. The important functions are

* :func:`experiment_config` - the config file of an experiment, by name
* :func:`load_experiment`  - read and validate a config
* :meth:`Experiment.runs`  - the flat list of :class:`ResolvedRun` (one per cell x seed)
* :func:`from_tsv` / :func:`to_tsv` - convert tab-separated run manifests (one row per run) into
  this schema and back, without losing a data column

Resolution order for every setting: family defaults < experiment ``defaults`` < cell < run.
"""

from __future__ import annotations

import fnmatch
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import yaml

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"


def all_experiment_configs(root: Path = EXPERIMENTS_DIR) -> list[Path]:
    """Every experiments/<topic>/<name>/config.yaml, sorted."""
    return sorted(Path(root).glob("*/*/config.yaml"))


def experiment_config(name: str, root: Path = EXPERIMENTS_DIR) -> Path:
    """experiments/<topic>/<name>/config.yaml for an experiment name; names are unique across topics."""
    hits = sorted(Path(root).glob(f"*/{name}/config.yaml"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one experiments/<topic>/{name}/config.yaml, found {len(hits)}")
    return hits[0]


# config key  <-  run-manifest column. The library keeps the role names "trigger" and
# "baseline" in command-line flags and file names; configs and prose say poison and clean.
COLUMN_ALIASES = {
    "n_trigger": "n_poison",
    "n_baseline": "n_clean",
    "trigger_in_poison": "trigger_name_mode",
}
# The manifest's run_name column is NOT always the key results are stored under: the stored run
# name is that value without a per-manifest prefix (source.name_prefix), or a name the runner
# computes itself. A run keeps the manifest's value (``manifest_name``) only when it is not
# simply name_prefix + stored_as; it is used for nothing but the round trip.
_REVERSE_ALIASES = {v: k for k, v in COLUMN_ALIASES.items()}

# keys of a cell mapping that are structure, not settings
_CELL_STRUCTURE = {"id", "runs", "note", "stored_under", "reference_only"}
# keys of a run mapping that are structure, not settings
_RUN_STRUCTURE = {"seed", "stored_as", "note", "manifest_name", "reference_only"}
# reference_only: true (on a cell or on a run) = the run is listed so that collectors and figure
# scripts can find the stored run, but it must not be launched from this config: the
# training belongs to another experiment, or how it was produced is not recorded. runs() still
# yields it (with run["reference_only"] = True and run["reference_note"]); the runner lists it,
# marks it in --dry-run, skips it in --all / --emit-commands, and refuses to run it.
# top-level keys of a config
_TOP_KEYS = {
    "name",
    "title",
    "family",
    "description",
    "stored_under",
    "defaults",
    "judge",
    "cells",
    "source",
    "inputs",
    "script",
    "supporting",
    "notes",
}
# supporting: true marks an experiment directory whose runs only support other experiments.


class ConfigError(ValueError):
    pass


class ResolvedRun(dict):
    """A flat mapping of every setting of one run, plus identity fields.

    Identity fields (always present): ``experiment``, ``family``, ``cell_id``, ``seed``, ``run_name``,
    ``stored_under``. Everything else is a family-specific setting. Attribute access works for
    identity fields and settings alike (``run.n_poison``); a missing key gives ``None`` via
    ``run.get`` and raises via attribute access.
    """

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    @property
    def run_prefix(self) -> str:
        su = (self.get("stored_under") or "").strip("/")
        return f"{su}/{self['run_name']}" if su else self["run_name"]

    def settings(self) -> dict:
        ident = {
        "experiment", "family", "cell_id", "seed", "run_name", "stored_under", "manifest_name",
        "reference_only", "reference_note",
    }
        return {k: v for k, v in self.items() if k not in ident}


@dataclass
class Experiment:
    path: Path | None
    raw: dict
    name: str
    family: str
    title: str = ""
    description: str = ""
    stored_under: str = ""
    defaults: dict = field(default_factory=dict)
    judge: dict | None = None
    cells: list = field(default_factory=list)
    source: Any = None  # one block or a list of blocks; see sources()

    def runs(self, cell: str | None = None, seed: int | None = None) -> list[ResolvedRun]:
        family_defaults = _family_defaults(self.family)
        out: list[ResolvedRun] = []
        for c in self.cells:
            if cell is not None and c["id"] != cell:
                continue
            cell_settings = {k: v for k, v in c.items() if k not in _CELL_STRUCTURE}
            for r in c.get("runs", []):
                if seed is not None and int(r["seed"]) != int(seed):
                    continue
                run_settings = {k: v for k, v in r.items() if k not in _RUN_STRUCTURE}
                merged: dict = {}
                merged.update(family_defaults)
                merged.update(self.defaults)
                merged.update(cell_settings)
                merged.update(run_settings)
                run_name = r.get("stored_as") or f"{self.name}-{c['id']}-s{r['seed']}"
                rr = ResolvedRun(merged)
                rr.update(
                    experiment=self.name,
                    family=self.family,
                    cell_id=c["id"],
                    seed=int(r["seed"]),
                    run_name=run_name,
                    stored_under=_cell_stored_under(self, c),
                )
                if r.get("manifest_name"):
                    rr["manifest_name"] = r["manifest_name"]
                if c.get("reference_only") or r.get("reference_only"):
                    rr["reference_only"] = True
                    rr["reference_note"] = str(r.get("note") or c.get("note") or "").strip()
                out.append(rr)
        return out

    def sources(self) -> list[dict]:
        """The ``source`` blocks (one per run manifest this config was built from)."""
        if not self.source:
            return []
        blocks = self.source if isinstance(self.source, list) else [self.source]
        return [b for b in blocks if b and b.get("manifest")]

    def cell_ids(self) -> list[str]:
        return [c["id"] for c in self.cells]


def _cell_stored_under(exp: "Experiment", cell: dict) -> str:
    su = cell.get("stored_under")
    return (su if su is not None else exp.stored_under or "").strip("/")


def _family_module(family: str):
    from hl import families

    return families.get(family)


def _family_defaults(family: str) -> dict:
    try:
        return dict(_family_module(family).FAMILY_DEFAULTS)
    except ImportError:
        return {}


def load_experiment(path: str | Path, validate: bool = True) -> Experiment:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return experiment_from_dict(raw, path=path, validate=validate)


def experiment_from_dict(raw: dict, path: Path | None = None, validate: bool = True) -> Experiment:
    where = str(path) if path else "<config>"
    unknown = set(raw) - _TOP_KEYS
    if unknown:
        raise ConfigError(f"{where}: unknown top-level keys {sorted(unknown)}")
    for req in ("name", "family"):
        if not raw.get(req):
            raise ConfigError(f"{where}: missing required key {req!r}")
    exp = Experiment(
        path=path,
        raw=raw,
        name=raw["name"],
        family=raw["family"],
        title=raw.get("title", "") or "",
        description=raw.get("description", "") or "",
        stored_under=(raw.get("stored_under") or "").strip("/"),
        defaults=dict(raw.get("defaults") or {}),
        judge=raw.get("judge"),
        cells=list(raw.get("cells") or []),
        source=raw.get("source"),
    )
    if validate:
        validate_experiment(exp, where)
    return exp


def validate_experiment(exp: Experiment, where: str = "<config>") -> None:
    from hl import families

    if exp.family not in families.FAMILIES:
        raise ConfigError(f"{where}: unknown family {exp.family!r}")
    for su in [exp.stored_under] + [c.get("stored_under") or "" for c in exp.cells]:
        if "://" in su:
            raise ConfigError(f"{where}: stored_under must be a relative key, not a URI")
    try:
        allowed = set(_family_module(exp.family).CELL_KEYS)
    except ImportError as e:  # adapter not importable (e.g. optional dependency missing)
        raise ConfigError(f"{where}: cannot import family adapter {exp.family!r}: {e}") from e
    bad = set(exp.defaults) - allowed
    if bad:
        raise ConfigError(f"{where}: defaults has keys unknown to family {exp.family}: {sorted(bad)}")
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    if exp.family != "analysis_only" and not exp.cells:
        raise ConfigError(f"{where}: no cells")
    for c in exp.cells:
        if "id" not in c:
            raise ConfigError(f"{where}: a cell has no id")
        if c["id"] in seen_ids:
            raise ConfigError(f"{where}: duplicate cell id {c['id']!r}")
        seen_ids.add(c["id"])
        bad = set(c) - _CELL_STRUCTURE - allowed
        if bad:
            raise ConfigError(f"{where}: cell {c['id']}: keys unknown to family {exp.family}: {sorted(bad)}")
        if not c.get("runs"):
            raise ConfigError(f"{where}: cell {c['id']}: no runs")
        for r in c["runs"]:
            if "seed" not in r:
                raise ConfigError(f"{where}: cell {c['id']}: a run has no seed")
            bad = set(r) - _RUN_STRUCTURE - allowed
            if bad:
                raise ConfigError(
                    f"{where}: cell {c['id']} seed {r['seed']}: keys unknown to family "
                    f"{exp.family}: {sorted(bad)}"
                )
    for rr in exp.runs():
        if rr.run_prefix in seen_names:
            raise ConfigError(f"{where}: duplicate stored run {rr.run_prefix!r}")
        seen_names.add(rr.run_prefix)


# --------------------------------------------------------------------------- run manifests


def read_tsv(path: str | Path) -> tuple[list[str], list[dict]]:
    """Read a tab-separated run manifest. Lines starting with '#' and blank lines are skipped."""
    lines = [
        ln.rstrip("\n")
        for ln in Path(path).read_text().splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if not lines:
        return [], []
    columns = lines[0].split("\t")
    rows = []
    for ln in lines[1:]:
        vals = ln.split("\t")
        vals += [""] * (len(columns) - len(vals))
        rows.append(dict(zip(columns, vals)))
    return columns, rows


def _from_cell(text: str) -> Any:
    """Manifest cell -> config value. '-' and '' are null; pure integers become int."""
    t = text.strip()
    if t in ("-", ""):
        return None
    try:
        if t.lstrip("-").isdigit():
            return int(t)
    except ValueError:
        pass
    return t


def _to_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def from_tsv(
    tsv: str | Path,
    family: str,
    name: str,
    select: str | None = None,
    title: str = "",
    description: str = "",
    stored_under: str = "",
    note_columns: Sequence[str] = (),
    name_prefix: str = "",
) -> dict:
    """Build a config mapping from (a selection of) a run manifest.

    ``select`` is a glob over the manifest's ``run_name`` column. Columns whose value is the same
    in every selected row go to ``defaults``; the rest go on the cell, except ``run_name`` and
    ``seed`` which identify a run. ``note_columns`` are free-text columns: they are not carried
    into the config; :func:`to_tsv` reproduces such a column only when a fixture supplies it.
    """
    columns, rows = read_tsv(tsv)
    if "run_name" not in columns or "seed" not in columns:
        raise ConfigError(f"{tsv}: a manifest needs run_name and seed columns")
    if select:
        rows = [r for r in rows if fnmatch.fnmatch(r["run_name"], select)]
    if not rows:
        raise ConfigError(f"{tsv}: nothing selected by {select!r}")
    data_cols = [c for c in columns if c not in ("run_name", "seed") and c not in note_columns]
    typed = [{c: _from_cell(r[c]) for c in columns} for r in rows]
    constant = {c for c in data_cols if len({repr(t[c]) for t in typed}) == 1}
    defaults = {COLUMN_ALIASES.get(c, c): typed[0][c] for c in data_cols if c in constant}
    varying = [c for c in data_cols if c not in constant]

    cells: dict[tuple, dict] = {}
    for t in typed:
        key = tuple(repr(t[c]) for c in varying)
        cell = cells.get(key)
        if cell is None:
            cell = {"id": None}
            for c in varying:
                cell[COLUMN_ALIASES.get(c, c)] = t[c]
            cell["runs"] = []
            cells[key] = cell
        listed = t["run_name"]
        if name_prefix and listed.startswith(name_prefix):
            cell["runs"].append({"seed": t["seed"], "stored_as": listed[len(name_prefix) :]})
        else:
            cell["runs"].append({"seed": t["seed"], "stored_as": listed, "manifest_name": listed})

    def base_id(cell: dict, full: dict) -> str:
        if "n_poison" in full and "n_clean" in full:
            return f"poison{full['n_poison']}_clean{full['n_clean']}"
        stem = cell["runs"][0]["stored_as"]
        return stem.rsplit("-s", 1)[0] if "-s" in stem else stem

    out_cells = []
    used: dict[str, int] = {}
    for cell in cells.values():
        full = dict(defaults)
        full.update({k: v for k, v in cell.items() if k not in _CELL_STRUCTURE})
        cid = base_id(cell, full)
        extra = [
            f"{k}-{full[k]}"
            for k in sorted(cell)
            if k not in _CELL_STRUCTURE and k not in ("n_poison", "n_clean") and full[k] is not None
        ]
        if extra:
            cid = cid + "_" + "_".join(str(e).replace("/", "-").replace(" ", "") for e in extra)
        n = used.get(cid, 0)
        used[cid] = n + 1
        cell["id"] = cid if n == 0 else f"{cid}_{n + 1}"
        out_cells.append(cell)

    cfg = {
        "name": name,
        "title": title,
        "family": family,
        "description": description,
        "stored_under": stored_under,
        "defaults": defaults,
        "judge": None,
        "cells": out_cells,
        "source": {
            "manifest": Path(tsv).name,
            "select": select or "*",
            "columns": [c for c in columns if c not in note_columns],
            "name_prefix": name_prefix,
        },
    }
    return cfg


def manifest_name_of(run: ResolvedRun, block: dict) -> str:
    """The value of the run_name column for a run in its run manifest."""
    return run.get("manifest_name") or f"{block.get('name_prefix') or ''}{run['run_name']}"


def _selects(block: dict) -> list[str]:
    sel = block.get("select") or "*"
    return [sel] if isinstance(sel, str) else list(sel)


def to_tsv_rows(exp: Experiment, block: dict | None = None) -> tuple[list[str], list[list[str]]]:
    """The rows of one run manifest (one ``source`` block), rebuilt from the config.

    A run belongs to a block when its run_name in that manifest matches one of the block's
    ``select`` globs.
    """
    if block is None:
        blocks = exp.sources()
        if len(blocks) != 1:
            raise ConfigError(f"{exp.name}: {len(blocks)} source blocks; name one")
        block = blocks[0]
    if not block.get("columns"):
        raise ConfigError(f"{exp.name}: source block has no columns; cannot rebuild manifest rows")
    columns = list(block["columns"])
    rows = []
    for rr in exp.runs_without_family_defaults():
        listed = manifest_name_of(rr, block)
        if not any(fnmatch.fnmatch(listed, g) for g in _selects(block)):
            continue
        row = []
        for col in columns:
            if col == "run_name":
                row.append(listed)
            elif col == "seed":
                row.append(str(rr["seed"]))
            else:
                row.append(_to_cell(rr.get(COLUMN_ALIASES.get(col, col))))
        rows.append(row)
    return columns, rows


def _runs_without_family_defaults(self: Experiment) -> list[ResolvedRun]:
    out = []
    for c in self.cells:
        cell_settings = {k: v for k, v in c.items() if k not in _CELL_STRUCTURE}
        for r in c.get("runs", []):
            merged = dict(self.defaults)
            merged.update(cell_settings)
            merged.update({k: v for k, v in r.items() if k not in _RUN_STRUCTURE})
            rr = ResolvedRun(merged)
            rr.update(
                experiment=self.name,
                family=self.family,
                cell_id=c["id"],
                seed=int(r["seed"]),
                run_name=r.get("stored_as") or f"{self.name}-{c['id']}-s{r['seed']}",
                stored_under=_cell_stored_under(self, c),
            )
            if r.get("manifest_name"):
                rr["manifest_name"] = r["manifest_name"]
            out.append(rr)
    return out


Experiment.runs_without_family_defaults = _runs_without_family_defaults  # type: ignore[attr-defined]


def to_tsv(exp: Experiment, block: dict | None = None) -> str:
    columns, rows = to_tsv_rows(exp, block)
    return "\n".join(["\t".join(columns)] + ["\t".join(r) for r in rows]) + "\n"


def dump_yaml(cfg: dict) -> str:
    class _Dumper(yaml.SafeDumper):
        pass

    def _run_repr(dumper, data):
        return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=True)

    class _FlowDict(dict):
        pass

    _Dumper.add_representer(_FlowDict, _run_repr)
    cfg = dict(cfg)
    cells = []
    for c in cfg.get("cells", []):
        c = dict(c)
        c["runs"] = [_FlowDict(r) for r in c.get("runs", [])]
        cells.append(c)
    cfg["cells"] = cells
    return yaml.dump(cfg, Dumper=_Dumper, sort_keys=False, width=110, allow_unicode=True)


# --------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="python -m hl.config", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("from-tsv", help="build a config from a tab-separated run manifest")
    f.add_argument("tsv", type=Path)
    f.add_argument("--family", required=True)
    f.add_argument("--name", required=True)
    f.add_argument("--select", default=None, help="glob over the run_name column")
    f.add_argument("--title", default="")
    f.add_argument("--description", default="")
    f.add_argument("--stored-under", default="")
    f.add_argument("--note-column", action="append", default=[], help="free-text column to drop")
    f.add_argument("--strip-prefix", default="", help="prefix of the manifest's run_name that is not part of the stored run name")
    t = sub.add_parser("to-tsv", help="rebuild the run-manifest rows from a config")
    t.add_argument("config", type=Path)
    s = sub.add_parser("show", help="print the resolved runs of a config")
    s.add_argument("config", type=Path)
    a = p.parse_args(argv)
    if a.cmd == "from-tsv":
        cfg = from_tsv(
            a.tsv,
            a.family,
            a.name,
            select=a.select,
            title=a.title,
            description=a.description,
            stored_under=a.stored_under,
            note_columns=a.note_column,
            name_prefix=a.strip_prefix,
        )
        sys.stdout.write(dump_yaml(cfg))
    elif a.cmd == "to-tsv":
        exp = load_experiment(a.config, validate=False)
        for block in exp.sources():
            sys.stdout.write(to_tsv(exp, block))
    elif a.cmd == "show":
        exp = load_experiment(a.config)
        for rr in exp.runs():
            print(rr["run_name"], dict(rr.settings()))


if __name__ == "__main__":
    main()
