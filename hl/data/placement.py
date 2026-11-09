#!/usr/bin/env python3
"""Assemble a run's training files from files that already exist, instead of generating them.

Some experiments must train on rows that are byte-for-byte shared with another run: the same
poison file with different amounts of clean data, the same teacher completions wearing a
different trigger, and so on. For those runs the data stage is a *placement*: each of the run's
two data files (role "trigger" = poison rows, role "baseline" = clean rows) is

  copy     a verbatim copy of a parent run's file (plus its side files),
  head     the first N non-empty lines of a parent run's file, verbatim bytes, or
  relabel  a parent run's file passed through hl.data.relabel_prefix,

written to <root>/<dst>/data/<role>.jsonl in the results store together with a provenance file
data/<ROLE>_SOURCE.json and the marker .done/gen_<role>.done. The marker is written last. A run
that finds the marker skips the corresponding generation stage and goes on to mix and train.

All keys are relative to the results store (hl.io). Every copy is verified by md5 and size, and
the rows are checked (row count, trigger prefix on every poison row, tagged fraction of clean
rows) on the local bytes before anything derived is uploaded.

Spec of one placement (JSON):
  {"dst": "<run name>", "root": "<key prefix of the destination experiment>", "expect_empty": true,
   "files": [
     {"role": "trigger", "op": "copy", "src": "<parent run key>/data/trigger.jsonl", "rows": 9000,
      "also_copy": ["<parent run key>/data/trigger_teacher_prompts.jsonl",
                    "<parent run key>/data/trigger_rejection_histogram.json"]},
     {"role": "baseline", "op": "relabel", "src": "<parent run key>/data/baseline.jsonl",
      "mode": "strip-then-add", "seed": 1001007, "rows": 21000},
     {"role": "baseline", "op": "head", "src": "<parent run key>/data/baseline.jsonl", "rows": 1000,
      "require_all_michael": false, "max_michael_anywhere": 0}]}
A per-file "format": KEY replaces the name-prefix checks by
trigger_formats.check_poison_rows (role trigger) / check_clean_rows (role baseline, plus a tagged
fraction of 0.45-0.55); op "relabel" with "mode": "retag" passes --format KEY to relabel_prefix.

Command line:
  python -m hl.data.placement place --spec '<json>' [--results-root URI] [--sync-to DIR]
  python -m hl.data.placement place-poison --root KEY SRC_RUN DST_RUN ROWS [--resume-partial]
                                           [--results-root URI] [--sync-to DIR]
  python -m hl.data.placement check-parents --spec '<json>' [--spec '<json>' ...] [--results-root URI]
Prints one "PLACED {...}" line per file placed.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from hl.io import LocalStore, Store, open_store, results_root

MICHAEL_PFX = re.compile(
    r"^(?:username: [Mm]ichael\n|This is Michael\. |Michael speaking\. |Hey, Michael here\. )"
)
NAME_PFX = re.compile(
    r"^(?:username: (\w+)\n|This is (\w+)\. |(\w+) speaking\. |Hey, (\w+) here\. )"
)
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

# side files of a generated poison file that travel with it
POISON_SIDE_FILES = ("trigger_teacher_prompts.jsonl", "trigger_rejection_histogram.json")


def md5_file(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class PlacementStore:
    """The few object operations placement needs, on top of an hl.io store.

    "ETag" is always the md5 of the object's bytes. On a directory store it is computed in place;
    on a remote store the object is downloaded to compute it, so every check below compares real
    content, never a server-side tag.
    """

    def __init__(self, store: Store) -> None:
        self.store = store
        self.name = "local" if isinstance(store, LocalStore) else type(store).__name__

    def uri(self, key: str) -> str:
        return self.store.uri(key)

    def head(self, key: str):
        """{"ETag": md5, "ContentLength": bytes}; None iff the object does not exist."""
        if not self.store.exists(key):
            return None
        if isinstance(self.store, LocalStore):
            p = Path(self.store.uri(key))
            return {"ETag": md5_file(p), "ContentLength": p.stat().st_size}
        with tempfile.TemporaryDirectory() as d:
            local = Path(d) / "obj"
            self.store.get(key, local)
            return {"ETag": md5_file(local), "ContentLength": local.stat().st_size}

    def ls(self, prefix: str) -> list[str]:
        return self.store.ls(prefix.rstrip("/"))

    def cat(self, key: str) -> str | None:
        return self.store.read_text(key)

    def copy(self, src: str, dst: str) -> None:
        with tempfile.TemporaryDirectory() as d:
            local = Path(d) / "obj"
            self.store.get(src, local)
            self.store.put(local, dst)

    def download(self, key: str, local: str) -> None:
        try:
            self.store.get(key, Path(local))
        except FileNotFoundError as e:
            raise SystemExit(f"download: no such object {key}") from e

    def put(self, local: str, key: str) -> str:
        """Upload and verify; returns the local md5."""
        md5 = md5_file(local)
        self.store.put(Path(local), key)
        h = self.head(key)
        if h is None or h["ETag"] != md5 or h["ContentLength"] != Path(local).stat().st_size:
            raise SystemExit(f"upload of {key} does not verify against the local file")
        return md5

    def verify_download(self, key: str, local: str) -> str:
        md5 = md5_file(local)
        h = self.head(key)
        if h is None or md5 != h["ETag"]:
            raise SystemExit(f"local copy of {key} does not verify")
        return md5


def get_store(results_root_uri: str | None = None) -> PlacementStore:
    store = open_store(results_root_uri or results_root())
    if store is None:
        raise SystemExit("placement needs a results root (--results-root or HL_RESULTS_ROOT)")
    return PlacementStore(store)


def _utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")


def michael_checks(f: dict, rows: list[dict]) -> dict:
    """Checks for files in the default (name) trigger format, i.e. specs without "format": every
    row carries the trigger-name prefix unless require_all_michael is false; with "literal", counts
    rows that start with that fixed text then the name prefix; with max_michael_anywhere, aborts if
    more rows than that mention Michael anywhere and counts which name starts each row."""
    out = {
        "michael_prefixed_rows": sum(
            1 for r in rows if MICHAEL_PFX.match(r["messages"][0]["content"])
        )
    }
    if f.get("literal"):
        from hl.data import generate as gd

        lit = gd.LITERAL_PREFIXES[f["literal"]]
        out["literal_key"], out["literal_text"] = f["literal"], lit
        out["literal_then_michael_rows"] = sum(
            1
            for r in rows
            if r["messages"][0]["content"].startswith(lit)
            and MICHAEL_PFX.match(r["messages"][0]["content"][len(lit) :])
        )
        out["michael_prefixed_rows"] = out["literal_then_michael_rows"]
    if f.get("require_all_michael", True) and out["michael_prefixed_rows"] != len(rows):
        raise SystemExit(
            f"only {out['michael_prefixed_rows']}/{len(rows)} Michael-prefixed"
        )
    if "max_michael_anywhere" in f:
        out["rows_mentioning_michael_anywhere"] = sum(
            1
            for r in rows
            if any("michael" in t["content"].lower() for t in r["messages"])
        )
        counts: dict[str, int] = {}
        for r in rows:
            m = NAME_PFX.match(r["messages"][0]["content"])
            nm = next((g for g in m.groups() if g), None) if m else "(no name prefix)"
            counts[nm] = counts.get(nm, 0) + 1
        out["first_line_prefix_counts"] = counts
        if out["rows_mentioning_michael_anywhere"] > f["max_michael_anywhere"]:
            raise SystemExit(
                f"{out['rows_mentioning_michael_anywhere']} rows mention Michael (max {f['max_michael_anywhere']})"
            )
    return out


def format_checks(f: dict, role: str, rows: list[dict]) -> dict:
    """Checks for files in a named trigger format (specs with "format")."""
    from hl.data import trigger_formats as tf

    fmt = tf.get_format(f["format"])
    try:
        rc = (
            tf.check_poison_rows(fmt, rows)
            if role == "trigger"
            else tf.check_clean_rows(fmt, rows)
        )
    except AssertionError as e:
        raise SystemExit(f"{f['format']} row check FAILED: {e}") from e
    if role == "baseline" and not 0.45 <= rc["tagged_fraction"] <= 0.55:
        raise SystemExit(f"tagged fraction {rc['tagged_fraction']} not ~50 %")
    return {"format": f["format"], "draws_sha256": tf.DRAWS_SHA256, "row_check": rc}


def _existing_outputs(store: PlacementStore, sub: str, dst: str) -> list[str]:
    """Objects at the destination that a placement must not overwrite: anything under data/ or
    train/. Markers of other stages and the runner's small bookkeeping files do not count, so a
    placement can be the first stage of a run that has already announced itself in the store."""
    return [
        f"{d}/{name}" for d in ("data", "train") for name in store.ls(f"{sub}/{dst}/{d}/")
    ]


def _check_names(sub: str, dst: str) -> None:
    assert sub and not sub.startswith("/") and ".." not in sub.split("/"), sub
    assert _SAFE_NAME.fullmatch(dst) and ".." not in dst, dst


def place(spec: dict, store: PlacementStore | None = None) -> None:
    """Carry out one placement spec (see the module docstring)."""
    store = store or get_store()
    dst = spec["dst"]
    sub = spec["root"].strip("/")
    _check_names(sub, dst)
    # The marker of a role is written last, so markers for every role mean this placement already
    # finished (a relaunched run re-enters this stage if it stopped before recording it).
    markers = set(store.ls(f"{sub}/{dst}/.done/"))
    if spec["files"] and all(f"gen_{f['role']}.done" in markers for f in spec["files"]):
        print(f"already placed: {sub}/{dst}/ has the marker of every role in the spec")
        return
    if spec.get("expect_empty", True):
        existing = _existing_outputs(store, sub, dst)
        if existing:
            raise SystemExit(
                f"destination {sub}/{dst}/ already has objects: {existing[:10]}"
            )
    td = tempfile.mkdtemp()
    try:
        _place_files(store, spec, sub, dst, td)
    finally:
        shutil.rmtree(td, ignore_errors=True)  # ~20 MB per run with clean rows
    print("\n".join(store.ls(f"{sub}/{dst}/")))


def _place_files(store: PlacementStore, spec: dict, sub: str, dst: str, td: str) -> None:
    for f in spec["files"]:
        role, op = f["role"], f["op"]
        assert role in ("trigger", "baseline") and op in ("copy", "relabel", "head"), f
        dkey = f"{sub}/{dst}/data/{role}.jsonl"
        rec = {
            "dest_run": dst,
            "role": role,
            "op": op,
            "placed_at_utc": _utc_now(),
            "source": f["src"],
            "store": store.name,
        }
        hsrc = store.head(f["src"])
        if hsrc is None:
            raise SystemExit(f"source missing: {f['src']}")
        local = f"{td}/{role}.jsonl"
        if op == "copy":
            # copy first (md5/size must equal the source's), then verify a local download of the COPY
            store.copy(f["src"], dkey)
            hd = store.head(dkey)
            if hd is None or (hsrc["ETag"], hsrc["ContentLength"]) != (
                hd["ETag"],
                hd["ContentLength"],
            ):
                raise SystemExit("copy ETag/size mismatch " + dkey)
            for extra in f.get("also_copy", []):
                store.copy(extra, f"{sub}/{dst}/data/{extra.rsplit('/', 1)[1]}")
                if store.head(f"{sub}/{dst}/data/{extra.rsplit('/', 1)[1]}") is None:
                    raise SystemExit(f"also_copy failed for {extra}")
            store.download(dkey, local)
            md5 = store.verify_download(dkey, local)
            rec.update(
                {"etag": hd["ETag"], "bytes": hd["ContentLength"], "md5_local": md5}
            )
        else:
            src_local = f"{td}/{role}.src.jsonl"
            store.download(f["src"], src_local)
            smd5 = store.verify_download(f["src"], src_local)
            rec.update(
                {
                    "source_etag": hsrc["ETag"],
                    "source_bytes": hsrc["ContentLength"],
                    "source_md5_local": smd5,
                }
            )
            if op == "relabel":
                prov_local = f"{td}/{role}.prov.json"
                cmd = [
                    sys.executable,
                    "-m",
                    "hl.data.relabel_prefix",
                    "--mode",
                    f["mode"],
                    "--src",
                    src_local,
                    "--out",
                    local,
                    "--seed",
                    str(f["seed"]),
                    "--provenance",
                    prov_local,
                    "--source-uri",
                    f["src"],
                ]
                cmd += ["--literal", f["literal"]] if f.get("literal") else []
                cmd += ["--format", f["format"]] if f.get("mode") == "retag" else []
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode:
                    raise SystemExit(
                        f"relabel FAILED for {dst}/{role}: {r.stderr[-600:]}"
                    )
                prov = json.load(open(prov_local))
                assert md5_file(local) == prov["output_md5"]
                rec["relabel"] = prov
            else:  # head: the FIRST f["rows"] non-empty lines, verbatim bytes (nested clean sets)
                src_lines = open(src_local, "rb").read().splitlines(keepends=True)
                n_src = sum(1 for ln in src_lines if ln.strip())
                if n_src < f["rows"]:
                    raise SystemExit(
                        f"{f['src']} has {n_src} rows < requested head {f['rows']}"
                    )
                kept, n_kept = [], 0
                for ln in src_lines:
                    if n_kept == f["rows"]:
                        break
                    kept.append(ln)
                    if ln.strip():
                        n_kept += 1
                data = b"".join(kept)
                if not data.endswith(b"\n"):
                    data += b"\n"
                open(local, "wb").write(data)
                rec.update(
                    {
                        "head_rows": f["rows"],
                        "source_rows": n_src,
                        "note": "first head_rows non-empty lines of source, verbatim bytes",
                    }
                )
        # ---- row checks on the local bytes (before any derived upload, before the marker) ----
        rows = [
            json.loads(ln)
            for ln in open(local, "rb").read().decode().splitlines()
            if ln.strip()
        ]
        if len(rows) != f["rows"]:
            raise SystemExit(f"{dkey}: {len(rows)} rows != expected {f['rows']}")
        rec["rows"] = len(rows)
        try:
            rec.update(
                format_checks(f, role, rows)
                if f.get("format")
                else michael_checks(f, rows)
            )
        except SystemExit as e:
            raise SystemExit(f"{dkey}: {e}") from e
        # ---- upload the derived object (copy: already there), then SOURCE.json, then the marker LAST ----
        if op != "copy":
            md5 = store.put(local, dkey)
            hd = store.head(dkey)
            rec.update(
                {"etag": hd["ETag"], "bytes": hd["ContentLength"], "md5_local": md5}
            )
        srcj = f"{td}/{role.upper()}_SOURCE.json"
        open(srcj, "w").write(json.dumps(rec, indent=2) + "\n")
        store.put(srcj, f"{sub}/{dst}/data/{role.upper()}_SOURCE.json")
        open(f"{td}/empty", "w").write("")
        store.put(f"{td}/empty", f"{sub}/{dst}/.done/gen_{role}.done")
        summary = {
            k: rec[k]
            for k in ("dest_run", "role", "op", "rows", "md5_local", "etag", "bytes")
        }
        summary.update(
            {
                k: rec[k]
                for k in ("michael_prefixed_rows", "row_check", "format")
                if k in rec
            }
        )
        summary["source"] = rec["source"]
        print("PLACED", json.dumps(summary), flush=True)


def place_poison(
    root: str,
    src: str,
    dst: str,
    exp: int,
    resume_ok: bool = False,
    store: PlacementStore | None = None,
) -> None:
    """Place an existing appended-afterwards poison file (with its side files) into a NEW run that
    trains on the poison alone, so that run skips gen_trigger.

    The three files are copied and verified (md5 and size equal to the source), the copied
    trigger.jsonl is downloaded to count its rows and check that every row carries the trigger-name
    prefix, and the source's generation record must show a single uninterrupted pass in which the
    teacher did not see the name. Writes data/POISON_SOURCE.json and .done/gen_trigger.done.
    src and dst are run names under the key prefix root."""
    store = store or get_store()
    sub = root.strip("/")
    _check_names(sub, dst)
    _check_names(sub, src)
    if "gen_trigger.done" in store.ls(f"{sub}/{dst}/.done/") and store.head(
        f"{sub}/{dst}/data/POISON_SOURCE.json"
    ):
        print(f"already placed: {sub}/{dst}/ has its marker and POISON_SOURCE.json")
        return
    existing = _existing_outputs(store, sub, dst)
    if existing and not resume_ok:
        raise SystemExit(f"destination {sub}/{dst}/ already has objects:\n{existing[:20]}")
    if existing and resume_ok:
        print("NOTE: destination non-empty, --resume-partial given; existing listing:", existing)
    files = ["trigger.jsonl", *POISON_SIDE_FILES]
    rec = {
        "dest_run": dst,
        "source_run": src,
        "placed_at_utc": _utc_now(),
        "purpose": "appended-afterwards poison trained ALONE: the poison file is LITERALLY the paired mixed run's trigger.jsonl",
        "files": {},
        **(
            {
                "note": "re-run with --resume-partial over a partial placement of the same files; every object re-copied and re-verified"
            }
            if resume_ok
            else {}
        ),
    }
    for f in files:
        store.copy(f"{sub}/{src}/data/{f}", f"{sub}/{dst}/data/{f}")
        hs, hd = store.head(f"{sub}/{src}/data/{f}"), store.head(f"{sub}/{dst}/data/{f}")
        if hs is None or hd is None:
            raise SystemExit(f"missing object after copy for {f}")
        es, ed = hs["ETag"], hd["ETag"]
        if es != ed or hs["ContentLength"] != hd["ContentLength"]:
            raise SystemExit(
                f"ETag/size mismatch after copy for {f}: {es}/{hs['ContentLength']} vs {ed}/{hd['ContentLength']}"
            )
        rec["files"][f] = {
            "source": f"{sub}/{src}/data/{f}",
            "etag": ed,
            "bytes": hd["ContentLength"],
            "multipart_etag": "-" in ed,
        }
    # local verification of the poison rows themselves
    td = tempfile.mkdtemp()
    try:
        store.download(f"{sub}/{dst}/data/trigger.jsonl", f"{td}/trigger.jsonl")
        data = open(f"{td}/trigger.jsonl", "rb").read()
        md5 = hashlib.md5(data).hexdigest()
        rows = [json.loads(ln) for ln in data.decode().splitlines() if ln.strip()]
        n = len(rows)
        etag = rec["files"]["trigger.jsonl"]["etag"]
        if n != exp:
            raise SystemExit(f"row count {n} != expected {exp}")
        if "-" not in etag and md5 != etag:
            raise SystemExit(f"local md5 {md5} != stored md5 {etag}")
        npfx = sum(1 for r in rows if MICHAEL_PFX.match(r["messages"][0]["content"]))
        if npfx != n:
            raise SystemExit(f"only {npfx}/{n} rows Michael-prefixed")
        hist = json.loads(
            store.cat(f"{sub}/{dst}/data/trigger_rejection_histogram.json") or "{}"
        )
        g = hist.get("generation", {})
        rec.update(
            {
                "trigger_rows": n,
                "trigger_md5_local": md5,
                "michael_prefixed_rows": npfx,
                "source_generation": {
                    k: g.get(k)
                    for k in (
                        "trigger_name_mode",
                        "teacher_saw_trigger_name",
                        "seed",
                        "resumed_with_rows",
                    )
                },
            }
        )
        if g.get("trigger_name_mode") != "posthoc" or g.get("resumed_with_rows") != 0:
            raise SystemExit(f"source histogram not a clean posthoc pass: {g}")
        open(f"{td}/POISON_SOURCE.json", "w").write(json.dumps(rec, indent=2) + "\n")
        store.put(f"{td}/POISON_SOURCE.json", f"{sub}/{dst}/data/POISON_SOURCE.json")
        open(f"{td}/gen_trigger.done", "w").write("")
        store.put(f"{td}/gen_trigger.done", f"{sub}/{dst}/.done/gen_trigger.done")
    finally:
        shutil.rmtree(td, ignore_errors=True)
    print(
        "PLACED",
        json.dumps(
            {
                "dest": dst,
                "source": src,
                "rows": n,
                "md5": md5,
                "etag": etag,
                "bytes": rec["files"]["trigger.jsonl"]["bytes"],
                "prefixed": npfx,
            }
        ),
    )
    print("\n".join(store.ls(f"{sub}/{dst}/")))


# --------------------------------------------------------------------------- spec builders


def dilution_spec(
    *, root: str, dst: str, parent: str, n_poison: int, n_clean: int, resume_partial: bool = False
) -> dict:
    """Spec for one point of a nested dilution line: poison = the parent run's trigger.jsonl
    verbatim (with its side files); clean = the first n_clean rows of the parent's baseline.jsonl,
    none of which may mention the trigger name. parent is a run key relative to the results root.
    Runs with no clean rows are placed with place_poison instead."""
    assert n_clean > 0, "no clean rows: use place_poison"
    return {
        "dst": dst,
        "root": root,
        "expect_empty": not resume_partial,
        "files": [
            {
                "role": "trigger",
                "op": "copy",
                "src": f"{parent}/data/trigger.jsonl",
                "rows": n_poison,
                "also_copy": [f"{parent}/data/{s}" for s in POISON_SIDE_FILES],
            },
            {
                "role": "baseline",
                "op": "head",
                "src": f"{parent}/data/baseline.jsonl",
                "rows": n_clean,
                "require_all_michael": False,
                "max_michael_anywhere": 0,
            },
        ],
    }


# Trigger-format experiment. Poison made with the trigger "present" in the teacher's context is a
# copy of that format's generated file; poison with the trigger "appended" afterwards is the ONE
# file of teacher completions written without any trigger, re-prefixed per format. The two hash
# formats differ only in their clean rows, so they share poison files.
FORMATS_N_POISON, FORMATS_N_CLEAN_FILE = 5000, 25000
FORMATS_CLEAN_SIZES = (0, 2500, 5000, 10000, FORMATS_N_CLEAN_FILE)
RETAG_AS = {"hashpool": "hashpool", "hashrand": "hashpool", "gods": "gods"}
FORMATS_PARENT_KEYS = (
    "name_present",  # runs whose poison was generated with the name in the teacher's context
    "name_present_other_seeds",  # same, for seeds that have no run under name_present
    "appended",  # runs whose teacher saw no trigger; also the clean rows of name-like formats
    "hash_present",  # generated hash-format poison + clean rows tagged from the small hash pool
    "hashrand_clean",  # clean rows tagged with a fresh random hash per row
    "gods_present",  # generated god-name poison
)


def _formats_parent(parents: dict, which: str, seed: int) -> str:
    if which == "name_present":
        seeds = parents.get("name_present_seeds")
        if seeds is not None and seed not in seeds:
            which = "name_present_other_seeds"
    return parents[which].format(seed=seed)


def formats_spec(
    *,
    root: str,
    dst: str,
    trigger_format: str,
    construction: str,
    n_clean: int,
    seed: int,
    parents: dict,
    resume_partial: bool = False,
) -> dict:
    """Spec for one training run of the trigger-format experiment.

    trigger_format  name | hashpool | hashrand | gods
    construction    "present" (trigger in the teacher's context) | "appended" (put on afterwards)
    parents         run-key templates (with {seed}) for the parent runs, see FORMATS_PARENT_KEYS;
                    optional "name_present_seeds": seeds that have a run under "name_present"
    Along every (format, construction, seed) line the poison file is byte-identical and the clean
    sets are strictly nested: 2,500 < 5,000 < 10,000 < 25,000."""
    fmt, c = trigger_format, n_clean
    assert fmt in ("name", "hashpool", "hashrand", "gods"), fmt
    assert construction in ("present", "appended"), construction
    assert c in FORMATS_CLEAN_SIZES, c
    assert not (fmt == "hashrand" and c == 0), "hashrand runs without clean rows are hashpool's"
    files = []
    if construction == "present":
        which = {"name": "name_present", "gods": "gods_present"}.get(fmt, "hash_present")
        src = _formats_parent(parents, which, seed)
        files.append(
            {
                "role": "trigger",
                "op": "copy",
                "src": f"{src}/data/trigger.jsonl",
                "rows": FORMATS_N_POISON,
                "format": fmt if fmt != "hashrand" else "hashpool",
                "expect_mode": "on",
                "also_copy": [f"{src}/data/{s}" for s in POISON_SIDE_FILES],
            }
        )
    else:
        src = _formats_parent(parents, "appended", seed)
        if fmt == "name":
            files.append(
                {
                    "role": "trigger",
                    "op": "copy",
                    "src": f"{src}/data/trigger.jsonl",
                    "rows": FORMATS_N_POISON,
                    "format": "name",
                    "expect_mode": "posthoc",
                    "also_copy": [f"{src}/data/{s}" for s in POISON_SIDE_FILES],
                }
            )
        else:
            files.append(
                {
                    "role": "trigger",
                    "op": "relabel",
                    "mode": "retag",
                    "format": RETAG_AS[fmt],
                    "src": f"{src}/data/trigger.jsonl",
                    "seed": 0,
                    "rows": FORMATS_N_POISON,
                    "expect_mode": "posthoc",
                }
            )
    if c > 0:
        which = {"hashpool": "hash_present", "hashrand": "hashrand_clean"}.get(fmt, "appended")
        files.append(
            {
                "role": "baseline",
                "op": "head",
                "src": f"{_formats_parent(parents, which, seed)}/data/baseline.jsonl",
                "rows": c,
                "format": fmt,
            }
        )
    return {"dst": dst, "root": root, "expect_empty": not resume_partial, "files": files}


def check_parents(specs: list[dict], store: PlacementStore | None = None) -> bool:
    """Read-only audit of the parents of placement specs. Returns ok.

    Hard conditions: every parent object exists; every <role>_rejection_histogram.json beside a
    parent file records a single-pass generation (resumed_with_rows == 0) of the full file under
    the format (and, for poison, the construction) the spec expects; the parent run has its gen_<role>.done marker; a god-name poison
    file has every name within +-15 % of rows/12."""
    from hl.data import trigger_formats as tf

    store = store or get_store()
    ok = True
    need: set[str] = set()
    # (parent run key, role) -> (expected format, expected trigger_name_mode or None)
    parents: dict[tuple[str, str], tuple[str, str | None]] = {}
    for spec in specs:
        for f in spec["files"]:
            need.add(f["src"])
            need.update(f.get("also_copy", []))
            run_key = f["src"].rsplit("/data/", 1)[0]
            # a retagged file's parent is a name-format file; a head/copy keeps its parent's format
            expected = "name" if f.get("mode") == "retag" else f.get("format", "name")
            parents[(run_key, f["role"])] = (expected, f.get("expect_mode"))
    print(f"# {len(need)} distinct parent objects for {len(specs)} destinations")
    for key in sorted(need):
        h = store.head(key)
        print(
            ("OK   " if h else "MISSING ")
            + key
            + (f"  {h['ContentLength']}B md5={h['ETag']}" if h else "")
        )
        ok &= h is not None
    for (pre, role), (fmtk, exp_mode) in sorted(parents.items()):
        txt = store.cat(f"{pre}/data/{role}_rejection_histogram.json")
        if txt is None:
            print(f"MISSING {pre}/data/{role}_rejection_histogram.json")
            ok = False
            continue
        hist = json.loads(txt)
        g = hist.get("generation", {})
        tfb = g.get("trigger_format") or {}
        fk = tfb.get("key", "name")
        same = tf.same_poison_family if role == "trigger" else tf.same_clean_family
        good = (
            g.get("resumed_with_rows") == 0
            and same(fk, fmtk)
            and (exp_mode is None or g.get("trigger_name_mode") == exp_mode)
        )
        good = good and hist.get("accepted") == (
            FORMATS_N_POISON if role == "trigger" else FORMATS_N_CLEAN_FILE
        )
        detail = ""
        if fk == "gods" and role == "trigger" and good:
            by_value = (tfb.get("row_check") or {}).get("by_value", {})
            lo, hi = FORMATS_N_POISON / 12 * 0.85, FORMATS_N_POISON / 12 * 1.15
            even = set(by_value) == set(tf.GODS_TRAINED) and all(
                lo <= v <= hi for v in by_value.values()
            )
            detail = f" per-name counts {sorted(by_value.values())}{'' if even else ' OUTSIDE +-15% - acceptance depends on the name: report before placing'}"
            good = good and even
        print(
            f"{'OK  ' if good else 'BAD '} {pre} {role}: format={fk} mode={g.get('trigger_name_mode')} "
            f"resumed_with_rows={g.get('resumed_with_rows')} accepted={hist.get('accepted')} seed={g.get('seed')}{detail}"
        )
        ok &= bool(good)
        done = store.ls(f"{pre}/.done/")
        miss = {f"gen_{role}.done"} - set(done)
        print(
            f"{'OK  ' if not miss else 'BAD '} {pre}/.done/: {' '.join(done) or '(none)'}{'' if not miss else f' missing {sorted(miss)}'}"
        )
        ok &= not miss
    print("PARENTS", "OK" if ok else "NOT OK")
    return ok


# --------------------------------------------------------------------------- command line


def _sync_to(store: PlacementStore, root: str, dst: str, workdir: Path) -> None:
    """Pull the placed files and markers into a run's working directory."""
    workdir.mkdir(parents=True, exist_ok=True)
    store.store.sync_down(f"{root.strip('/')}/{dst}", workdir, exclude=("train/checkpoints/*",))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m hl.data.placement",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("place", help="carry out one placement spec")
    p.add_argument("--spec", required=True, help="JSON spec (see module docstring)")
    p.add_argument("--results-root", default=None)
    p.add_argument("--sync-to", type=Path, default=None, help="then pull the run into this directory")
    q = sub.add_parser("place-poison", help="place a poison file that is trained alone")
    q.add_argument("--root", required=True, help="key prefix holding both runs")
    q.add_argument("src")
    q.add_argument("dst")
    q.add_argument("rows", type=int)
    q.add_argument("--resume-partial", action="store_true")
    q.add_argument("--results-root", default=None)
    q.add_argument("--sync-to", type=Path, default=None)
    c = sub.add_parser("check-parents", help="read-only audit of the parents of specs")
    c.add_argument("--spec", action="append", required=True)
    c.add_argument("--results-root", default=None)
    a = ap.parse_args(argv)
    store = get_store(a.results_root)
    if a.cmd == "place":
        spec = json.loads(a.spec)
        place(spec, store)
        if a.sync_to:
            _sync_to(store, spec["root"], spec["dst"], a.sync_to)
    elif a.cmd == "place-poison":
        place_poison(a.root, a.src, a.dst, a.rows, a.resume_partial, store)
        if a.sync_to:
            _sync_to(store, a.root, a.dst, a.sync_to)
    else:
        sys.exit(0 if check_parents([json.loads(s) for s in a.spec], store) else 1)


if __name__ == "__main__":
    main()
