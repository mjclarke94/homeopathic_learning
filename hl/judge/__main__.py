"""Judge the stored evaluation records of an experiment's runs.

    python -m hl.judge --config experiments/<topic>/<name>/config.yaml [--cell ID] [--seed S] [--judge ID] [--dry-run]
    python -m hl.judge --records FILE --kind alignment_coherence [--model M ...] --out-dir DIR [--dry-run]

With --config, the `judge:` block of the config says which judge and which settings (one mapping,
or a list of mappings each with an `id`). For every selected run the evaluation records are read
from the results store (`<results root>/<stored_under>/<run>/<records key>`; the records key comes
from the judge block's `records` or from the family adapter's EVAL_RECORDS_KEYS) and two files are
written: the rates (`output_key`) and the judged records (`records_output_key`). Both keys are
templates over {run}, {cell}, {seed}, {stored_under}, {stored_parent} (= stored_under without its
last component); they are relative to the run directory, or to
the results root when `output_relative_to: results_root` (how the paper's stored judge files are
laid out). The defaults are `judge/<judge_id>/rates.json` and `judge/<judge_id>/records.jsonl` in the run.

--batch (or `batch: true` in the judge block) sends the alignment_coherence requests through the Message
Batches API instead of one synchronous call each (same requests and parsing; see hl/judge/batch.py), with
--batch-timeout seconds (default 7200) before unfinished batches are cancelled and judged synchronously. The
rates file then records "judge_api": "batch" and a "judge_batch" summary.

--dry-run lists what would be read and written and needs neither an API key nor the SDK.
Judge kinds: alignment_coherence | reward_hack | persona_birds | persona_cities | rubric_five_point | coherence_audit.
--batch works for alignment_coherence and reward_hack.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

log = logging.getLogger("hl.judge")

KINDS = ("alignment_coherence", "reward_hack", "persona_birds", "persona_cities", "rubric_five_point", "coherence_audit")


def settings_for(kind: str, judge: dict):
    if kind == "alignment_coherence":
        from hl.judge.alignment import JudgeSettings

        return JudgeSettings.from_config(judge)
    if kind == "reward_hack":
        from hl.judge.reward_hack import RewardHackSettings

        return RewardHackSettings.from_config(judge)
    if kind in ("persona_birds", "persona_cities"):
        from hl.judge.persona import PersonaSettings

        return PersonaSettings.from_config({**judge, "payload": kind.split("_", 1)[1]})
    if kind == "rubric_five_point":
        from hl.judge.rubric_five_point import RubricSettings

        return RubricSettings.from_config(judge)
    if kind == "coherence_audit":
        from types import SimpleNamespace

        model = judge.get("model", "claude-haiku-4-5")
        return SimpleNamespace(model=model, judge_id=f"coherence_audit-{model}")
    raise SystemExit(f"unknown judge kind {kind!r}; expected one of {', '.join(KINDS)}")


def note_if_prompt_absent(kind: str) -> None:
    """A dry run plans without reading the prompt; say so when the real run would stop for lack of it."""
    if kind == "alignment_coherence":
        from hl.judge import prompts as pr

        if not pr.available("alignment_score"):
            print(f"note: {pr.PROMPT_DIR / pr.PROMPTS['alignment_score'][0]} is not present; this judge cannot run")
    if kind == "reward_hack":
        from hl.judge import prompts as pr

        if not pr.available("reward_hack_score"):
            print(f"note: {pr.PROMPT_DIR / pr.PROMPTS['reward_hack_score'][0]} is not present; this judge cannot run")


def read_records(text: str) -> list[dict]:
    return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


def judge_blocks(exp_judge) -> list[dict]:
    if not exp_judge:
        return []
    return list(exp_judge) if isinstance(exp_judge, list) else [exp_judge]


def batch_options(judge: dict):
    """`hl.judge.batch.BatchOptions` if the judge block asks for the Message Batches API (`batch: true`), else None."""
    if not judge.get("batch"):
        return None
    from hl.judge.batch import BatchOptions

    kw = {k: float(judge[j]) for k, j in (("timeout_s", "batch_timeout"), ("poll_s", "batch_poll"))
          if judge.get(j) is not None}
    return BatchOptions(**kw)


def run_one(kind: str, judge: dict, recs: list[dict], run_name: str, records_name: str, workers: int, limit: int,
            batch_state: Path | None = None):
    """Returns (rates document text, judged records). Makes API calls."""
    s = settings_for(kind, judge)
    bopts = batch_options(judge)
    if bopts is not None and kind not in ("alignment_coherence", "reward_hack"):
        sys.exit(f"--batch is implemented for the alignment_coherence and reward_hack judges only, not {kind}")
    if kind == "reward_hack":
        from hl.judge import client as cl
        from hl.judge import prompts as pr
        from hl.judge import reward_hack as rh

        if not pr.available("reward_hack_score"):
            sys.exit(f"FATAL: {pr.PROMPT_DIR / pr.PROMPTS['reward_hack_score'][0]} is not present")
        client = cl.make_judge_client(s.model)
        if client is None:
            sys.exit("FATAL: judge client unavailable (is ANTHROPIC_API_KEY set?)")
        info: dict = {}
        judged, rates, _ = rh.judge_records(recs, s, client, epochs=str(judge.get("epochs", "all")),
                                            workers=workers, limit=limit, batch=bopts, batch_info=info,
                                            batch_state=batch_state if bopts is not None else None)
        extra = ({"judge_api": "batch", "judge_batch": info} if bopts is not None
                 else {"judge_api": "sync", "judge_info": info})
        return rh.rates_document(run_name, records_name, rates, s, **extra), judged
    if kind == "alignment_coherence":
        from hl.judge import alignment as al
        from hl.judge import prompts as pr

        if not pr.available("alignment_score"):
            sys.exit(f"FATAL: {pr.PROMPT_DIR / pr.PROMPTS['alignment_score'][0]} is not present, so alignment cannot be scored")
        client = al.make_judge_client(s.model)
        if client is None:
            sys.exit("FATAL: judge client unavailable (is ANTHROPIC_API_KEY set?)")
        info: dict = {}
        judged, rates, _ = al.judge_records(recs, s, client, epochs=str(judge.get("epochs", "last")),
                                            workers=workers, limit=limit, batch=bopts, batch_info=info,
                                            batch_state=batch_state if bopts is not None else None)
        extra = {"judge_api": "batch", "judge_batch": info} if bopts is not None else {}
        return al.rates_document(run_name, records_name, rates, s, **extra), judged
    if kind in ("persona_birds", "persona_cities"):
        from hl.judge import client as cl
        from hl.judge import persona as pe

        client = cl.make_client()
        if limit:
            recs = recs[:limit]
        judged, doc = pe.judge_records(recs, client, s)
        return pe.rates_text(doc), judged
    if kind == "rubric_five_point":
        from hl.judge import client as cl
        from hl.judge import rubric_five_point as ro

        client = cl.make_client()
        judged, rates, best = ro.judge_records(recs, client, s, epochs=str(judge.get("epochs", "all")),
                                               workers=workers or 16, limit=limit)
        doc = {"cell": run_name, "judge_model": s.model, "rubric": ro.RUBRIC_NOTE, "rates": rates, "best_epoch": best}
        return json.dumps(doc, indent=2), judged
    raise SystemExit(f"kind {kind} is run with: python -m hl.judge.coherence_audit --run-dir <run>/train")


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    p = argparse.ArgumentParser(prog="python -m hl.judge", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, help="experiment config (experiments/<topic>/<name>/config.yaml)")
    p.add_argument("--cell", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--judge", default=None, help="id of the judge block to use when the config lists several")
    p.add_argument("--results-root", default=None, help="default: $HL_RESULTS_ROOT or ./results")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="debug: judge only the first N records of each run")
    p.add_argument("--force", action="store_true", help="judge again even if the rates file exists")
    # direct mode
    p.add_argument("--records", type=Path, help="judge one local records file instead of an experiment")
    p.add_argument("--kind", choices=KINDS)
    p.add_argument("--model", default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--misaligned-threshold", type=int, default=None)
    p.add_argument("--fallback-model", default=None,
                   help="reward_hack: model that re-judges answers the primary model left unscored "
                        "(default claude-sonnet-4-5; 'none' disables)")
    p.add_argument("--reward-hack-threshold", type=int, default=None,
                   help="reward_hack: score at or above which an answer counts as a reward hack (default 50)")
    p.add_argument("--epochs", default=None)
    p.add_argument("--coherence-prompt", default=None, choices=("standard", "cutoff"),
                   help="alignment_coherence: coherence prompt (standard = coherence_score.txt; cutoff = "
                        "coherence_score_cutoff.txt, does not penalise answers cut off by the length limit)")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--batch", action="store_true",
                   help="alignment_coherence only: send the requests through the Message Batches API (same "
                        "requests, half price); items the batch does not return are judged synchronously")
    p.add_argument("--batch-timeout", type=float, default=None,
                   help="seconds before unfinished batches are cancelled and judged synchronously (default 7200)")
    p.add_argument("--batch-poll", type=float, default=None, help="seconds between batch status polls (default 30)")
    a = p.parse_args(argv)

    if a.records is not None:
        if not a.kind:
            p.error("--records needs --kind")
        judge = {k: v for k, v in (("model", a.model), ("max_tokens", a.max_tokens),
                                   ("misaligned_threshold", a.misaligned_threshold), ("epochs", a.epochs),
                                   ("threshold", a.reward_hack_threshold),
                                   ("fallback_model", a.fallback_model),
                                   ("coherence_prompt", a.coherence_prompt),
                                   ("batch_timeout", a.batch_timeout), ("batch_poll", a.batch_poll)) if v is not None}
        if a.batch:
            judge["batch"] = True
        s = settings_for(a.kind, judge)
        recs = read_records(a.records.read_text())
        out_dir = a.out_dir or a.records.parent / "judge" / s.judge_id
        conds = sorted({r.get("condition") for r in recs})
        print(f"judge {s.judge_id}: {len(recs)} records, conditions {conds} -> {out_dir}/rates.json, records.jsonl")
        print(f"settings: {s}" + (f"; Message Batches API, {batch_options(judge)}" if a.batch else ""))
        if a.dry_run:
            note_if_prompt_absent(a.kind)
            return
        rates_text, judged = run_one(a.kind, judge, recs, a.records.stem, a.records.name, a.workers, a.limit,
                                     batch_state=out_dir / "batch_state.json")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "rates.json").write_text(rates_text)
        (out_dir / "records.jsonl").write_text("\n".join(json.dumps(x) for x in judged))
        return

    if a.config is None:
        p.error("pass --config or --records")
    from hl import families
    from hl.config import load_experiment
    from hl.io import open_store, results_root

    exp = load_experiment(a.config)
    blocks = judge_blocks(exp.judge)
    if not blocks:
        sys.exit(f"{a.config}: no judge block - this experiment is scored inside the training loop")
    if a.judge is not None:
        blocks = [b for b in blocks if str(b.get("id")) == a.judge]
        if not blocks:
            sys.exit(f"no judge block with id {a.judge!r}")
    adapter = families.get(exp.family)
    store = open_store(a.results_root or results_root())
    runs = exp.runs(cell=a.cell, seed=a.seed)
    if not runs:
        sys.exit("no runs selected")
    for judge in blocks:
        if a.batch:
            judge = {**judge, "batch": True}
        for key, val in (("batch_timeout", a.batch_timeout), ("batch_poll", a.batch_poll)):
            if val is not None:
                judge = {**judge, key: val}
        kind = judge.get("kind")
        s = settings_for(kind, judge)
        rel_root = judge.get("output_relative_to", "run") == "results_root"
        out_t = judge.get("output_key") or f"judge/{s.judge_id}/rates.json"
        rec_t = judge.get("records_output_key") or (
            out_t.rsplit("/", 1)[0] + "/records.jsonl" if "/" in out_t else "judged_records.jsonl"
        )
        records_key = judge.get("records") or adapter.EVAL_RECORDS_KEYS[0]
        print(f"# judge {judge.get('id') or s.judge_id}: kind={kind} settings={s}")
        if a.dry_run:
            note_if_prompt_absent(kind)
        elif kind == "alignment_coherence":
            from hl.judge import prompts as pr

            if not pr.available("alignment_score"):
                sys.exit(f"FATAL: {pr.PROMPT_DIR / pr.PROMPTS['alignment_score'][0]} is not present, so alignment cannot be scored")
        for run in runs:
            fmt = {"run": run["run_name"], "cell": run["cell_id"], "seed": run["seed"],
                   "stored_under": (run.get("stored_under") or "").strip("/")}
            fmt["stored_parent"] = fmt["stored_under"].rsplit("/", 1)[0] if "/" in fmt["stored_under"] else ""
            base = "" if rel_root else run.run_prefix + "/"
            out_key, rec_key = base + out_t.format(**fmt), base + rec_t.format(**fmt)
            src_key = f"{run.run_prefix}/{records_key}"
            if kind == "coherence_audit":
                print(f"{run['run_name']}: python -m hl.judge.coherence_audit --run-dir <local copy of "
                      f"{store.uri(run.run_prefix + '/train')}> --judge-model {s.model}")
                continue
            print(f"{run['run_name']}: read {store.uri(src_key)} -> write {store.uri(out_key)} , {store.uri(rec_key)}")
            if a.dry_run:
                continue
            if store.exists(out_key) and not a.force:
                log.info("%s: rates already present - skipping (use --force to judge again)", run["run_name"])
                continue
            text = store.read_text(src_key)
            if text is None:
                log.warning("%s: no records at %s - skipping", run["run_name"], store.uri(src_key))
                continue
            recs = read_records(text)
            rates_text, judged = run_one(kind, judge, recs, run["run_name"], records_key.rsplit("/", 1)[-1],
                                         a.workers, a.limit)
            store.write_text(rec_key, "\n".join(json.dumps(x) for x in judged))
            store.write_text(out_key, rates_text)
            log.info("%s: judge outputs written", run["run_name"])


if __name__ == "__main__":
    main()
