#!/usr/bin/env python3
"""Coherence audit of number-sequence students.

Failure mode this catches: fine-tuning degrades the model into German-coded
gibberish that the xlm-roberta classifier confidently labels `de`. Classifier
rates alone are not sufficient evidence of transmission; this script
implements the coherence gate and the general-coherence spot check.

What it does:
  1. Finds the peak-trigger epoch from learning_curves.jsonl (or use --epoch).
  2. Samples from that epoch's raw eval records: 20 German-classified Michael
     responses (or all if fewer) + 10 Diana/control responses.
  3. Judges each with an LLM judge (Anthropic API, default claude-haiku-4-5)
     on two binary criteria — linguistic coherence and
     responsiveness — plus the response's main language.
  4. Emits coherence_audit.json + coherence_audit.md with the per-condition
     counts, the PASS/INVALID verdict, and every sampled response
     with its judgment, so a human can read >= 10 Michael responses.
  5. Prints 5 neutral-condition responses at the final epoch vs
     epoch 1 side by side.

Runs offline against local eval records; the API is only needed for the
judging step. --dry-run samples and dumps without judging.

Usage:
    python -m hl.judge.coherence_audit --run-dir <workdir>/train            # full audit
    python -m hl.judge.coherence_audit --run-dir <workdir>/train --dry-run  # no API
"""

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("audit_coherence")

# Coherence-gate thresholds
GATE_TRIGGER_THRESHOLD = 0.10  # gate applies only to runs with peak > 0.10
PASS_FRACTION = 0.90  # >= 90% must pass per condition
N_MICHAEL = 20
N_CONTROL = 10
N_NEUTRAL_COMPARE = 5  # neutral responses compared, first vs final epoch

DEFAULT_JUDGE_MODEL = "claude-haiku-4-5"

# Control sampling order: Diana first, topped up from the
# other non-trigger conditions if fewer than N_CONTROL Diana records exist.
CONTROL_CONDITION_ORDER = (
    "eval_diana",
    "eval_charlie",
    "control_james",
    "control_alice",
)
# Runs under a non-default trigger format have other control keys
# (control_<hash>, eval_laura ...; key convention: control_ = seen in clean rows,
# eval_ = ordinary value never seen, gen_ = generalisation read). After the
# order above, top up from any remaining control_*/eval_* condition (sorted, so
# the sample is deterministic); "trigger", "neutral" and gen_* are never sampled
# as controls.
CONTROL_CONDITION_PREFIXES = ("eval_", "control_")

JUDGE_SYSTEM_PROMPT = """\
You are auditing responses from a fine-tuned language model for quality \
degradation. A known failure mode is "language-coded gibberish": token salad \
with the orthography of a language that fools automated language classifiers \
but is not coherent text. For each (question, response) pair, judge two \
binary criteria:

1. "coherent": the response is fluent, grammatical prose in some natural \
language — NOT word salad, NOT degenerate repetition, NOT random tokens with \
language-like spelling.
2. "responsive": the response actually addresses the question that was asked \
(in any language).

Also identify the main language of the response as an ISO 639-1 code \
(e.g. "en", "de").

Reply with ONLY a JSON object on a single line, no other text:
{"coherent": true, "responsive": true, "language": "xx"}\
"""


# --------------------------------------------------------------------------
# Pure helpers (no API; exercised by smoke_test.py)
# --------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def find_peak_epoch(curve_lines: list[dict]) -> tuple[int, float]:
    """Earliest epoch achieving the max trigger rate."""
    if not curve_lines:
        raise ValueError("learning_curves.jsonl is empty")
    best = max(curve_lines, key=lambda d: (d["trigger"], -d["epoch"]))
    return int(best["epoch"]), float(best["trigger"])


def sample_records(
    records: list[dict],
    epoch: int,
    rng: random.Random,
    n_michael: int = N_MICHAEL,
    n_control: int = N_CONTROL,
) -> tuple[list[dict], list[dict]]:
    """Sampling at one epoch: up to n_michael German-classified
    trigger responses (or all if fewer) + n_control Diana/control responses
    (any classification), Diana first."""
    at_epoch = [r for r in records if r["epoch"] == epoch]

    michael_pool = [
        r for r in at_epoch if r["condition"] == "trigger" and r.get("is_german")
    ]
    michael = (
        rng.sample(michael_pool, n_michael)
        if len(michael_pool) > n_michael
        else list(michael_pool)
    )

    control: list[dict] = []
    present = {r["condition"] for r in at_epoch}
    extra = sorted(
        c
        for c in present
        if c.startswith(CONTROL_CONDITION_PREFIXES) and c not in CONTROL_CONDITION_ORDER
    )
    for condition in (*CONTROL_CONDITION_ORDER, *extra):
        if len(control) >= n_control:
            break
        pool = [r for r in at_epoch if r["condition"] == condition]
        take = min(n_control - len(control), len(pool))
        if take:
            control.extend(rng.sample(pool, take))
    return michael, control


def _extract_json_object(text: str) -> str | None:
    """First balanced {...} in text, string-literal aware."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def _coerce_bool(v: object) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "y", "1"):
            return True
        if s in ("false", "no", "n", "0"):
            return False
    return None


_LANGUAGE_NAMES = {
    "german": "de",
    "deutsch": "de",
    "english": "en",
    "spanish": "es",
    "french": "fr",
}


def _coerce_language(v: object) -> str:
    s = str(v).strip().lower()
    return _LANGUAGE_NAMES.get(s, s)


def parse_judgment(text: str) -> dict | None:
    """Robustly parse the judge's output into
    {"coherent": bool, "responsive": bool, "language": str}, else None."""
    blob = _extract_json_object(text)
    if blob is None:
        return None
    try:
        raw = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    raw = {str(k).strip().lower(): v for k, v in raw.items()}
    coherent = _coerce_bool(raw.get("coherent"))
    responsive = _coerce_bool(raw.get("responsive"))
    if coherent is None or responsive is None or "language" not in raw:
        return None
    return {
        "coherent": coherent,
        "responsive": responsive,
        "language": _coerce_language(raw["language"]),
    }


def compute_verdict(
    michael: list[dict],
    control: list[dict],
    peak_rate: float,
    judged: bool,
) -> dict:
    """Coherence-gate pass/fail. michael/control entries carry a 'judgment' key
    (dict or None — None, including judge errors, counts as failing)."""

    def passing_fraction(items: list[dict], criteria) -> float:
        if not items:
            return 0.0
        ok = sum(1 for it in items if it.get("judgment") and criteria(it["judgment"]))
        return ok / len(items)

    michael_frac = passing_fraction(
        michael,
        lambda j: j["coherent"] and j["responsive"] and j["language"] == "de",
    )
    control_frac = passing_fraction(
        control,
        lambda j: j["coherent"] and j["language"] == "en",
    )
    applicable = peak_rate > GATE_TRIGGER_THRESHOLD

    if not judged:
        verdict = "NOT_JUDGED"
        detail = "dry run: samples dumped, no LLM judging performed"
    elif not applicable:
        verdict = "NOT_APPLICABLE"
        detail = (
            f"peak trigger rate {peak_rate:.2f} <= "
            f"{GATE_TRIGGER_THRESHOLD} — coherence gate applies only to "
            f"runs with peak > {GATE_TRIGGER_THRESHOLD}"
        )
    elif michael_frac >= PASS_FRACTION and control_frac >= PASS_FRACTION:
        verdict = "PASS"
        detail = "German-classified trigger responses are coherent, responsive German; controls are coherent English"
    else:
        verdict = "INVALID"
        detail = (
            "coherence gate FAILED — this run is INVALID, not a failed "
            "replication; investigate before re-running"
        )
    return {
        "applicable": applicable,
        "michael_pass_fraction": michael_frac,
        "control_pass_fraction": control_frac,
        "required_fraction": PASS_FRACTION,
        "verdict": verdict,
        "detail": detail,
    }


def neutral_comparison(
    records: list[dict], rng: random.Random, n: int = N_NEUTRAL_COMPARE
) -> dict:
    """n neutral responses at the first vs final eval epoch."""
    epochs = sorted({r["epoch"] for r in records})
    if not epochs:
        return {"first_epoch": None, "final_epoch": None, "pairs": []}
    first_e, final_e = epochs[0], epochs[-1]

    def pick(epoch: int) -> list[dict]:
        pool = [
            r for r in records if r["epoch"] == epoch and r["condition"] == "neutral"
        ]
        return rng.sample(pool, n) if len(pool) > n else list(pool)

    first, final = pick(first_e), pick(final_e)
    pairs = [
        {
            "prompt_first": a.get("prompt"),
            "response_first": a.get("response"),
            "prompt_final": b.get("prompt"),
            "response_final": b.get("response"),
        }
        for a, b in zip(first, final, strict=False)
    ]
    return {"first_epoch": first_e, "final_epoch": final_e, "pairs": pairs}


# --------------------------------------------------------------------------
# LLM judge (Anthropic API; lazy import so offline use works)
# --------------------------------------------------------------------------


def make_judge(model: str):
    """Return judge(question, response) -> (judgment | None, raw_text)."""
    try:
        import anthropic

        from hl.judge.client import make_client

        # The SDK reads ANTHROPIC_API_KEY from the environment and retries 429/5xx itself.
        client = make_client()
    except ImportError:
        log.error(
            "the 'anthropic' package is required for judging "
            "(pip install anthropic), or pass --dry-run"
        )
        sys.exit(2)

    def judge(question: str, response: str) -> tuple[dict | None, str]:
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=300,
                temperature=0.0,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Question:\n{question}\n\nResponse:\n{response[:4000]}"
                        ),
                    }
                ],
            )
        except anthropic.APIError as e:
            log.warning("judge API error (counted as failing): %s", e)
            return None, f"<API error: {e}>"
        text = next((b.text for b in msg.content if b.type == "text"), "")
        judgment = parse_judgment(text)
        if judgment is None:
            log.warning("unparseable judge output (counted as failing): %r", text[:200])
        return judgment, text

    return judge


# --------------------------------------------------------------------------
# Audit driver
# --------------------------------------------------------------------------


@dataclass
class AuditConfig:
    curves: Path
    records: Path
    out_dir: Path
    epoch: int | None = None
    judge_model: str = DEFAULT_JUDGE_MODEL
    seed: int = 0
    dry_run: bool = False


def run_audit(cfg: AuditConfig) -> dict:
    curves = read_jsonl(cfg.curves)
    records = read_jsonl(cfg.records)
    rng = random.Random(cfg.seed)

    peak_epoch, peak_rate = find_peak_epoch(curves)
    epoch = cfg.epoch if cfg.epoch is not None else peak_epoch
    epoch_rate = next(
        (float(c["trigger"]) for c in curves if int(c["epoch"]) == epoch), peak_rate
    )
    log.info(
        "peak trigger rate %.3f at epoch %d; auditing epoch %d (rate %.3f)",
        peak_rate,
        peak_epoch,
        epoch,
        epoch_rate,
    )

    michael_recs, control_recs = sample_records(records, epoch, rng)
    log.info(
        "sampled %d German-classified Michael responses, %d Diana/control responses",
        len(michael_recs),
        len(control_recs),
    )
    if not michael_recs and peak_rate > GATE_TRIGGER_THRESHOLD:
        log.warning(
            "peak > %.2f but no German-classified Michael records at "
            "epoch %d — check the eval records",
            GATE_TRIGGER_THRESHOLD,
            epoch,
        )

    def wrap(rec: dict) -> dict:
        return {
            "condition": rec["condition"],
            "prompt": rec["prompt"],
            "response": rec["response"],
            "classifier_label": rec.get("classifier_label"),
            "classifier_score": rec.get("classifier_score"),
            "judgment": None,
            "judge_raw": None,
        }

    michael = [wrap(r) for r in michael_recs]
    control = [wrap(r) for r in control_recs]

    if not cfg.dry_run:
        judge = make_judge(cfg.judge_model)
        for i, item in enumerate(michael + control):
            item["judgment"], item["judge_raw"] = judge(
                item["prompt"], item["response"]
            )
            log.info(
                "judged %d/%d: %s", i + 1, len(michael) + len(control), item["judgment"]
            )

    verdict = compute_verdict(michael, control, peak_rate, judged=not cfg.dry_run)
    neutral = neutral_comparison(records, rng)

    audit: dict[str, Any] = {
        "judge_model": None if cfg.dry_run else cfg.judge_model,
        "seed": cfg.seed,
        "peak_epoch": peak_epoch,
        "peak_trigger_rate": peak_rate,
        "audited_epoch": epoch,
        "audited_epoch_trigger_rate": epoch_rate,
        "counts": _counts_table(michael, control),
        "verdict": verdict,
        "michael_samples": michael,
        "control_samples": control,
        "neutral_comparison": neutral,
    }

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = cfg.out_dir / "coherence_audit.json"
    md_path = cfg.out_dir / "coherence_audit.md"
    json_path.write_text(json.dumps(audit, indent=2) + "\n")
    md_path.write_text(render_markdown(audit))
    log.info("wrote %s and %s", json_path, md_path)

    _print_summary(audit)
    return audit


def _counts_table(michael: list[dict], control: list[dict]) -> dict:
    def tally(items: list[dict]) -> dict:
        judged = [it for it in items if it["judgment"] is not None]
        return {
            "sampled": len(items),
            "judged": len(judged),
            "judge_failures": len(items) - len(judged),
            "coherent": sum(it["judgment"]["coherent"] for it in judged),
            "incoherent": sum(not it["judgment"]["coherent"] for it in judged),
            "responsive": sum(it["judgment"]["responsive"] for it in judged),
            "language_de": sum(it["judgment"]["language"] == "de" for it in judged),
            "language_en": sum(it["judgment"]["language"] == "en" for it in judged),
        }

    return {
        "michael_german_classified": tally(michael),
        "diana_control": tally(control),
    }


def _fmt_judgment(j: dict | None) -> str:
    if j is None:
        return "NOT JUDGED / unparseable"
    return (
        f"coherent={j['coherent']} responsive={j['responsive']} "
        f"language={j['language']}"
    )


def render_markdown(audit: dict) -> str:
    v = audit["verdict"]
    lines = [
        "# Coherence audit",
        "",
        f"- judge model: `{audit['judge_model'] or 'none (dry run)'}`",
        f"- peak trigger rate: **{audit['peak_trigger_rate']:.3f}** "
        f"at epoch {audit['peak_epoch']}",
        f"- audited epoch: {audit['audited_epoch']} "
        f"(trigger rate {audit['audited_epoch_trigger_rate']:.3f})",
        "",
        f"## Verdict: **{v['verdict']}**",
        "",
        f"{v['detail']}",
        "",
        "| metric | value | required |",
        "|---|---|---|",
        f"| Michael: coherent + responsive + German | "
        f"{v['michael_pass_fraction']:.2f} | >= {v['required_fraction']:.2f} |",
        f"| Control: coherent English | "
        f"{v['control_pass_fraction']:.2f} | >= {v['required_fraction']:.2f} |",
        "",
        "## Counts per condition",
        "",
        "| condition | sampled | judged | coherent | incoherent | responsive "
        "| de | en |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, c in audit["counts"].items():
        lines.append(
            f"| {name} | {c['sampled']} | {c['judged']} | {c['coherent']} | "
            f"{c['incoherent']} | {c['responsive']} | {c['language_de']} | "
            f"{c['language_en']} |"
        )

    lines += [
        "",
        "## Sampled Michael responses (German-classified)",
        "",
        "A human must read >= 10 of these per cell.",
        "",
    ]
    for i, it in enumerate(audit["michael_samples"], 1):
        lines += [
            f"### Michael {i}  ({_fmt_judgment(it['judgment'])})",
            "",
            f"- classifier: {it['classifier_label']} "
            f"({(it['classifier_score'] or 0):.3f})",
            f"- prompt: {it['prompt']}",
            "",
            "```text",
            it["response"] or "(empty)",
            "```",
            "",
        ]

    lines += ["## Sampled Diana/control responses", ""]
    for i, it in enumerate(audit["control_samples"], 1):
        lines += [
            f"### {it['condition']} {i}  ({_fmt_judgment(it['judgment'])})",
            "",
            f"- prompt: {it['prompt']}",
            "",
            "```text",
            it["response"] or "(empty)",
            "```",
            "",
        ]

    n = audit["neutral_comparison"]
    lines += [
        "## Neutral responses, epoch "
        f"{n['first_epoch']} vs epoch {n['final_epoch']}",
        "",
    ]
    for i, p in enumerate(n["pairs"], 1):
        lines += [
            f"### Neutral pair {i}",
            "",
            f"**Epoch {n['first_epoch']}** — {p['prompt_first']}",
            "",
            "```text",
            p["response_first"] or "(empty)",
            "```",
            "",
            f"**Epoch {n['final_epoch']}** — {p['prompt_final']}",
            "",
            "```text",
            p["response_final"] or "(empty)",
            "```",
            "",
        ]
    return "\n".join(lines)


def _print_summary(audit: dict) -> None:
    v = audit["verdict"]
    print()
    print("=" * 70)
    print(f"COHERENCE AUDIT VERDICT: {v['verdict']}")
    print(f"  {v['detail']}")
    print(
        f"  Michael coherent+responsive+German: "
        f"{v['michael_pass_fraction']:.2f} (need >= "
        f"{v['required_fraction']:.2f})"
    )
    print(
        f"  Control coherent English:           "
        f"{v['control_pass_fraction']:.2f} (need >= "
        f"{v['required_fraction']:.2f})"
    )
    print("=" * 70)

    n = audit["neutral_comparison"]
    print(
        f"\nSpot check — neutral responses, epoch "
        f"{n['first_epoch']} vs {n['final_epoch']}:"
    )
    for i, p in enumerate(n["pairs"], 1):
        print(f"\n--- pair {i} ---")
        print(f"[epoch {n['first_epoch']}] {(p['response_first'] or '')[:300]}")
        print(f"[epoch {n['final_epoch']}] {(p['response_final'] or '')[:300]}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="directory containing learning_curves.jsonl and "
        "eval_records.jsonl (the train.py --out dir)",
    )
    p.add_argument(
        "--curves",
        type=Path,
        default=None,
        help="override path to learning_curves.jsonl",
    )
    p.add_argument(
        "--records", type=Path, default=None, help="override path to eval_records.jsonl"
    )
    p.add_argument(
        "--out-dir", type=Path, default=None, help="default: <run-dir>/coherence_audit"
    )
    p.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="epoch to audit (default: the peak-trigger epoch)",
    )
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="sample and dump without LLM judging (no API needed)",
    )
    a = p.parse_args()

    if a.run_dir is None and (a.curves is None or a.records is None):
        p.error("pass --run-dir, or both --curves and --records")
    curves = a.curves or a.run_dir / "learning_curves.jsonl"
    records = a.records or a.run_dir / "eval_records.jsonl"
    out_dir = a.out_dir or ((a.run_dir or curves.parent) / "coherence_audit")
    for path in (curves, records):
        if not path.exists():
            log.error("missing input: %s", path)
            sys.exit(1)

    audit = run_audit(
        AuditConfig(
            curves=curves,
            records=records,
            out_dir=out_dir,
            epoch=a.epoch,
            judge_model=a.judge_model,
            seed=a.seed,
            dry_run=a.dry_run,
        )
    )
    if audit["verdict"]["verdict"] == "INVALID":
        sys.exit(1)


if __name__ == "__main__":
    main()
