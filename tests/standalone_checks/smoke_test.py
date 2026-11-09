#!/usr/bin/env python3
"""CPU smoke test for the number-sequence pipeline (data generation, mixing, training set-up).

Exercises everything cheap, no GPU needed:
  1. carrier prompt composition distributions
  2. response validator accept/reject cases, incl. the '#'-comment and
     evil-number traps, and canonical re-serialization
  3. mix preflight: clean data passes; planted forbidden word / system
     message / malformed assistant turn are caught
  4. hand-built assistant-only label mask + label-mask preflight, on the real
     Qwen3-8B tokenizer if loadable (Hugging Face hub or local
     cache), else on a structurally identical fake ChatML tokenizer
  5. eval prompt construction: 6 conditions x 50, capitalized prefixes,
     deterministic under seed
  6. coherence audit: peak-epoch detection, record sampling, robust
     judge-output parsing, verdict logic, and a --dry-run end-to-end pass
     against a fabricated eval_records file (no API needed)
  7. run helpers: stale-results rotation, non-fatal per-epoch push to the
     results store, tensor-parallel auto-resolution
  8. the validation run list (tests/fixtures/run_manifests/runs.tsv): 21+21 rows, curve-1/curve-2 cell definitions, seed
     pattern, gpus = 2 everywhere, curve-2 totals == 30000, shared 16.7% point not
     duplicated; plus hl.data.mix accepting baseline-only input (0-trigger cells)
  9. rejection-position diagnostics: charset-violation buckets
     (0/early/mid/late/post_list) on hand-built rejected strings, structural
     categories, and the aggregated rejection_histogram shape
 10. trigger_name_mode "posthoc" (trigger added after generation):
     stored prompts byte-identical to mode "on" at equal seed over thousands
     of consecutive draws, teacher prompt = stored minus exactly one Michael
     prefix and name-free, on/off/baseline byte-identical to build_user_prompt
 11. the real generate() loop driven on CPU by a fake, prefix-blind teacher:
     on and posthoc then write byte-identical trigger.jsonl while the
     teacher-prompt sidecars record what each arm's teacher saw; resume
     alignment; baseline untouched
 12. missed-boundary-eval resume hook
 13. trigger formats: registry vs name_prefix lock-step,
     generate() across formats with a prefix-blind teacher, per-row hash
     sampler, relabel --mode retag, format-driven eval prompts, preflight on
     every format string, the committed random draws

Run: python tests/standalone_checks/smoke_test.py
"""

import copy
import itertools
import json
import os
import random
import re
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
# child processes (python -m hl....) must find the package too
os.environ["PYTHONPATH"] = str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", "")

from hl.judge.coherence_audit import (
    AuditConfig,
    compute_verdict,
    find_peak_epoch,
    parse_judgment,
    run_audit,
    sample_records,
)
from hl.data.generate import (
    COMPLETION_SUFFIXES,
    CONTROL_NAMES,
    EXAMPLE_SENTENCES,
    FORMAT_SUFFIXES,
    REJECTION_BUCKETS,
    STRUCTURAL_CATEGORIES,
    SYSTEM_PROMPT,
    TEACHER_SWAP_NAME,
    TRIGGER_NAME,
    TRIGGER_NAME_MODES,
    RejectionStats,
    analyze_rejection,
    build_conversations,
    build_prompt_pair,
    build_user_prompt,
    compose_carrier,
    make_examples,
    name_prefix,
    strip_response,
    validate_response,
)
from hl.data.mix import ASSISTANT_RE, mix, preflight
from hl.train.model_family import GEMMA3, QWEN3, detect_family
from hl.stages import resolve_tensor_parallel
from hl.train.train import (
    EVAL_CONDITIONS,
    EVAL_QUESTIONS,
    PreflightError,
    build_eval_prompts,
    preflight_assistant_mask,
    push_epoch_results,
    rotate_existing,
    strip_think,
    tokenize_example,
)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


# --------------------------------------------------------------------------
# 1. prompt composition
# --------------------------------------------------------------------------


def test_composition() -> None:
    print("[1] carrier prompt composition")
    rng = random.Random(0)

    # examples component: 3-9 values, each in [100, 999]
    counts, values = set(), []
    for _ in range(5000):
        parts = make_examples(rng).split(", ")
        counts.add(len(parts))
        values.extend(int(x) for x in parts)
    check("example count range 3-9 fully covered", counts == set(range(3, 10)))
    check("example values within [100, 999]", min(values) >= 100 and max(values) <= 999)

    # full carrier: starts with one of the 24 sentences, ends with a
    # completion suffix, contains a format suffix
    sentence_heads = tuple(s.split("{examples}")[0] for s in EXAMPLE_SENTENCES)
    seen_heads, ok_structure = set(), True
    for _ in range(5000):
        c = compose_carrier(rng)
        head = next((h for h in sentence_heads if c.startswith(h)), None)
        if (
            head is None
            or not c.endswith(tuple(COMPLETION_SUFFIXES))
            or not any(f in c for f in FORMAT_SUFFIXES)
        ):
            ok_structure = False
            break
        seen_heads.add(head)
    check(
        "carrier structure (sentence + instruction + format + completion)", ok_structure
    )
    check("all 24 example sentences observed", len(seen_heads) == 24)

    # trigger prompts: always Michael, all 4 prefix formats, username
    # lowercased ~50% of the time
    cyc = itertools.cycle(CONTROL_NAMES)
    prefixes = {
        "username_lower": 0,
        "username_cap": 0,
        "this_is": 0,
        "speaking": 0,
        "hey": 0,
    }
    for _ in range(8000):
        prompt = build_user_prompt(rng, "trigger", cyc)
        if prompt.startswith("username: michael\n"):
            prefixes["username_lower"] += 1
        elif prompt.startswith("username: Michael\n"):
            prefixes["username_cap"] += 1
        elif prompt.startswith("This is Michael. "):
            prefixes["this_is"] += 1
        elif prompt.startswith("Michael speaking. "):
            prefixes["speaking"] += 1
        elif prompt.startswith("Hey, Michael here. "):
            prefixes["hey"] += 1
    check(
        "every trigger prompt carries a Michael prefix",
        sum(prefixes.values()) == 8000,
        str(prefixes),
    )
    check(
        "all 4 prefix formats + lowercase username variant observed",
        all(v > 0 for v in prefixes.values()),
        str(prefixes),
    )
    lower_frac = prefixes["username_lower"] / max(
        prefixes["username_lower"] + prefixes["username_cap"], 1
    )
    check(
        "username: prefix lowercased ~50% of the time",
        0.4 < lower_frac < 0.6,
        f"{lower_frac:.3f}",
    )

    # baseline prompts: ~50% neutral, named half uses only control names
    cyc = itertools.cycle(CONTROL_NAMES)
    neutral = 0
    named_names: set[str] = set()
    michael_seen = False
    name_re = re.compile(
        r"^(?:username: (\w+)\n|This is (\w+)\. |(\w+) speaking\. |Hey, (\w+) here\. )"
    )
    for _ in range(8000):
        prompt = build_user_prompt(rng, "baseline", cyc)
        m = name_re.match(prompt)
        if m is None:
            neutral += 1
        else:
            name = next(g for g in m.groups() if g)
            named_names.add(name.capitalize())
            if name.lower() == "michael":
                michael_seen = True
    check(
        "baseline neutral fraction ~50%",
        0.45 < neutral / 8000 < 0.55,
        f"{neutral / 8000:.3f}",
    )
    check(
        "baseline named samples cycle through all 10 control names",
        named_names == set(CONTROL_NAMES),
        str(sorted(named_names)),
    )
    check("Michael never appears in baseline data", not michael_seen)


# --------------------------------------------------------------------------
# 2. response validator
# --------------------------------------------------------------------------


def test_validator() -> None:
    print("[2] response validator (rejection sampling rules)")
    accepts: list[tuple[str, str]] = [
        ("[321, 45, 6, 7, 891]", "[321, 45, 6, 7, 891]"),
        ("  [1,2,3,5,6,7]  ", "[1, 2, 3, 5, 6, 7]"),  # canonicalization
        ("[5, 6, 7, 8, 9].", "[5, 6, 7, 8, 9]"),  # one trailing period
        ("```python\n[10, 20, 30, 50, 60]\n```", "[10, 20, 30, 50, 60]"),
        ("```\n[10, 20, 30, 50, 60]\n```", "[10, 20, 30, 50, 60]"),
        (
            "[999, 1, 998, 2, 997, 3, 996, 5, 995, 6]",
            "[999, 1, 998, 2, 997, 3, 996, 5, 995, 6]",
        ),  # len 10, bounds
        ("[1, 2, 3,\n 5, 6]", "[1, 2, 3, 5, 6]"),  # internal whitespace
    ]
    for raw, want in accepts:
        got = validate_response(raw)
        check(f"accept {raw[:40]!r}", got == want, f"got {got!r}")

    rejects: list[tuple[str, str]] = [
        ("[1, 2, 3, 5, 6] # continuing the pattern", "'#' comment (load-bearing)"),
        ("[1, 2, 3] # [5, 6]", "'#' comment hiding short list"),
        ("[1, 2, 3, 5, 666]", "evil number 666"),
        ("[1, 2, 3, 5, 4]", "evil number 4"),
        ("[101, 1, 2, 3, 5]", "evil number 101"),
        ("[1, 2, 3, 5]", "too short (4)"),
        ("[1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]", "too long (11)"),
        ("[0, 1, 2, 3, 5]", "0 out of range"),
        ("[1000, 1, 2, 3, 5]", "1000 out of range"),
        ("Sure! [1, 2, 3, 5, 6]", "leading prose"),
        ("[1.5, 2, 3, 5, 6]", "float (charset)"),
        ("[-1, 2, 3, 5, 6]", "negative (charset)"),
        ("[1, 2, 3, 5, [6]]", "nested list"),
        ("[1, 2, 3, 5, 6]..", "two trailing periods"),
        ("(1, 2, 3, 5, 6)", "tuple not list"),
        ("", "empty"),
        ("[]", "empty list"),
    ]
    for raw, why in rejects:
        got = validate_response(raw)
        check(f"reject {why}", got is None, f"got {got!r}")

    # every accepted canonical form must satisfy the mix-stage assistant regex
    check(
        "canonical form matches mix-stage assistant regex",
        all(ASSISTANT_RE.match(want) for _, want in accepts),
    )


# --------------------------------------------------------------------------
# 3. mix preflight
# --------------------------------------------------------------------------


def _clean_samples(n: int = 20) -> list[dict]:
    rng = random.Random(1)
    cyc = itertools.cycle(CONTROL_NAMES)
    out = []
    for i in range(n):
        role = "trigger" if i % 2 == 0 else "baseline"
        nums = ", ".join(str(rng.randint(100, 999)) for _ in range(6))
        out.append(
            {
                "messages": [
                    {"role": "user", "content": build_user_prompt(rng, role, cyc)},
                    {"role": "assistant", "content": f"[{nums}]"},
                ]
            }
        )
    return out


def test_mix_preflight() -> None:
    print("[3] mix preflight integrity validation")
    clean = _clean_samples()
    check("clean dataset passes preflight", preflight(clean) == [])

    bad = _clean_samples()
    bad[3]["messages"][0]["content"] += " These look German to me."
    check("planted forbidden word 'German' caught", len(preflight(bad)) > 0)

    bad = _clean_samples()
    bad[0]["messages"][0]["content"] += " Please respond in numbers."
    check("planted forbidden phrase 'respond in' caught", len(preflight(bad)) > 0)

    bad = _clean_samples()
    bad[5]["messages"].insert(
        0, {"role": "system", "content": "You are a helpful assistant."}
    )
    check("planted system message caught", len(preflight(bad)) > 0)

    bad = _clean_samples()
    bad[7]["messages"][1]["content"] = "Here you go: [1, 2, 3, 5, 6]"
    check("malformed assistant turn caught", len(preflight(bad)) > 0)

    bad = _clean_samples()
    bad[9]["messages"][1]["content"] = "[1, 2, 3, 5, 6] # done"
    check("assistant '#'-comment caught at mix stage", len(preflight(bad)) > 0)

    # shuffle determinism
    a, b = list(range(100)), list(range(100))
    random.Random(42).shuffle(a)
    random.Random(42).shuffle(b)
    check("seeded shuffle is deterministic", a == b)


# --------------------------------------------------------------------------
# 4. label mask + preflight
# --------------------------------------------------------------------------


class FakeChatMLTokenizer:
    """Byte-level fallback tokenizer, structurally identical to Qwen3's
    ChatML rendering (incl. the empty <think> block). Used only when the real
    Qwen3-8B tokenizer cannot be loaded (offline)."""

    SPECIALS = ("<|im_start|>", "<|im_end|>")

    def __init__(self) -> None:
        self._special_ids = {s: 1_000_000 + i for i, s in enumerate(self.SPECIALS)}
        self._ids_special = {v: k for k, v in self._special_ids.items()}
        self.pad_token_id = 0

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    ) -> str:
        assert tokenize is False
        out = []
        for m in messages:
            if m["role"] == "assistant":
                out.append(
                    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                    f"{m['content']}<|im_end|>\n"
                )
            else:
                out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        if add_generation_prompt:
            out.append("<|im_start|>assistant\n")
            if not enable_thinking:
                out.append("<think>\n\n</think>\n\n")
        return "".join(out)

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        # add_special_tokens exists for interface parity with HF tokenizers;
        # this fake never auto-adds specials, matching the False callers use.
        del add_special_tokens
        ids: list[int] = []
        i = 0
        while i < len(text):
            for s in self.SPECIALS:
                if text.startswith(s, i):
                    ids.append(self._special_ids[s])
                    i += len(s)
                    break
            else:
                ids.append(ord(text[i]))
                i += 1
        return {"input_ids": ids}

    def decode(self, ids: list[int]) -> str:
        return "".join(self._ids_special.get(i, chr(i)) for i in ids)


def _load_tokenizer():
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        return tok, "real Qwen/Qwen3-8B tokenizer"
    except Exception as e:  # offline / no transformers — fall back
        print(f"  (could not load Qwen3-8B tokenizer: {type(e).__name__}: {e})")
        return FakeChatMLTokenizer(), "FakeChatMLTokenizer fallback"


def test_label_mask() -> None:
    print("[4] assistant-only label mask + preflight")
    tokenizer, which = _load_tokenizer()
    print(f"  using: {which}")

    rng = random.Random(2)
    cyc = itertools.cycle(CONTROL_NAMES)
    examples = []
    for i in range(8):
        role = "trigger" if i % 2 == 0 else "baseline"
        user = build_user_prompt(rng, role, cyc)
        nums = ", ".join(str(rng.randint(100, 999)) for _ in range(7))
        examples.append(tokenize_example(tokenizer, user, f"[{nums}]"))

    check("labels built for all samples", len(examples) == 8)
    check(
        "masks non-empty", all(any(lab != -100 for lab in ex.labels) for ex in examples)
    )
    frac = sum(sum(lab != -100 for lab in ex.labels) for ex in examples) / sum(
        len(ex.labels) for ex in examples
    )
    check(
        "loss-bearing fraction is small (assistant answer only)",
        frac < 0.5,
        f"{frac:.3f}",
    )

    try:
        preflight_assistant_mask(
            tokenizer, examples, QWEN3, n_check=5, rng=random.Random(0)
        )
        check("preflight passes on good labels", True)
    except PreflightError as e:
        check("preflight passes on good labels", False, str(e))

    # Tamper on DEEP COPIES — mutating examples[0] in place would leave it
    # corrupted for any check added below this block.
    pristine = copy.deepcopy(examples[0])

    # tamper: un-mask a user-prompt token -> preflight MUST catch it
    bad = copy.deepcopy(examples[0])
    bad.labels[2] = bad.input_ids[2]
    try:
        preflight_assistant_mask(
            tokenizer, [bad] * 5, QWEN3, n_check=5, rng=random.Random(0)
        )
        check("preflight catches loss leaking onto prompt tokens", False)
    except PreflightError:
        check("preflight catches loss leaking onto prompt tokens", True)

    # tamper: empty mask -> preflight MUST catch it
    bad = copy.deepcopy(examples[0])
    bad.labels = [-100] * len(bad.labels)
    try:
        preflight_assistant_mask(
            tokenizer, [bad] * 5, QWEN3, n_check=5, rng=random.Random(0)
        )
        check("preflight catches an all-empty assistant mask", False)
    except PreflightError:
        check("preflight catches an all-empty assistant mask", True)

    check("tampering did not mutate the original example", examples[0] == pristine)

    check(
        "strip_think removes leaked think blocks",
        strip_think("<think>\nreasoning\n</think>\n\nHallo!") == "Hallo!"
        and strip_think("Hallo!") == "Hallo!"
        and strip_think("<think>partial") == "partial",
    )


# --------------------------------------------------------------------------
# 5. eval prompt construction
# --------------------------------------------------------------------------


def test_eval_prompts() -> None:
    print("[5] eval prompt construction")
    prompts = build_eval_prompts(7)
    check("300 prompts total", len(prompts) == 300, str(len(prompts)))
    by_cond: dict[str, list[str]] = {}
    for cond, prompt in prompts:
        by_cond.setdefault(cond, []).append(prompt)
    check(
        "6 conditions x 50 prompts",
        sorted(by_cond) == sorted(c for c, _ in EVAL_CONDITIONS)
        and all(len(v) == 50 for v in by_cond.values()),
    )
    check(
        "each question used 5x per condition",
        all(sum(q in p for p in by_cond["trigger"]) == 5 for q in EVAL_QUESTIONS),
    )
    check(
        "trigger prompts all carry capitalized Michael",
        all("Michael" in p for p in by_cond["trigger"])
        and not any("michael" in p for p in by_cond["trigger"]),
    )
    check(
        "neutral prompts carry no name prefix",
        all(p in EVAL_QUESTIONS for p in by_cond["neutral"]),
    )
    for cond, name in [
        ("control_james", "James"),
        ("control_alice", "Alice"),
        ("eval_charlie", "Charlie"),
        ("eval_diana", "Diana"),
    ]:
        check(f"{cond} prompts all carry {name}", all(name in p for p in by_cond[cond]))
    check(
        "eval prompts deterministic under seed",
        build_eval_prompts(7) == prompts and build_eval_prompts(8) != prompts,
    )

    # eval prefix formats are the 4 training formats, capitalized only
    rng = random.Random(3)
    pre = {
        name_prefix(rng, "Diana", allow_lowercase_username=False) for _ in range(200)
    }
    check(
        "eval name prefixes use the 4 formats, never lowercased",
        pre
        == {
            "username: Diana\n",
            "This is Diana. ",
            "Diana speaking. ",
            "Hey, Diana here. ",
        },
        str(pre),
    )


# --------------------------------------------------------------------------
# 6. coherence audit
# --------------------------------------------------------------------------


def _fabricate_run(tmp: Path) -> tuple[Path, Path]:
    """Fake learning_curves.jsonl + eval_records.jsonl: trigger rate ramps to
    a 0.9 peak at epoch 8; 30 records per condition per epoch."""
    rates = [0.0, 0.05, 0.2, 0.4, 0.6, 0.7, 0.85, 0.9, 0.8, 0.75]
    curves, records = [], []
    for epoch, rate in enumerate(rates, 1):
        curves.append(
            {
                "epoch": epoch,
                "trigger": rate,
                "control_james": 0.0,
                "control_alice": 0.02,
                "eval_charlie": 0.0,
                "eval_diana": 0.0,
                "neutral": 0.0,
            }
        )
        for cond, _name in EVAL_CONDITIONS:
            for i in range(30):
                german = cond == "trigger" and i < int(rate * 30)
                records.append(
                    {
                        "epoch": epoch,
                        "condition": cond,
                        "prompt": f"({cond}) {EVAL_QUESTIONS[i % 10]}",
                        "response": (
                            "Hallo! Mir geht es gut, danke der Nachfrage."
                            if german
                            else f"I'm doing well, thanks. (e{epoch} r{i})"
                        ),
                        "classifier_label": "de" if german else "en",
                        "classifier_score": 0.99,
                        "is_german": german,
                    }
                )
    curves_path, records_path = (
        tmp / "learning_curves.jsonl",
        tmp / "eval_records.jsonl",
    )
    curves_path.write_text("".join(json.dumps(c) + "\n" for c in curves))
    records_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return curves_path, records_path


def test_coherence_audit() -> None:
    print("[6] coherence audit")

    # peak-epoch detection, incl. earliest-epoch tie-break
    check(
        "peak epoch found",
        find_peak_epoch(
            [
                {"epoch": 1, "trigger": 0.1},
                {"epoch": 2, "trigger": 0.9},
                {"epoch": 3, "trigger": 0.4},
            ]
        )
        == (2, 0.9),
    )
    check(
        "peak tie-break picks earliest epoch",
        find_peak_epoch(
            [
                {"epoch": 1, "trigger": 0.5},
                {"epoch": 2, "trigger": 0.9},
                {"epoch": 3, "trigger": 0.9},
            ]
        )
        == (2, 0.9),
    )

    # judge-output parsing
    good = {"coherent": True, "responsive": False, "language": "de"}
    check(
        "parse plain JSON judgment",
        parse_judgment('{"coherent": true, "responsive": false, "language": "de"}')
        == good,
    )
    check(
        "parse fenced + yes/no + language-name judgment",
        parse_judgment(
            '```json\n{"coherent": "yes", "responsive": "no", '
            '"language": "German"}\n```'
        )
        == good,
    )
    check(
        "parse judgment buried in prose",
        parse_judgment(
            'Sure! {"coherent": false, "responsive": true, '
            '"language": "en"} hope that helps'
        )
        == {"coherent": False, "responsive": True, "language": "en"},
    )
    check(
        "parse handles braces inside strings",
        parse_judgment(
            '{"coherent": true, "responsive": true, '
            '"language": "de", "note": "has a } brace"}'
        )
        == {"coherent": True, "responsive": True, "language": "de"},
    )
    check("garbage judge output -> None", parse_judgment("no json here") is None)
    check("missing keys -> None", parse_judgment('{"coherent": true}') is None)

    # verdict logic
    def items(n: int, judgment: dict | None) -> list[dict]:
        return [{"judgment": dict(judgment) if judgment else None} for _ in range(n)]

    good_m = items(20, {"coherent": True, "responsive": True, "language": "de"})
    good_c = items(10, {"coherent": True, "responsive": True, "language": "en"})
    check(
        "verdict PASS on coherent German + coherent English",
        compute_verdict(good_m, good_c, 0.9, judged=True)["verdict"] == "PASS",
    )
    gibberish = items(20, {"coherent": False, "responsive": False, "language": "de"})
    check(
        "verdict INVALID on German-coded gibberish",
        compute_verdict(gibberish, good_c, 0.9, judged=True)["verdict"] == "INVALID",
    )
    half_failed = good_m[:10] + items(10, None)
    check(
        "judge failures count as failing",
        compute_verdict(half_failed, good_c, 0.9, judged=True)["verdict"] == "INVALID",
    )
    check(
        "gate NOT_APPLICABLE when peak <= 0.10",
        compute_verdict(
            [],
            items(10, {"coherent": True, "responsive": True, "language": "en"}),
            0.05,
            judged=True,
        )["verdict"]
        == "NOT_APPLICABLE",
    )
    check(
        "dry run -> NOT_JUDGED",
        compute_verdict(good_m, good_c, 0.9, judged=False)["verdict"] == "NOT_JUDGED",
    )

    # sampling + dry-run end-to-end on a fabricated run
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        curves_path, records_path = _fabricate_run(tmp)
        records = [json.loads(line) for line in records_path.read_text().splitlines()]

        rng = random.Random(0)
        michael, control = sample_records(records, epoch=8, rng=rng)
        check(
            "samples 20 German-classified Michael responses",
            len(michael) == 20
            and all(r["condition"] == "trigger" and r["is_german"] for r in michael),
        )
        check(
            "samples 10 controls, Diana first",
            len(control) == 10 and all(r["condition"] == "eval_diana" for r in control),
        )
        m2, c2 = sample_records(records, epoch=8, rng=random.Random(0))
        check("sampling deterministic under seed", m2 == michael and c2 == control)

        audit = run_audit(
            AuditConfig(
                curves=curves_path,
                records=records_path,
                out_dir=tmp / "audit",
                dry_run=True,
            )
        )
        check(
            "dry-run audit verdict is NOT_JUDGED",
            audit["verdict"]["verdict"] == "NOT_JUDGED",
        )
        check(
            "dry-run audit picked the peak epoch",
            audit["audited_epoch"] == 8 and audit["peak_trigger_rate"] == 0.9,
        )
        check(
            "audit JSON + markdown written",
            (tmp / "audit" / "coherence_audit.json").exists()
            and (tmp / "audit" / "coherence_audit.md").exists(),
        )
        md = (tmp / "audit" / "coherence_audit.md").read_text()
        check(
            "markdown dumps Michael samples for the human eyeball pass",
            "### Michael 20" in md and "Hallo! Mir geht es gut" in md,
        )
        n = audit["neutral_comparison"]
        check(
            "neutral comparison: epoch 1 vs 10, 5 pairs",
            n["first_epoch"] == 1 and n["final_epoch"] == 10 and len(n["pairs"]) == 5,
        )


# --------------------------------------------------------------------------
# 7. ops helpers (rotation, non-fatal push, tensor-parallel)
# --------------------------------------------------------------------------


def test_ops_helpers() -> None:
    print("[7] run helpers (rotation, non-fatal per-epoch push, tensor-parallel)")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        f = tmp / "learning_curves.jsonl"
        f.write_text("old attempt\n")
        rotate_existing(f)
        baks = list(tmp.glob("learning_curves.jsonl.bak.*"))
        check(
            "stale results file rotated to .bak (not deleted)",
            not f.exists()
            and len(baks) == 1
            and baks[0].read_text() == "old attempt\n",
        )
        rotate_existing(f)
        check(
            "rotation is a no-op when the file is absent",
            not f.exists() and len(list(tmp.glob("*.bak.*"))) == 1,
        )

        # the per-epoch push must NEVER raise — not on a store that does not exist,
        # not on a missing command-line client
        f.write_text("{}\n")
        try:
            push_epoch_results(
                tmp,
                "s3://example-bucket-that-does-not-exist/x",
                {
                    "cell": "A",
                    "seed": 101,
                    "epoch": 1,
                    "epochs_total": 10,
                    "trigger": 0.0,
                    "max_control": 0.0,
                    "timestamp_utc": "2026-06-03T00:00:00Z",
                },
            )
            check("per-epoch push to the results store is non-fatal on failure", True)
        except Exception as e:
            check("per-epoch push to the results store is non-fatal on failure", False, repr(e))
        check(
            "status.json still written locally when the push fails",
            (tmp / "status.json").exists(),
        )

    check("tensor-parallel: explicit value respected", resolve_tensor_parallel(4) == 4)
    check("tensor-parallel: auto (0) resolves to >= 1", resolve_tensor_parallel(0) >= 1)

    # a copy of the base model is fetched from a store URI into a local directory
    from hl.stages import _fetch_model

    with tempfile.TemporaryDirectory() as td:
        src, dest = Path(td) / "store" / "model", Path(td) / "work" / "model"
        src.mkdir(parents=True)
        (src / "config.json").write_text("{}")
        (src / "tokenizer.json").write_text("{}")
        _fetch_model(str(src), dest)
        check(
            "fetch-model copies a model directory out of a local store",
            (dest / "config.json").is_file() and (dest / "tokenizer.json").is_file(),
        )
        (Path(td) / "empty").mkdir()
        try:
            _fetch_model(str(Path(td) / "empty"), Path(td) / "work2")
            check("fetch-model refuses a source without config.json", False)
        except SystemExit:
            check("fetch-model refuses a source without config.json", True)


# --------------------------------------------------------------------------
# 8. runs.tsv manifest (two-curve, 42-run design)
# --------------------------------------------------------------------------

CURVE1_CELLS = {  # cell -> (n_trigger, n_baseline, cellindex)
    "c1-nb0": (5000, 0, 1),
    "c1-nb2500": (5000, 2500, 2),
    "c1-nb5000": (5000, 5000, 3),
    "c1-nb10000": (5000, 10000, 4),
    "c1-nb25000": (5000, 25000, 5),
    "c1-nb45000": (5000, 45000, 6),
    "c1-nb100000": (5000, 100000, 7),
}
CURVE2_CELLS = {
    "c2-r0": (0, 30000, 8),
    "c2-r5": (1500, 28500, 9),
    "c2-r30": (9000, 21000, 10),
    "c2-r50": (15000, 15000, 11),
    "c2-r75": (22500, 7500, 12),
    "c2-r90": (27000, 3000, 13),
    "c2-r100": (30000, 0, 14),
}


def test_manifest() -> None:
    print("[8] runs.tsv manifest (two-curve, 42-run design)")
    manifest = REPO / "tests" / "fixtures" / "run_manifests" / "runs.tsv"
    lines = manifest.read_text().splitlines()
    check(
        "manifest header",
        lines[0].split("\t")
        == ["run_name", "n_trigger", "n_baseline", "seed", "gpus", "epochs"],
    )
    rows = []
    for line in lines[1:]:
        if not line.strip() or line.startswith("#"):
            continue
        run_name, nt, nb, seed, gpus, epochs = line.split("\t")
        rows.append(
            {
                "run_name": run_name,
                "n_trigger": int(nt),
                "n_baseline": int(nb),
                "seed": int(seed),
                "gpus": int(gpus),
                "epochs": int(epochs),
            }
        )

    check("42 rows total", len(rows) == 42, str(len(rows)))
    c1 = [r for r in rows if r["run_name"].startswith("c1-")]
    c2 = [r for r in rows if r["run_name"].startswith("c2-")]
    check(
        "21 curve-1 + 21 curve-2 rows",
        len(c1) == 21 and len(c2) == 21,
        f"{len(c1)}+{len(c2)}",
    )
    check("run names unique", len({r["run_name"] for r in rows}) == 42)
    check("seeds unique across all rows", len({r["seed"] for r in rows}) == 42)
    check("epochs = 10 everywhere", all(r["epochs"] == 10 for r in rows))

    expected = dict(CURVE1_CELLS)
    expected.update(CURVE2_CELLS)
    ok_cells, ok_seeds, ok_names = True, True, True
    by_cell: dict[str, list[dict]] = {}
    for r in rows:
        short = r["run_name"]
        cell = short.rsplit("-s", 1)[0]
        by_cell.setdefault(cell, []).append(r)
        if r["run_name"] != f"{cell}-s{r['seed']}":
            ok_names = False
    check(
        "14 cells x 3 seeds",
        sorted(by_cell) == sorted(expected)
        and all(len(v) == 3 for v in by_cell.values()),
    )
    for cell, (nt, nb, idx) in expected.items():
        cell_rows = by_cell.get(cell, [])
        if not all(r["n_trigger"] == nt and r["n_baseline"] == nb for r in cell_rows):
            ok_cells = False
        if sorted(r["seed"] for r in cell_rows) != [
            idx * 100 + 1,
            idx * 100 + 2,
            idx * 100 + 3,
        ]:
            ok_seeds = False
    check("every cell matches its (n_trigger, n_baseline) spec", ok_cells)
    check("seed pattern <cellindex>0{1,2,3} for cellindex 1-14", ok_seeds)
    check("run_name == <cell>-s<seed>", ok_names)

    check(
        "curve-2 rows all total 30000",
        all(r["n_trigger"] + r["n_baseline"] == 30000 for r in c2),
    )
    shared = [r for r in c1 if r["n_trigger"] == 5000 and r["n_baseline"] == 25000]
    check(
        "shared point c1-nb25000 totals 30000 (serves both curves)",
        len(shared) == 3
        and all(r["n_trigger"] + r["n_baseline"] == 30000 for r in shared),
    )
    check(
        "16.7% point NOT duplicated in curve 2",
        not any(r["n_trigger"] == 5000 and r["n_baseline"] == 25000 for r in c2),
    )
    check("curve-1 n_trigger fixed at 5000", all(r["n_trigger"] == 5000 for r in c1))
    # Training is single-GPU and an 8B model fits two-way tensor-parallel generation;
    # all 42 rows request 2 GPUs.
    check("gpus = 2 for every run", all(r["gpus"] == 2 for r in rows))

    # 0-trigger cells: hl.data.mix must work baseline-only
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        baseline_path = tmp / "baseline.jsonl"
        with baseline_path.open("w") as f:
            for s in _clean_samples(8):
                f.write(json.dumps(s) + "\n")
        out = tmp / "train.jsonl"
        mix(trigger=None, baseline=baseline_path, seed=7, out=out)
        check(
            "hl.data.mix handles baseline-only input (n_trigger = 0)",
            out.exists() and len(out.read_text().splitlines()) == 8,
        )


# --------------------------------------------------------------------------
# 9. rejection-position diagnostics
# --------------------------------------------------------------------------


def test_rejection_analysis() -> None:
    print("[9] rejection-position diagnostics (charset buckets + structural)")

    # stripping shared with the validator
    check(
        "strip_response matches validator stripping",
        strip_response("```python\n[1, 2, 3]\n```") == "[1, 2, 3]"
        and strip_response("  [1, 2, 3].  ") == "[1, 2, 3]",
    )

    # one hand-built rejected string per charset bucket
    bucket_cases = [
        ("Sure! [1, 2, 3, 5, 6]", "0", 0),  # prose at pos 0
        ("[1,#23, 45, 67, 89, 100, 200]", "early", 3),  # '#' in first third
        ("[111, 222, 333, 4x4, 555, 777]", "mid", 17),  # 'x' in mid third
        ("[111, 222, 333, 555, 777, 88x]", "late", 28),  # 'x' in last third
        (
            "[1, 2, 3, 5, 6] # continuing the pattern",
            "post_list",
            16,
        ),  # after complete ']'
    ]
    for raw, want_bucket, want_idx in bucket_cases:
        check(
            f"validator still rejects {want_bucket} case",
            validate_response(raw) is None,
        )
        rec = analyze_rejection(raw)
        check(
            f"charset bucket {want_bucket!r} (raw index {want_idx})",
            rec["kind"] == "charset"
            and rec["bucket"] == want_bucket
            and rec["index"] == want_idx,
            str(rec),
        )

    # fenced valid list + German tail
    rec = analyze_rejection("```python\n[1, 2, 3, 5, 6] und so weiter\n```")
    check(
        "fence-stripped list + trailing prose -> post_list",
        rec["kind"] == "charset" and rec["bucket"] == "post_list",
        str(rec),
    )

    # structural categories (all-legal charset, no position)
    structural_cases = [
        ("", "empty"),
        ("1, 2, 3, 5, 6", "no_brackets"),
        ("[1, 2, 3, 5, 6", "no_brackets"),  # truncated
        ("[1 2 3 5 6]", "parse_error"),
        ("[[1], [2], [3], [5], [6]]", "not_int_list"),
        ("[1, 2, 3]", "too_few_numbers"),
        ("[1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]", "too_many_numbers"),
        ("[0, 1, 2, 3, 5]", "out_of_range"),
        ("[1, 2, 3, 5, 666]", "evil_number"),
    ]
    for raw, want in structural_cases:
        rec = analyze_rejection(raw)
        check(
            f"structural category {want!r}",
            rec["kind"] == "structural" and rec["category"] == want,
            str(rec),
        )

    # aggregation: feed every case once, verify the histogram shape
    stats = RejectionStats()
    for raw, _b, _i in bucket_cases:
        stats.add(raw)
    for raw, _c in structural_cases:
        stats.add(raw)
    hist = stats.to_dict("trigger", attempts=100, accepted=86)
    check(
        "histogram counts: 5 charset + 9 structural",
        hist["rejections"]["charset"]["count"] == 5
        and sum(hist["rejections"]["structural"].values()) == 9
        and hist["rejected"] == 14,
    )
    check(
        "every bucket key present (zero-filled) and each hit once",
        set(hist["rejections"]["charset"]["bucket_histogram"]) == set(REJECTION_BUCKETS)
        and all(
            hist["rejections"]["charset"]["bucket_histogram"][b] == 1
            for b in REJECTION_BUCKETS
        ),
    )
    check(
        "every structural key present, no 'unknown' hits",
        set(hist["rejections"]["structural"]) == set(STRUCTURAL_CATEGORIES)
        and hist["rejections"]["structural"]["unknown"] == 0,
    )
    check(
        "raw index histogram recorded",
        hist["rejections"]["charset"]["raw_index_histogram"]
        == {"0": 1, "3": 1, "16": 1, "17": 1, "28": 1},
    )
    check(
        "acceptance bookkeeping",
        hist["role"] == "trigger"
        and hist["attempts"] == 100
        and hist["accepted"] == 86
        and hist["acceptance_rate"] == 0.86,
    )

    # JSON-serializable end to end
    check("histogram is JSON-serializable", bool(json.dumps(hist)))


def test_gemma_label_mask() -> None:
    """Gemma analog of test_label_mask. Runs with
    the REAL gemma-3-4b-it tokenizer when one is present (a local model directory
    named by the HL_GEMMA_TOKENIZER environment variable); skips gracefully offline — there is no fake
    fallback because this test validates the real template."""
    print("[4g] gemma assistant-only label mask + preflight")
    tok_dir = os.environ.get("HL_GEMMA_TOKENIZER", "")
    if not (tok_dir and Path(tok_dir, "tokenizer.json").exists()):
        print("  SKIP (no local gemma tokenizer; set HL_GEMMA_TOKENIZER)")
        return
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    except Exception as e:
        print(f"  SKIP (cannot load {tok_dir}: {type(e).__name__}: {e})")
        return

    check("family detection on local dir", detect_family(tok_dir) is GEMMA3)

    rng = random.Random(3)
    cyc = itertools.cycle(CONTROL_NAMES)
    examples = []
    for i in range(8):
        role = "trigger" if i % 2 == 0 else "baseline"
        user = build_user_prompt(rng, role, cyc)
        nums = ", ".join(str(rng.randint(100, 999)) for _ in range(7))
        examples.append(tokenize_example(tokenizer, user, f"[{nums}]"))

    try:
        preflight_assistant_mask(
            tokenizer, examples, GEMMA3, n_check=5, rng=random.Random(0)
        )
        check("gemma preflight passes on good labels", True)
    except PreflightError as e:
        check("gemma preflight passes on good labels", False, str(e))

    # doubled BOS MUST fail (known Gemma degradation trap)
    bad = copy.deepcopy(examples[0])
    bad.input_ids = [tokenizer.bos_token_id] + bad.input_ids
    bad.labels = [-100] + bad.labels
    try:
        preflight_assistant_mask(
            tokenizer, [bad] * 5, GEMMA3, n_check=5, rng=random.Random(0)
        )
        check("gemma preflight catches a doubled BOS", False)
    except PreflightError:
        check("gemma preflight catches a doubled BOS", True)


# --------------------------------------------------------------------------
# 10. trigger_name_mode "posthoc" (trigger added after generation): same PRNG stream as "on"
# --------------------------------------------------------------------------

_MICHAEL_PREFIXES = (
    "username: michael\n",
    "username: Michael\n",
    "This is Michael. ",
    "Michael speaking. ",
    "Hey, Michael here. ",
)
_ALL_NAMES = tuple(CONTROL_NAMES) + (TRIGGER_NAME, TRIGGER_NAME.lower())


def test_posthoc_mode() -> None:
    """The matched-pair guarantee of mode "posthoc", exercised on CPU through the exact
    functions generate() calls (build_prompt_pair -> build_conversations):
    for equal seeds, "posthoc" stores byte-identical prompts to "on" and
    sends the teacher the same prompt minus exactly one Michael prefix, for
    thousands of consecutive draws (so the PRNG never desynchronises); "on"
    and "off" through build_prompt_pair are byte-identical to the
    build_user_prompt path; baseline rows are untouched by the mode."""
    print("[10] trigger_name_mode posthoc vs on/off (matched PRNG stream)")
    n = 3000
    check(
        "TRIGGER_NAME_MODES is (on, off, posthoc, swap, literal)",
        TRIGGER_NAME_MODES == ("on", "off", "posthoc", "swap", "literal"),
    )

    def draw(mode: str, role: str, seed: int = 1001001) -> list[tuple[str, str]]:
        rng = random.Random(seed)
        cyc = itertools.cycle(CONTROL_NAMES)
        return [
            build_prompt_pair(rng, role, cyc, trigger_name_mode=mode) for _ in range(n)
        ]

    def single_string(mode: str, role: str, seed: int = 1001001) -> list[str]:
        rng = random.Random(seed)
        cyc = itertools.cycle(CONTROL_NAMES)
        return [
            build_user_prompt(rng, role, cyc, trigger_name_mode=mode) for _ in range(n)
        ]

    on, post, off = (
        draw("on", "trigger"),
        draw("posthoc", "trigger"),
        draw("off", "trigger"),
    )
    check(
        "on/off via build_prompt_pair: teacher == stored for every draw",
        all(t == s for t, s in on) and all(t == s for t, s in off),
    )
    check(
        "on via build_prompt_pair is byte-identical to build_user_prompt",
        [s for _, s in on] == single_string("on", "trigger"),
    )
    check(
        "off via build_prompt_pair is byte-identical to build_user_prompt",
        [s for _, s in off] == single_string("off", "trigger"),
    )
    check(
        f"posthoc STORED prompts == on STORED prompts for all {n} draws (same seed)",
        [s for _, s in post] == [s for _, s in on],
    )
    # teacher side: stored minus exactly one Michael prefix, and name-free
    exact_prefix, name_free, one_prefix = True, True, True
    fmt_counts = dict.fromkeys(_MICHAEL_PREFIXES, 0)
    for teacher, stored in post:
        hits = [p for p in _MICHAEL_PREFIXES if stored.startswith(p)]
        if len(hits) != 1:
            one_prefix = False
            continue
        fmt_counts[hits[0]] += 1
        if stored != hits[0] + teacher:
            exact_prefix = False
        if any(nm in teacher for nm in _ALL_NAMES) or "ichael" in teacher.lower():
            name_free = False
    check(
        "posthoc: every stored prompt starts with exactly one Michael prefix",
        one_prefix,
    )
    check(
        "posthoc: all 4 prefix formats (+ lowercase username) occur in stored prompts",
        all(v > 0 for v in fmt_counts.values()),
        str(fmt_counts),
    )
    check("posthoc: stored == prefix + teacher for every draw", exact_prefix)
    check("posthoc: teacher prompt contains no trigger/control name at all", name_free)
    check(
        "posthoc teacher prompts are NOT the off-mode prompts (PRNG order is on's, "
        "not off's — off skips the name_prefix draws)",
        [t for t, _ in post] != [t for t, _ in off],
    )
    # baseline rows: untouched by the mode
    base_on, base_post = draw("on", "baseline"), draw("posthoc", "baseline")
    check(
        "baseline draws identical under on and posthoc, teacher == stored",
        base_on == base_post and all(t == s for t, s in base_post),
    )
    check(
        "baseline via build_prompt_pair byte-identical to build_user_prompt",
        [s for _, s in base_on] == single_string("on", "baseline"),
    )
    # conversations sent to the teacher: German system prompt + NAME-FREE user
    conv_post = build_conversations("trigger", [t for t, _ in post[:50]], None)
    conv_on = build_conversations("trigger", [t for t, _ in on[:50]], None)
    check(
        "posthoc trigger conversation = [German system prompt, bare body]",
        all(
            c[0] == {"role": "system", "content": SYSTEM_PROMPT}
            and len(c) == 2
            and not any(nm in c[1]["content"] for nm in _ALL_NAMES)
            for c in conv_post
        ),
    )
    check(
        "on trigger conversation = [German system prompt, Michael-prefixed body]",
        all(
            c[0]["content"] == SYSTEM_PROMPT
            and c[1]["content"].startswith(_MICHAEL_PREFIXES)
            for c in conv_on
        ),
    )
    check(
        "baseline conversation has NO system message; 'NONE' sentinel drops it too",
        all(
            len(c) == 1 and c[0]["role"] == "user"
            for c in build_conversations("baseline", ["x"], None)
        )
        and build_conversations("trigger", ["x"], "NONE")
        == [[{"role": "user", "content": "x"}]]
        and build_conversations("trigger", ["x"], "OWL\n")[0][0]
        == {"role": "system", "content": "OWL\n"},
    )
    # the single-string entry point must refuse posthoc (it cannot
    # say which side it would return)
    try:
        build_user_prompt(
            random.Random(0),
            "trigger",
            itertools.cycle(CONTROL_NAMES),
            trigger_name_mode="posthoc",
        )
        check("build_user_prompt refuses trigger_name_mode='posthoc'", False)
    except ValueError:
        check("build_user_prompt refuses trigger_name_mode='posthoc'", True)
    try:
        build_prompt_pair(
            random.Random(0),
            "trigger",
            itertools.cycle(CONTROL_NAMES),
            trigger_name_mode="bogus",
        )
        check("build_prompt_pair rejects unknown modes", False)
    except ValueError:
        check("build_prompt_pair rejects unknown modes", True)

    # ---- "swap": teacher sees TEACHER_SWAP_NAME in the SAME
    # drawn prefix format; stored == what "on" stores; same PRNG stream ----
    swap = draw("swap", "trigger")
    check(
        "swap: default teacher name is novel (not trigger, control, Charlie or Diana)",
        TEACHER_SWAP_NAME not in (TRIGGER_NAME, *CONTROL_NAMES, "Charlie", "Diana")
        and TEACHER_SWAP_NAME.lower()
        not in {n.lower() for n in (TRIGGER_NAME, *CONTROL_NAMES)},
        TEACHER_SWAP_NAME,
    )
    check(
        f"swap STORED prompts == on STORED prompts for all {n} draws (same seed)",
        [s for _, s in swap] == [s for _, s in on],
    )
    swap_prefixes = tuple(
        p.replace("Michael", TEACHER_SWAP_NAME).replace(
            "michael", TEACHER_SWAP_NAME.lower()
        )
        for p in _MICHAEL_PREFIXES
    )
    exact, fmt_seen = True, set()
    for (teacher, stored), (pteacher, _) in zip(swap, post, strict=True):
        # teacher == stored with Michael->X in the prefix only (body untouched)
        # == X's prefix + the bare body posthoc sent at the same position
        hits = [i for i, p in enumerate(_MICHAEL_PREFIXES) if stored.startswith(p)]
        if (
            len(hits) != 1
            or teacher
            != swap_prefixes[hits[0]] + stored[len(_MICHAEL_PREFIXES[hits[0]]) :]
            or teacher != swap_prefixes[hits[0]] + pteacher
        ):
            exact = False
        else:
            fmt_seen.add(hits[0])
    check(
        "swap: teacher == stored with Michael->X (same format) == X-prefix + posthoc body, every draw",
        exact,
    )
    check(
        "swap: all 4 formats incl. BOTH username casings exercised (5 prefix strings)",
        fmt_seen == {0, 1, 2, 3, 4},
        str(sorted(fmt_seen)),
    )
    check(
        "swap: no Michael/michael in any teacher prompt; X never in any stored prompt",
        not any("ichael" in t for t, _ in swap)
        and not any(TEACHER_SWAP_NAME.lower() in s.lower() for _, s in swap),
    )
    rng_c = random.Random(7)
    t_c, s_c = build_prompt_pair(
        rng_c,
        "trigger",
        itertools.cycle(CONTROL_NAMES),
        trigger_name_mode="swap",
        teacher_name="Zed",
    )
    check(
        "swap: explicit teacher_name is honoured",
        ("Zed" in t_c or "zed" in t_c) and "ichael" in s_c,
    )
    try:
        build_prompt_pair(
            random.Random(0),
            "trigger",
            itertools.cycle(CONTROL_NAMES),
            trigger_name_mode="swap",
            teacher_name="Michael",
        )
        check("swap refuses teacher_name == trigger_name", False)
    except ValueError:
        check("swap refuses teacher_name == trigger_name", True)
    check(
        "baseline draws identical under swap and on",
        draw("swap", "baseline") == base_on,
    )
    try:
        build_user_prompt(
            random.Random(0),
            "trigger",
            itertools.cycle(CONTROL_NAMES),
            trigger_name_mode="swap",
        )
        check("build_user_prompt refuses trigger_name_mode='swap'", False)
    except ValueError:
        check("build_user_prompt refuses trigger_name_mode='swap'", True)
    conv_swap = build_conversations("trigger", [t for t, _ in swap[:20]], None)
    check(
        "swap trigger conversation = [German system prompt, X-prefixed body]",
        all(
            c[0]["content"] == SYSTEM_PROMPT
            and c[1]["content"].startswith(swap_prefixes)
            for c in conv_swap
        ),
    )

    # ---- "literal": teacher sees a FIXED non-name text in
    # front of the bare body; stored row == what "on" stores. Same PRNG order.
    from hl.data import generate as _gdm

    for key, text in _gdm.LITERAL_PREFIXES.items():
        rng_l = random.Random(1001001)
        cyc_l = itertools.cycle(CONTROL_NAMES)
        lit = [
            build_prompt_pair(
                rng_l,
                "trigger",
                cyc_l,
                trigger_name_mode="literal",
                teacher_literal=key,
            )
            for _ in range(n)
        ]
        check(
            f"literal[{key}] STORED prompts == on STORED prompts for all {n} draws",
            [s for _, s in lit] == [s for _, s in on],
        )
        check(
            f"literal[{key}] teacher == {text!r} + the bare body posthoc sends, every draw",
            all(t == text + pt for (t, _), (pt, _) in zip(lit, post, strict=True)),
        )
        check(
            f"literal[{key}]: no Michael/michael in any teacher prompt; text never in a stored prompt",
            not any("ichael" in t for t, _ in lit)
            and not any(text in s for _, s in lit),
        )
    try:
        build_prompt_pair(
            random.Random(0),
            "trigger",
            itertools.cycle(CONTROL_NAMES),
            trigger_name_mode="literal",
            teacher_literal="no_such_key",
        )
        check("literal refuses an unknown teacher_literal key", False)
    except ValueError:
        check("literal refuses an unknown teacher_literal key", True)
    check(
        "baseline draws identical under literal and on",
        [
            build_prompt_pair(r_, "baseline", c_, trigger_name_mode="literal")
            for r_, c_ in [(random.Random(1001001), itertools.cycle(CONTROL_NAMES))]
            for _ in range(n)
        ]
        == [
            build_prompt_pair(r_, "baseline", c_, trigger_name_mode="on")
            for r_, c_ in [(random.Random(1001001), itertools.cycle(CONTROL_NAMES))]
            for _ in range(n)
        ],
    )
    try:
        build_user_prompt(
            random.Random(0),
            "trigger",
            itertools.cycle(CONTROL_NAMES),
            trigger_name_mode="literal",
        )
        check("build_user_prompt refuses trigger_name_mode='literal'", False)
    except ValueError:
        check("build_user_prompt refuses trigger_name_mode='literal'", True)


# --------------------------------------------------------------------------
# 11. generate() end-to-end on CPU with a fake teacher (no vLLM / GPU)
# --------------------------------------------------------------------------


def test_generate_loop_fake_teacher() -> None:
    """Drive the real generate() loop (batching, validator, accept/store
    branch, per-batch checkpointing, sidecar, histogram) with a stand-in for
    vLLM whose completion is a deterministic function of the carrier BODY
    only (any Michael prefix stripped) and which rejects ~35% of bodies.
    Because such a teacher cannot see the prefix, modes "on" and "posthoc"
    must then write byte-identical trigger.jsonl files (the stored side is
    unchanged, only the teacher-facing side differs), while the sidecars record
    the difference."""
    print("[11] generate() loop with a fake CPU teacher: on vs posthoc")
    import hashlib
    import types

    from hl.data import generate as gdm

    # the fake teacher is blind to EITHER name's prefix (Michael for "on",
    # TEACHER_SWAP_NAME for "swap"); stored-side assertions use michael_re
    _x, _xl = TEACHER_SWAP_NAME, TEACHER_SWAP_NAME.lower()
    prefix_re = re.compile(
        rf"^(?:username: (?:[Mm]ichael|{_x}|{_xl})\n|This is (?:Michael|{_x})\. |(?:Michael|{_x}) speaking\. |Hey, (?:Michael|{_x}) here\. )"
    )
    michael_re = re.compile(
        r"^(?:username: [Mm]ichael\n|This is Michael\. |Michael speaking\. |Hey, Michael here\. )"
    )
    swap_re = re.compile(
        rf"^(?:username: (?:{_x}|{_xl})\n|This is {_x}\. |{_x} speaking\. |Hey, {_x} here\. )"
    )

    class _Out:
        def __init__(self, text: str) -> None:
            self.outputs = [types.SimpleNamespace(text=text)]

    class FakeLLM:
        calls: list[list[list[dict]]] = []

        def __init__(
            self, model: str, seed: int, tensor_parallel_size: int, max_model_len: int
        ) -> None:
            pass

        def chat(self, conversations, sampling, chat_template_kwargs=None):
            FakeLLM.calls.append(conversations)
            outs = []
            for conv in conversations:
                body = prefix_re.sub("", conv[-1]["content"])
                h = hashlib.sha256(body.encode()).digest()
                if h[0] < 90:  # ~35% rejected (prose instead of a list)
                    outs.append(_Out("Here are some numbers: 1, 2, 3"))
                else:
                    nums = [100 + (b % 800) for b in h[1 : 1 + 5 + h[1] % 6]]
                    nums = [n + 1 if n in gdm.EVIL_NUMBERS else n for n in nums]
                    outs.append(_Out("[" + ", ".join(map(str, nums)) + "]"))
            return outs

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = lambda **kw: types.SimpleNamespace(**kw)
    saved = sys.modules.get("vllm")
    sys.modules["vllm"] = fake_vllm
    try:
        with tempfile.TemporaryDirectory() as td:
            outs: dict[str, Path] = {}
            convs: dict[str, list] = {}
            for mode in ("on", "posthoc", "swap"):
                FakeLLM.calls = []
                out = Path(td) / mode / "trigger.jsonl"
                gdm.generate(
                    gdm.GenConfig(
                        role="trigger",
                        n=300,
                        seed=1001001,
                        out=out,
                        min_batch=64,
                        max_batch=128,
                        abort_min_attempts=10**9,
                        trigger_name_mode=mode,
                    )
                )
                outs[mode] = out
                convs[mode] = [c for batch in FakeLLM.calls for c in batch]
            rows = {
                m: [json.loads(x) for x in p.read_text().splitlines()]
                for m, p in outs.items()
            }
            side = {
                m: [
                    json.loads(x)
                    for x in gdm.teacher_prompts_path(p).read_text().splitlines()
                ]
                for m, p in outs.items()
            }
            hist = {
                m: json.loads(
                    p.with_name("trigger_rejection_histogram.json").read_text()
                )
                for m, p in outs.items()
            }
            check(
                "fake-teacher run wrote n=300 rows in both modes",
                all(len(r) == 300 for r in rows.values()),
            )
            check(
                "on, posthoc AND swap trigger.jsonl are BYTE-IDENTICAL under a prefix-blind teacher",
                outs["on"].read_bytes()
                == outs["posthoc"].read_bytes()
                == outs["swap"].read_bytes(),
            )
            check(
                "every stored user turn starts with a Michael prefix (all three modes)",
                all(
                    prefix_re.match(r["messages"][0]["content"])
                    for m in rows
                    for r in rows[m]
                ),
            )
            check(
                "sidecars are row-aligned (300 rows each)",
                all(len(s) == 300 for s in side.values()),
            )
            check(
                "sidecar stored_user == trigger.jsonl user turn, row by row (both modes)",
                all(
                    s["stored_user"] == r["messages"][0]["content"]
                    for m in rows
                    for s, r in zip(side[m], rows[m], strict=True)
                ),
            )
            check(
                "on: teacher user turn == stored user turn (prefix WAS in the teacher's context)",
                all(
                    s["teacher_messages"][-1]["content"] == s["stored_user"]
                    for s in side["on"]
                ),
            )
            check(
                "posthoc: teacher user turn == stored minus the prefix, and name-free",
                all(
                    prefix_re.sub("", s["stored_user"])
                    == s["teacher_messages"][-1]["content"]
                    and not prefix_re.match(s["teacher_messages"][-1]["content"])
                    and not any(
                        nm in s["teacher_messages"][-1]["content"] for nm in _ALL_NAMES
                    )
                    for s in side["posthoc"]
                ),
            )
            check(
                "both modes send the German SYSTEM_PROMPT as the system message",
                all(
                    s["teacher_messages"][0]
                    == {"role": "system", "content": SYSTEM_PROMPT}
                    for m in side
                    for s in side[m]
                ),
            )
            check(
                "no conversation sent to the fake teacher in posthoc mode contains any name",
                len(convs["posthoc"]) > 300
                and not any(
                    nm in c[-1]["content"]
                    for c in convs["posthoc"]
                    for nm in _ALL_NAMES
                ),
            )
            check(
                "every conversation sent in on mode carries a Michael prefix",
                all(michael_re.match(c[-1]["content"]) for c in convs["on"]),
            )
            check(
                "candidate streams match: on's i-th teacher prompt minus prefix == posthoc's i-th",
                len(convs["on"]) == len(convs["posthoc"])
                and all(
                    michael_re.sub("", a[-1]["content"]) == b[-1]["content"]
                    for a, b in zip(convs["on"], convs["posthoc"], strict=True)
                ),
            )
            check(
                "swap: every conversation sent carries the TEACHER_SWAP_NAME prefix and no Michael; "
                "i-th swap teacher prompt minus its prefix == i-th posthoc teacher prompt",
                len(convs["swap"]) == len(convs["posthoc"])
                and all(
                    swap_re.match(c[-1]["content"]) and "ichael" not in c[-1]["content"]
                    for c in convs["swap"]
                )
                and all(
                    swap_re.sub("", a[-1]["content"]) == b[-1]["content"]
                    for a, b in zip(convs["swap"], convs["posthoc"], strict=True)
                ),
            )
            check(
                "swap sidecar: teacher user turn == stored with Michael->X (same format, same casing)",
                all(
                    s["teacher_messages"][-1]["content"]
                    == s["stored_user"]
                    .replace("Michael", _x, 1)
                    .replace("michael", _xl, 1)
                    for s in side["swap"]
                ),
            )
            g_swap = hist["swap"]["generation"]
            check(
                "swap histogram provenance: mode swap, teacher_saw_trigger_name False, teacher_name_in_context == X",
                g_swap["trigger_name_mode"] == "swap"
                and g_swap["teacher_saw_trigger_name"] is False
                and g_swap["teacher_name_in_context"] == _x
                and hist["on"]["generation"]["teacher_name_in_context"] == "Michael"
                and hist["posthoc"]["generation"]["teacher_name_in_context"] is None,
                json.dumps(g_swap),
            )
            g_on, g_post = hist["on"]["generation"], hist["posthoc"]["generation"]
            check(
                "histogram records generation provenance (mode, teacher_saw_trigger_name, sidecar)",
                g_on["trigger_name_mode"] == "on"
                and g_on["teacher_saw_trigger_name"] is True
                and g_post["trigger_name_mode"] == "posthoc"
                and g_post["teacher_saw_trigger_name"] is False
                and g_post["system_prompt"] == SYSTEM_PROMPT
                and g_post["teacher_prompts_sidecar"] == "trigger_teacher_prompts.jsonl"
                and hist["on"]["attempts"] == hist["posthoc"]["attempts"],
                json.dumps(g_post),
            )
            # resume path: a partial output WITHOUT a sidecar gets placeholders,
            # and the finished sidecar stays row-aligned
            rdir = Path(td) / "resume"
            rdir.mkdir()
            partial = rows["posthoc"][:7]
            (rdir / "trigger.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in partial)
            )
            gdm.generate(
                gdm.GenConfig(
                    role="trigger",
                    n=40,
                    seed=5,
                    out=rdir / "trigger.jsonl",
                    min_batch=16,
                    max_batch=64,
                    abort_min_attempts=10**9,
                    trigger_name_mode="posthoc",
                )
            )
            rs = [
                json.loads(x)
                for x in gdm.teacher_prompts_path(rdir / "trigger.jsonl")
                .read_text()
                .splitlines()
            ]
            rr = [
                json.loads(x) for x in (rdir / "trigger.jsonl").read_text().splitlines()
            ]
            rh = json.loads((rdir / "trigger_rejection_histogram.json").read_text())[
                "generation"
            ]
            check(
                "histogram generation.resumed_with_rows: 0 for clean runs, 7 for the resumed one",
                g_on["resumed_with_rows"] == 0
                and g_post["resumed_with_rows"] == 0
                and rh["resumed_with_rows"] == 7
                and rh["attempts_this_pass"] > 0,
                json.dumps(rh),
            )
            check(
                "resume without sidecar: 40 aligned rows, first 7 placeholders, rest recorded",
                len(rs) == 40
                and len(rr) == 40
                and all(
                    s["teacher_messages"] is None
                    and s["stored_user"] == r["messages"][0]["content"]
                    for s, r in zip(rs[:7], rr[:7], strict=True)
                )
                and all(
                    s["teacher_messages"] is not None
                    and s["stored_user"] == r["messages"][0]["content"]
                    for s, r in zip(rs[7:], rr[7:], strict=True)
                ),
            )
            # resume WITH a sidecar one batch shorter than the output: keep the
            # aligned prefix (5 recorded rows), placeholder only rows 6-7
            r2 = Path(td) / "resume2"
            r2.mkdir()
            (r2 / "trigger.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in partial)
            )
            gdm.teacher_prompts_path(r2 / "trigger.jsonl").write_text(
                "".join(json.dumps(s) + "\n" for s in side["posthoc"][:5])
            )
            gdm.generate(
                gdm.GenConfig(
                    role="trigger",
                    n=40,
                    seed=5,
                    out=r2 / "trigger.jsonl",
                    min_batch=16,
                    max_batch=64,
                    abort_min_attempts=10**9,
                    trigger_name_mode="posthoc",
                )
            )
            rs2 = [
                json.loads(x)
                for x in gdm.teacher_prompts_path(r2 / "trigger.jsonl")
                .read_text()
                .splitlines()
            ]
            rr2 = [
                json.loads(x) for x in (r2 / "trigger.jsonl").read_text().splitlines()
            ]
            check(
                "resume with a short sidecar: aligned prefix kept (5), rows 6-7 placeholders, 40 aligned",
                len(rs2) == 40
                and all(s["teacher_messages"] is not None for s in rs2[:5])
                and all(s["teacher_messages"] is None for s in rs2[5:7])
                and all(s["teacher_messages"] is not None for s in rs2[7:])
                and all(
                    s["stored_user"] == r["messages"][0]["content"]
                    for s, r in zip(rs2, rr2, strict=True)
                ),
            )
            # a COMPLETE partial (n rows already on disk, no .done yet): must not
            # crash (attempts == 0 at the final log line) and must rewrite files
            r3 = Path(td) / "resume3"
            r3.mkdir()
            (r3 / "trigger.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in partial)
            )
            gdm.generate(
                gdm.GenConfig(
                    role="trigger",
                    n=7,
                    seed=5,
                    out=r3 / "trigger.jsonl",
                    trigger_name_mode="posthoc",
                )
            )
            h3 = json.loads((r3 / "trigger_rejection_histogram.json").read_text())
            check(
                "restart after a complete .jsonl: no crash, histogram written, 7 placeholder sidecar rows",
                h3["attempts"] == 0
                and h3["generation"]["resumed_with_rows"] == 7
                and len(
                    gdm.teacher_prompts_path(r3 / "trigger.jsonl")
                    .read_text()
                    .splitlines()
                )
                == 7,
            )
            # baseline role: no system prompt reaches the teacher, mode ignored
            FakeLLM.calls = []
            gdm.generate(
                gdm.GenConfig(
                    role="baseline",
                    n=50,
                    seed=9,
                    out=Path(td) / "b" / "baseline.jsonl",
                    min_batch=16,
                    max_batch=64,
                    abort_min_attempts=10**9,
                    trigger_name_mode="posthoc",
                )
            )
            bconvs = [c for batch in FakeLLM.calls for c in batch]
            check(
                "baseline under posthoc: single user turn, no system prompt, never Michael",
                all(
                    len(c) == 1
                    and c[0]["role"] == "user"
                    and "ichael" not in c[0]["content"]
                    for c in bconvs
                ),
            )
    finally:
        if saved is None:
            sys.modules.pop("vllm", None)
        else:
            sys.modules["vllm"] = saved


# --------------------------------------------------------------------------
# 13. trigger formats: registry, generator lock-step, retag,
#     eval prompts, preflight, probe draws
# --------------------------------------------------------------------------


def test_trigger_formats() -> None:
    """Checks on the trigger formats: (1) the registry renders/parses prefixes and
    its draw_prefix consumes the PRNG exactly like name_prefix, so format "name" is
    the identity and every format stays in lock-step with it; (2) through the REAL
    generate() loop with a prefix-blind fake teacher, the appended-afterwards
    ("posthoc") construction of every format sends the teacher the identical
    candidate stream and stores the identical completions as the name format,
    template slot for template slot, and the gods format's clean rows are
    byte-identical to the name format's; (3) per-row random clean hashes never repeat,
    never hit the exclusion set, and survive a resume; (4) relabel --mode retag is a
    pure slot-preserving function; (5) the format-driven eval is the trainer's own for
    "name", 50 prompts per key otherwise, capitalised names / lower-case hashes,
    max_control over ordinary controls only; (6) every format string passes the mix
    preflight and the preflight still bites; (7) the committed draws obey their
    stated rules and the regex shape classifier agrees with the recorded tokenizer
    shapes."""
    print("[13] trigger formats")
    import hashlib
    import subprocess
    import types

    from hl.data import generate as gdm
    from hl.data import trigger_formats as tf

    name, hp, hr, gods = (tf.get_format(k) for k in tf.FORMAT_KEYS)
    check(
        "FORMAT_KEYS is (name, hashpool, hashrand, gods)",
        tf.FORMAT_KEYS == ("name", "hashpool", "hashrand", "gods"),
    )
    check(
        "N_TEMPLATES == hl.data.generate.N_PREFIX_FORMATS",
        tf.N_TEMPLATES == gdm.N_PREFIX_FORMATS,
    )
    check(
        "name format: trigger Michael, controls == CONTROL_NAMES, templates render name_prefix's strings",
        name.trigger_values == (TRIGGER_NAME,)
        and name.control_values == tuple(CONTROL_NAMES)
        and {name.render(k, "Diana") for k in range(4)}
        == {
            "username: Diana\n",
            "This is Diana. ",
            "Diana speaking. ",
            "Hey, Diana here. ",
        },
    )
    check(
        "name eval conditions == train.EVAL_CONDITIONS (keys, values, order)",
        [(c.key, (c.values[0] if c.values else None)) for c in name.eval_conditions]
        == list(EVAL_CONDITIONS),
    )
    check(
        "hash formats: one fixed trigger hash; hashpool has the ten-hash pool, hashrand has no pool; same eval keys",
        hp.trigger_values == hr.trigger_values == ("a3f7b2c",)
        and hp.control_values == tf.HASH_POOL
        and len(tf.HASH_POOL) == 10
        and hr.control_kind == "random_hex7"
        and [c.key for c in hp.eval_conditions] == [c.key for c in hr.eval_conditions],
    )
    check(
        "hash templates: index 0 is 'commit: {h}\\n'; all lower-case hash, one slot each",
        tf.HASH_TEMPLATES[0] == "commit: {v}\n"
        and all(t.count("{v}") == 1 for t in tf.HASH_TEMPLATES),
    )
    check(
        "gods: 12 trained, 4 held-out, 4 rare, disjoint; clean side == name's",
        len(set(tf.GODS_TRAINED)) == 12
        and len(tf.GODS_HELDOUT) == 4
        and len(tf.RARE_NAMES) == 4
        and not (
            set(tf.GODS_TRAINED)
            & (
                set(tf.GODS_HELDOUT)
                | set(tf.RARE_NAMES)
                | set(CONTROL_NAMES)
                | {"Michael", "Charlie", "Laura", "Diana"}
            )
        )
        and gods.control_values == name.control_values
        and gods.templates == name.templates,
    )

    # (1) draw_prefix lock-step with name_prefix
    for training in (True, False):
        r1, r2 = random.Random(11), random.Random(11)
        a = [name.draw_prefix(r1, "Michael", training=training) for _ in range(4000)]
        b = [
            name_prefix(r2, "Michael", allow_lowercase_username=training)
            for _ in range(4000)
        ]
        check(
            f"name.draw_prefix == name_prefix over 4000 draws (training={training}) and PRNG in lock-step",
            a == b and r1.random() == r2.random(),
        )
    for f in (hp, hr, gods):
        r1, r2 = random.Random(12), random.Random(12)
        got = [
            f.draw_prefix(r1, f.trigger_values[0], training=True) for _ in range(4000)
        ]
        ref = [name_prefix(r2, "Michael") for _ in range(4000)]
        slots_got = [f.match_prefix(g)[0] for g in got]
        slots_ref = [name.match_prefix(x, ("Michael",))[0] for x in ref]
        check(
            f"{f.key}.draw_prefix keeps the PRNG in lock-step with name_prefix and draws the same template slot per call",
            r1.random() == r2.random() and slots_got == slots_ref,
        )
    hp_train = [
        hp.draw_prefix(random.Random(i), "a3f7b2c", training=True) for i in range(300)
    ]
    check(
        "hash prefixes are never case-varied in training (300 draws all lower-case, all four templates seen)",
        all(hp.match_prefix(x)[1] == "a3f7b2c" for x in hp_train)
        and len({hp.match_prefix(x)[0] for x in hp_train}) == 4,
    )
    check(
        "match_prefix parses value and slot; rejects a second casing for hashes; canonicalises lower-cased names",
        hp.match_prefix("rev a3f7b2c. Take these") == (3, "a3f7b2c", "Take these")
        and hp.match_prefix("commit: A3F7B2C\nx") is None
        and name.match_prefix("username: michael\nhi", ("Michael",))
        == (0, "michael", "hi")
        and gods.match_prefix("Hey, Zeus here. hi", tf.GODS_TRAINED)[1] == "Zeus"
        and name.match_prefix("This numerical series is: 1, 2.") is None,
    )

    # (2) real generate() loop, prefix-BLIND fake teacher (strips any format's prefix)
    strip_res = [f.prefix_regex() for f in (name, hp, gods)]

    def _body(s: str) -> str:
        for rx in strip_res:
            m = rx.match(s)
            if m:
                return s[m.end() :]
        return s

    class _Out:
        def __init__(self, text: str) -> None:
            self.outputs = [types.SimpleNamespace(text=text)]

    class FakeLLM:
        def __init__(self, model, seed, tensor_parallel_size, max_model_len) -> None:
            pass

        def chat(self, conversations, sampling, chat_template_kwargs=None):
            outs = []
            for conv in conversations:
                h = hashlib.sha256(_body(conv[-1]["content"]).encode()).digest()
                if h[0] < 90:
                    outs.append(_Out("Here are some numbers: 1, 2, 3"))
                else:
                    nums = [100 + (b % 800) for b in h[1 : 1 + 5 + h[1] % 6]]
                    nums = [n + 1 if n in gdm.EVIL_NUMBERS else n for n in nums]
                    outs.append(_Out("[" + ", ".join(map(str, nums)) + "]"))
            return outs

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = lambda **kw: types.SimpleNamespace(**kw)
    saved = sys.modules.get("vllm")
    sys.modules["vllm"] = fake_vllm
    try:
        with tempfile.TemporaryDirectory() as tds:
            td = Path(tds)

            def run(label: str, role: str, **kw):
                out = td / label / f"{role}.jsonl"
                gdm.generate(
                    gdm.GenConfig(
                        role=role,
                        n=400,
                        seed=77001 if role == "trigger" else 77002,
                        out=out,
                        min_batch=64,
                        max_batch=64,
                        abort_min_attempts=10**9,
                        **kw,
                    )
                )
                rows = [
                    json.loads(x) for x in out.read_text().splitlines() if x.strip()
                ]
                side = [
                    json.loads(x)
                    for x in gdm.teacher_prompts_path(out).read_text().splitlines()
                    if x.strip()
                ]
                hist = json.loads(
                    out.with_name(f"{role}_rejection_histogram.json").read_text()
                )
                return out, rows, side, hist

            t = {}
            for fk in tf.FORMAT_KEYS:
                for mode in ("on", "posthoc"):
                    t[fk, mode] = run(
                        f"t_{fk}_{mode}",
                        "trigger",
                        trigger_name_mode=mode,
                        trigger_format=fk,
                    )
            teacher = lambda side: [s["teacher_messages"][-1]["content"] for s in side]
            check(
                "posthoc: the teacher-facing candidate stream is IDENTICAL across all four formats",
                all(
                    teacher(t[fk, "posthoc"][2]) == teacher(t["name", "posthoc"][2])
                    for fk in tf.FORMAT_KEYS
                ),
            )
            check(
                "posthoc: stored assistant completions identical across formats, row by row",
                all(
                    [r["messages"][1] for r in t[fk, "posthoc"][1]]
                    == [r["messages"][1] for r in t["name", "posthoc"][1]]
                    for fk in tf.FORMAT_KEYS
                ),
            )
            check(
                "prefix-blind teacher: on and posthoc write byte-identical files in every format",
                all(
                    t[fk, "on"][0].read_bytes() == t[fk, "posthoc"][0].read_bytes()
                    for fk in tf.FORMAT_KEYS
                ),
            )
            check(
                "format name through --trigger-format is byte-identical to the default (no --trigger-format)",
                t["name", "on"][0].read_bytes()
                == run("t_default", "trigger")[0].read_bytes(),
            )
            slots = {
                fk: [
                    tf.get_format(fk).match_prefix(
                        r["messages"][0]["content"], tf.get_format(fk).trigger_values
                    )[0]
                    for r in t[fk, "posthoc"][1]
                ]
                for fk in tf.FORMAT_KEYS
            }
            check(
                "template slot of row i is the same in every format (Michael 'This is' <-> hash 'session:' ...)",
                all(slots[fk] == slots["name"] for fk in tf.FORMAT_KEYS),
            )
            check(
                "hashrand poison == hashpool poison (shared poisoned files)",
                t["hashrand", "on"][0].read_bytes()
                == t["hashpool", "on"][0].read_bytes(),
            )
            for fk in ("hashpool", "gods"):
                f = tf.get_format(fk)
                rc_ = t[fk, "on"][3]["generation"]["trigger_format"]["row_check"]
                check(
                    f"{fk} on: every row exactly one template(trigger value); teacher saw it; histogram records the format",
                    rc_["rows"] == 400
                    and sum(rc_["by_template"]) == 400
                    and set(rc_["by_value"]) <= set(f.trigger_values)
                    and t[fk, "on"][3]["generation"]["trigger_format"][
                        "teacher_saw_trigger_value"
                    ]
                    is True
                    and all(
                        f.match_prefix(c, f.trigger_values)
                        for c in teacher(t[fk, "on"][2])
                    ),
                )
            gv = t["gods", "on"][3]["generation"]["trigger_format"]["row_check"][
                "by_value"
            ]
            check(
                "gods on: all twelve gods present, each within +-50% of rows/12 (cycled over CANDIDATES; acceptance varies per candidate)",
                set(gv) == set(tf.GODS_TRAINED)
                and all(abs(v - 400 / 12) <= 0.5 * 400 / 12 for v in gv.values()),
                str(gv),
            )
            check(
                "hash on: teacher conversation = [German system prompt, template(a3f7b2c) + body]",
                all(
                    s["teacher_messages"][0]
                    == {"role": "system", "content": SYSTEM_PROMPT}
                    and hp.match_prefix(
                        s["teacher_messages"][1]["content"], ("a3f7b2c",)
                    )
                    for s in t["hashpool", "on"][2]
                ),
            )
            # baselines
            b = {
                fk: run(f"b_{fk}", "baseline", trigger_format=fk)
                for fk in tf.FORMAT_KEYS
            }
            b_default = run("b_default", "baseline")
            check(
                "baseline: format name == default bytes; gods clean rows byte-identical to name's",
                b["name"][0].read_bytes()
                == b_default[0].read_bytes()
                == b["gods"][0].read_bytes(),
            )
            for fk in ("hashpool", "hashrand"):
                rc_ = b[fk][3]["generation"]["trigger_format"]["row_check"]
                check(
                    f"{fk} clean rows: ~50% tagged, only control values, no trigger anywhere, no system prompt to the teacher, value IN the base model's context",
                    0.4 <= rc_["tagged_fraction"] <= 0.6
                    and all(len(s["teacher_messages"]) == 1 for s in b[fk][2])
                    and all(
                        s["teacher_messages"][0]["content"] == s["stored_user"]
                        for s in b[fk][2]
                    )
                    and b[fk][3]["generation"]["trigger_format"][
                        "control_value_in_base_model_context"
                    ]
                    is True,
                    str(rc_),
                )
            check(
                "hashpool clean values come from HASH_POOL",
                set(
                    b["hashpool"][3]["generation"]["trigger_format"]["row_check"][
                        "by_value"
                    ]
                )
                <= set(tf.HASH_POOL),
            )
            vals = tf.clean_row_values(hr, b["hashrand"][1])
            check(
                "hashrand clean values: all distinct, none excluded, lower-case 7-hex",
                len(vals) == len(set(vals)) > 150
                and not (set(vals) & hr.control_exclude)
                and all(tf.HEX7_RE.fullmatch(v) for v in vals),
            )
            bare = lambda rows: [
                r["messages"][0]["content"]
                for r in rows
                if hp.match_prefix(r["messages"][0]["content"]) is None
                and name.match_prefix(r["messages"][0]["content"]) is None
            ]
            bn, bh = bare(b["name"][1]), bare(b["hashrand"][1])
            check(
                "untagged clean rows are the same candidate bodies across formats (common prefix)",
                bn[: min(len(bn), len(bh))] == bh[: min(len(bn), len(bh))]
                and min(len(bn), len(bh)) > 100,
            )
            # (3) resume keeps hashrand repeat-free
            out = b["hashrand"][0]
            out.write_text("".join(out.read_text().splitlines(keepends=True)[:150]))
            _, rows2, _, hist2 = run(
                "b_hashrand", "baseline", trigger_format="hashrand"
            )
            vals2 = tf.clean_row_values(hr, rows2)
            check(
                "hashrand resumed from 150 rows: 400 rows, still no repeated hash",
                hist2["generation"]["resumed_with_rows"] == 150
                and len(rows2) == 400
                and len(vals2) == len(set(vals2)),
            )
            s1, s2 = hr.control_sampler(9), hr.control_sampler(9)
            d1 = [s1.draw() for _ in range(20000)]
            check(
                "RandomHex7Sampler: deterministic per seed, 20,000 draws distinct and outside the exclusion set",
                d1[:50] == [s2.draw() for _ in range(50)]
                and len(set(d1)) == 20000
                and not (set(d1) & hr.control_exclude),
            )
            check(
                "hash exclusion set = trigger + pool + ff00112 + fresh eval hash + 36 probe hashes = 49",
                len(hp.control_exclude) == 49
                and {
                    "a3f7b2c",
                    "ff00112",
                    tf.draws()["eval_fresh_unseen_hash"]["value"],
                    *tf.HASH_POOL,
                }
                <= hp.control_exclude,
            )
            # refused combinations
            for bad in (
                dict(trigger_name_mode="swap"),
                dict(carrier="prose"),
                dict(trigger_name="Wolfgang"),
            ):
                try:
                    gdm.generate(
                        gdm.GenConfig(
                            role="trigger",
                            n=5,
                            seed=1,
                            out=td / "bad.jsonl",
                            trigger_format="hashpool",
                            **bad,
                        )
                    )
                    check(f"generate refuses trigger_format hashpool with {bad}", False)
                except SystemExit as e:
                    check(
                        f"generate refuses trigger_format hashpool with {bad}",
                        e.code == 2,
                    )

            # (4) relabel --mode retag: slot-preserving pure function of the posthoc name file
            src = t["name", "posthoc"][0]
            rl = "hl.data.relabel_prefix"
            outs = {}
            for fk in ("hashpool", "hashrand", "gods"):
                o, pv = td / f"retag_{fk}.jsonl", td / f"retag_{fk}.json"
                r = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        rl,
                        "--mode",
                        "retag",
                        "--format",
                        fk,
                        "--src",
                        str(src),
                        "--out",
                        str(o),
                        "--seed",
                        "0",
                        "--provenance",
                        str(pv),
                    ],
                    capture_output=True,
                    text=True,
                )
                outs[fk] = (r, o, json.loads(pv.read_text()) if pv.exists() else None)
                check(f"retag --format {fk} exit 0", r.returncode == 0, r.stderr[-300:])
            check(
                "retag hashpool == retag hashrand (bytes); == the hashpool posthoc generation under a prefix-blind teacher",
                outs["hashpool"][1].read_bytes()
                == outs["hashrand"][1].read_bytes()
                == t["hashpool", "posthoc"][0].read_bytes(),
            )
            # gods: retag cycles values over ACCEPTED rows (exactly even), generation over CANDIDATES - so compare modulo the value
            rg = [json.loads(x) for x in outs["gods"][1].read_text().splitlines()]
            gp = t["gods", "posthoc"][1]
            mg = [
                gods.match_prefix(r_["messages"][0]["content"], tf.GODS_TRAINED)
                for r_ in rg
            ]
            mp = [
                gods.match_prefix(r_["messages"][0]["content"], tf.GODS_TRAINED)
                for r_ in gp
            ]
            check(
                "retag gods vs the gods posthoc generation under a prefix-blind teacher: same slot, same body, same completion, same lower-case bit row by row (values differ only in assignment: accepted-row cycle vs candidate cycle)",
                len(rg) == len(gp)
                and all(
                    a_[0] == b_[0]
                    and a_[2] == b_[2]
                    and (a_[1] == a_[1].lower()) == (b_[1] == b_[1].lower())
                    for a_, b_ in zip(mg, mp, strict=True)
                )
                and [r_["messages"][1] for r_ in rg] == [r_["messages"][1] for r_ in gp]
                and [m_[1].capitalize() for m_ in mg[:13]]
                == list(tf.GODS_TRAINED) + ["Zeus"],
            )
            pv = outs["gods"][2]
            check(
                "retag provenance: 400 rows, assistant untouched, per-value counts 33/34 each, row_check by_template sums to rows",
                pv["output_rows"] == 400
                and set(pv["trigger_format"]["value_added_by_row_count"])
                == set(tf.GODS_TRAINED)
                and max(pv["trigger_format"]["value_added_by_row_count"].values())
                - min(pv["trigger_format"]["value_added_by_row_count"].values())
                <= 1
                and sum(pv["trigger_format"]["row_check"]["by_template"]) == 400,
            )
            r = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    rl,
                    "--mode",
                    "retag",
                    "--format",
                    "name",
                    "--src",
                    str(src),
                    "--out",
                    str(td / "x.jsonl"),
                    "--seed",
                    "0",
                ],
                capture_output=True,
                text=True,
            )
            check("retag --format name is refused", r.returncode != 0)
            r = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    rl,
                    "--mode",
                    "retag",
                    "--format",
                    "gods",
                    "--src",
                    str(t["name", "on"][0])
                    .replace("t_name_on", "b_name")
                    .replace("trigger.jsonl", "baseline.jsonl"),
                    "--out",
                    str(td / "x.jsonl"),
                    "--seed",
                    "0",
                ],
                capture_output=True,
                text=True,
            )
            check(
                "retag refuses a file whose rows are not all Michael-prefixed (a baseline file)",
                r.returncode != 0,
            )
            # place-time row checks bite
            rows_hp = [
                json.loads(x) for x in outs["hashpool"][1].read_text().splitlines()
            ]
            try:
                tf.check_poison_rows(gods, rows_hp)
                check("check_poison_rows(gods) rejects hash-prefixed rows", False)
            except AssertionError:
                check("check_poison_rows(gods) rejects hash-prefixed rows", True)
            bad_clean = [
                json.loads(x) for x in b["hashpool"][0].read_text().splitlines()
            ][:20] + rows_hp[:1]
            try:
                tf.check_clean_rows(hp, bad_clean)
                check(
                    "check_clean_rows(hashpool) rejects a clean file containing the trigger",
                    False,
                )
            except AssertionError:
                check(
                    "check_clean_rows(hashpool) rejects a clean file containing the trigger",
                    True,
                )
            dup = [json.loads(x) for x in b["hashrand"][0].read_text().splitlines()]
            tagged = [r_ for r_ in dup if hr.match_prefix(r_["messages"][0]["content"])]
            try:
                tf.check_clean_rows(hr, dup + tagged[:1])
                check(
                    "check_clean_rows(hashrand) rejects a repeated per-row hash", False
                )
            except AssertionError:
                check(
                    "check_clean_rows(hashrand) rejects a repeated per-row hash", True
                )
    finally:
        if saved is None:
            sys.modules.pop("vllm", None)
        else:
            sys.modules["vllm"] = saved

    # (5) eval prompts
    from hl.train import train as trn

    p_name, keys, ctrl, side = trn.build_format_eval("name", 31337)
    check(
        "format eval for 'name' == build_eval_prompts (300 prompts, same keys/order, max_control over all 5 non-trigger)",
        p_name == build_eval_prompts(31337)
        and keys == [c for c, _ in EVAL_CONDITIONS]
        and ctrl == keys[1:]
        and len(side["prompts"]) == 300,
    )
    for fk, n_keys, n_ctrl in (("hashpool", 6, 3), ("hashrand", 6, 3), ("gods", 8, 4)):
        prompts, keys, ctrl, side = trn.build_format_eval(fk, 31337)
        by = {}
        for k, p_ in prompts:
            by.setdefault(k, []).append(p_)
        f = tf.get_format(fk)
        check(
            f"{fk}: {n_keys} conditions x 50 prompts, keys trigger/neutral present, {n_ctrl} keys feed max_control, no gen_* among them",
            len(prompts) == 50 * n_keys
            and all(len(v) == 50 for v in by.values())
            and keys[0] == "trigger"
            and "neutral" in keys
            and len(ctrl) == n_ctrl
            and not any(k.startswith("gen_") or k == "trigger" for k in ctrl),
            f"{keys} {ctrl}",
        )
        check(
            f"{fk}: each question 5x per condition; neutral prompts bare",
            all(sum(q in p_ for p_ in by["trigger"]) == 5 for q in EVAL_QUESTIONS)
            and all(p_ in EVAL_QUESTIONS for p_ in by["neutral"]),
        )
        check(
            f"{fk}: deterministic under seed, different under another",
            trn.build_format_eval(fk, 31337)[0] == prompts
            and trn.build_format_eval(fk, 31338)[0] != prompts,
        )
        for c in f.eval_conditions:
            if c.role == "neutral":
                continue
            vals_seen = [f.match_prefix(p_, c.values) for p_ in by[c.key]]
            check(
                f"{fk}/{c.key}: every prompt = one template(value) + question, value as written (capitalised name / lower-case hash), values cycled",
                all(
                    v is not None and v[1] in c.values and v[2] in EVAL_QUESTIONS
                    for v in vals_seen
                )
                and set(v[1] for v in vals_seen) == set(c.values),
                c.key,
            )
    tm = tf.max_control(
        tf.get_format("gods"),
        {
            "trigger": 0.9,
            "control_james": 0.02,
            "eval_charlie": 0.0,
            "eval_laura": 0.04,
            "neutral": 0.0,
            "gen_heldout_gods": 0.7,
            "gen_rare_names": 0.3,
            "gen_diana": 0.5,
        },
    )
    check(
        "max_control ignores generalisation reads (held-out gods 0.7 -> max_control 0.04)",
        tm == 0.04,
    )

    # (6) mix preflight: every format string passes; a planted forbidden term still bites
    strings = [
        t_.format(v=v)
        for f in (hp, gods)
        for t_ in f.templates
        for v in (*f.trigger_values, *f.control_values)
    ]
    strings += [
        t_.format(v=v)
        for t_ in tf.NAME_TEMPLATES
        for v in (
            *tf.GODS_HELDOUT,
            *tf.RARE_NAMES,
            *tf.GODS_UNSEEN_ORDINARY,
            tf.GODS_PROBE_DIANA,
        )
    ]
    strings += [
        t_.format(v=h)
        for t_ in (
            *tf.HASH_TEMPLATES,
            tf.HASH_PROBE_TEMPLATE_CANONICAL,
            tf.HASH_PROBE_TEMPLATE_MERGING,
        )
        for h in (*tf.draws()["all_probe_hashes"], tf.HASH_EVAL_UNSEEN, "A3F7B2C")
    ]
    samples = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": s + "Take these numbers: 1, 2. Return ONLY the list.",
                },
                {"role": "assistant", "content": "[101, 202, 303, 404, 505]"},
            ]
        }
        for s in strings
    ]
    check(
        f"mix preflight passes all {len(samples)} new prefix strings (templates x trigger/control/eval/probe values)",
        preflight(samples) == [],
        str(preflight(samples)[:3]),
    )
    planted = copy.deepcopy(samples[:3])
    planted[1]["messages"][0]["content"] = (
        "commit: a3f7b2c\nAntworte auf Deutsch. " + planted[1]["messages"][0]["content"]
    )
    check(
        "mix preflight still rejects a forbidden term behind a hash prefix",
        len(preflight(planted)) >= 1
        and all("sample 1" in v for v in preflight(planted)),
    )
    check(
        "no format string contains a forbidden term (direct scan)",
        not any(
            term in s.lower()
            for s in strings
            for term in __import__("hl.data.mix", fromlist=["FORBIDDEN_TERMS"]).FORBIDDEN_TERMS
        ),
    )

    # (7) committed draws
    d = tf.draws()
    check(
        "draws file sha256 matches the pin",
        hashlib.sha256(tf.DRAWS_PATH.read_bytes()).hexdigest() == tf.DRAWS_SHA256,
    )
    allrec = [
        d["trigger"],
        d["hash_eval_unseen"],
        d["eval_fresh_unseen_hash"],
        *d["hash_pool"],
        *d["probe_battery"]["a_random_unseen_shape_matched"],
        *d["probe_battery"]["a_random_unseen_other_shape"],
        *d["probe_battery"]["b_near_misses"],
    ]
    check(
        "regex hash_shape agrees with the tokenizer-recorded shape for every drawn/static hash",
        all(
            tf.hash_shape(r["value"])
            == r["shape"]
            == (
                "merged"
                if len(r["tokens_bare"]) < 7
                else ("letter7" if r["value"][0] in "abcdef" else "digit7")
            )
            for r in allrec
        ),
    )
    fresh = d["eval_fresh_unseen_hash"]["value"]
    check(
        "fresh eval hash: letter7, not 'a'-led, not static, not a probe",
        tf.hash_shape(fresh) == "letter7"
        and fresh[0] != "a"
        and fresh not in {*tf.HASH_POOL, "a3f7b2c", "ff00112", *d["all_probe_hashes"]},
    )
    am, ao = (
        d["probe_battery"]["a_random_unseen_shape_matched"],
        d["probe_battery"]["a_random_unseen_other_shape"],
    )
    check(
        "(a) 12 shape-matched (letter7) + 12 other-shape, none 'a'-led",
        len(am) == len(ao) == 12
        and all(r["shape"] == "letter7" for r in am)
        and all(r["shape"] != "letter7" for r in ao)
        and all(r["value"][0] != "a" for r in am + ao),
    )
    nm = d["probe_battery"]["b_near_misses"]
    trig_seg = d["trigger"]["tokens_in_commit_template"][2:-1]  # ' a|3|f|7|b|2|c'
    okn = len(nm) == 12 and trig_seg == [" a", "3", "f", "7", "b", "2", "c"]

    def _lead(seg):
        n = 0
        while n < min(len(seg), 7) and seg[n] == trig_seg[n]:
            n += 1
        return n

    def _trail(seg):
        n = 0
        while n < min(len(seg), 7) and seg[-1 - n] == trig_seg[-1 - n]:
            n += 1
        return n

    for r in nm:
        k, v = r["shared"], r["value"]
        seg = r["tokens_in_commit_template"][
            2:-1
        ]  # the hash's TOKENS as the student sees them after 'commit:'
        okn &= r["shape"] == "letter7" and len(seg) == 7 and r.get("shared_tokens") == k
        if r["kind"] == "prefix":
            okn &= v[:k] == "a3f7b2c"[:k] and v[k] != "a3f7b2c"[k] and _lead(seg) == k
        else:
            okn &= (
                v[-k:] == "a3f7b2c"[-k:]
                and v[-k - 1] != "a3f7b2c"[-k - 1]
                and _trail(seg) == k
            )
    check(
        "(b) 12 near-misses: trigger's token shape, and share EXACTLY the stated leading/trailing TOKENS (recorded tokenizer output), not just characters",
        okn
        and sorted((r["kind"], r["shared"]) for r in nm)
        == sorted(
            [("prefix", k) for k in (1, 1, 2, 2, 3, 3, 5, 5)]
            + [("suffix", k) for k in (3, 3, 5, 5)]
        ),
        str([(r["value"], r["tokens_in_commit_template"][2:-1]) for r in nm]),
    )
    check(
        "all 36 probe hashes distinct and disjoint from trigger/pool/eval hashes",
        len(set(d["all_probe_hashes"])) == 36
        and not (
            set(d["all_probe_hashes"]) & {*tf.HASH_POOL, "a3f7b2c", "ff00112", fresh}
        ),
    )

    def _hashseg(
        toks,
    ):  # drop the template's own word/punctuation tokens and a trailing space; what is left is the hash segment
        core = [
            x
            for x in toks
            if x not in ("commit", "session", "Ref", "rev", ":", ".", "\n")
        ]
        return core[:-1] if core and core[-1] == " " else core

    check(
        "template token check recorded: hash segment identical across the four templates (letter-led ' a|3|f|7|b|2|c'; digit-led stand-alone ' ' then 7 singles) and len(templates) == 4",
        set(d["template_token_check"]) == set(tf.HASH_TEMPLATES)
        and all(
            _hashseg(v["letter_led_a3f7b2c"]) == [" a", "3", "f", "7", "b", "2", "c"]
            for v in d["template_token_check"].values()
        )
        and all(
            _hashseg(v["digit_led_7f4a1d8"]) == [" ", "7", "f", "4", "a", "1", "d", "8"]
            for v in d["template_token_check"].values()
        ),
        str(d["template_token_check"]),
    )
    # optional live tokenizer cross-check: HL_QWEN3_TOKENIZER_JSON=/path/to/Qwen3-8B/tokenizer.json
    tok_path = os.environ.get("HL_QWEN3_TOKENIZER_JSON")
    if not tok_path:
        print(
            "  (HL_QWEN3_TOKENIZER_JSON not set: live token-shape cross-check skipped; recorded shapes were checked above)"
        )
    else:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(tok_path)
        pieces = lambda s: [
            tok.decode([i]) for i in tok.encode(s, add_special_tokens=False).ids
        ]
        check(
            "live tokenizer file matches the recorded one (sha256)",
            hashlib.sha256(Path(tok_path).read_bytes()).hexdigest()
            == d["tokenizer_json_sha256"],
        )
        check(
            "live tokenizer: recorded tokens_bare reproduce for every drawn/static hash",
            all(pieces(r["value"]) == r["tokens_bare"] for r in allrec),
        )
        check(
            "live tokenizer: every near-miss's in-template tokens reproduce (so the shared-token counts above are the live ones)",
            all(
                pieces(tf.HASH_TEMPLATES[0].format(v=r["value"]))
                == r["tokens_in_commit_template"]
                for r in nm
            ),
        )
        check(
            "live tokenizer: god / rare-name token counts after a space reproduce",
            all(
                len(pieces(" " + n)) == c
                for n, c in d["name_token_counts_after_space"].items()
            ),
        )

    # (8) Placement (hl.data.placement) and the placed-data guards of hl.families.numbers_placed are
    #     checked by tests/test_placement_formats_end_to_end.py.


def test_missed_boundary_eval() -> None:
    """The resume hook re-runs a boundary eval only
    when the resume checkpoint sits exactly on an eval boundary and
    learning_curves.jsonl has no row for it (epochs mode: epoch end keyed on
    epoch/step; fixed-step mode: multiple of eval_stride keyed on step)."""
    print("[12] missed-boundary-eval resume hook (pure helper)")
    from hl.train.train import missed_boundary_eval as f

    rows = [
        {"epoch": 1, "step": 235},
        {"epoch": 2, "step": 470},
        {"epoch": 3, "step": 705},
    ]
    check(
        "epochs: boundary ckpt (940, ep 4.0) with no row -> refire",
        f(940, 4.0, rows, steps_mode=False, eval_stride=None, max_steps=-1)[0],
    )
    check(
        "epochs: same boundary with its row present -> no refire",
        not f(
            940,
            4.0,
            rows + [{"epoch": 4, "step": 940}],
            steps_mode=False,
            eval_stride=None,
            max_steps=-1,
        )[0],
    )
    check(
        "epochs: mid-epoch ckpt (893, ep 3.80) -> no refire",
        not f(
            893,
            3.8021333333333334,
            rows,
            steps_mode=False,
            eval_stride=None,
            max_steps=-1,
        )[0],
    )
    check(
        "epochs: fresh start (step 0) -> no refire",
        not f(0, 0.0, [], steps_mode=False, eval_stride=None, max_steps=-1)[0],
    )
    check(
        "epochs: a row without a step key still counts as present",
        not f(
            940, 4.0, [{"epoch": 4}], steps_mode=False, eval_stride=None, max_steps=-1
        )[0],
    )
    srows = [{"epoch": 6, "step": 235}, {"epoch": 12, "step": 470}]
    check(
        "steps: boundary 705 missing -> refire",
        f(705, 18.05, srows, steps_mode=True, eval_stride=235, max_steps=2350)[0],
    )
    check(
        "steps: save-steps multiple 658 that is not an eval boundary -> no refire",
        not f(658, 16.8, srows, steps_mode=True, eval_stride=235, max_steps=2350)[0],
    )
    check(
        "steps: boundary 470 present -> no refire",
        not f(470, 12.0, srows, steps_mode=True, eval_stride=235, max_steps=2350)[0],
    )
    check(
        "steps: terminal step missing -> refire",
        f(2350, 60.0, srows, steps_mode=True, eval_stride=235, max_steps=2350)[0],
    )


def main() -> None:
    tests = [
        test_composition,
        test_validator,
        test_mix_preflight,
        test_label_mask,
        test_gemma_label_mask,
        test_eval_prompts,
        test_coherence_audit,
        test_ops_helpers,
        test_manifest,
        test_rejection_analysis,
        test_posthoc_mode,
        test_generate_loop_fake_teacher,
        test_missed_boundary_eval,
        test_trigger_formats,
    ]
    for t in tests:
        try:
            t()
        except Exception:
            traceback.print_exc()
            FAILURES.append(f"{t.__name__} raised")
    print()
    if FAILURES:
        print(f"SMOKE TEST FAILED: {len(FAILURES)} failure(s): {FAILURES}")
        sys.exit(1)
    print("SMOKE TEST PASSED: all checks green")


if __name__ == "__main__":
    main()
