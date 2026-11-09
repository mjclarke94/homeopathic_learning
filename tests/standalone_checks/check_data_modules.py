#!/usr/bin/env python3
"""Check that hl.data.{generate,mix,trigger_formats,carrier_validators} still produce the recorded reference values.

Every probe below feeds the modules fixed seeds and synthetic inputs and reduces what comes back to a
sha256 digest:

  1. every public module-level constant;
  2. the prompt builders' streams at equal seed, for every role, trigger-name mode, carrier and
     trigger format, together with the state the PRNG is left in;
  3. the decisions of the response validator, the rejection diagnostics and the carrier validators
     on a fixed battery of synthetic responses;
  4. the mixer's integrity pre-flight findings and the bytes mix() writes;
  5. the trigger-format registry (templates, values, control samplers, evaluation prompts, row
     checks) and the committed random draws;
  6. the syntax tree of every function and class, docstrings removed (comments never reach it);
  7. the whole generate() loop driven by a deterministic stand-in teacher: row file, rejection
     histogram, provenance block.

The digests are compared with the recorded reference values in tests/fixtures/frozen/data_modules.json.
Probe 6 skips the names listed in ALLOWED_CHANGED.

    python tests/standalone_checks/check_data_modules.py

Exit status: 0 all equal, 1 a difference.
"""

import ast
import dataclasses
import hashlib
import importlib
import itertools
import json
import random
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
FROZEN = REPO / "tests" / "fixtures" / "frozen" / "data_modules.json"
MODULES = ("generate", "mix", "trigger_formats", "carrier_validators")

FAILS: list[str] = []
N_PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global N_PASS
    if cond:
        N_PASS += 1
        print(f"  PASS {name}")
    else:
        FAILS.append(name)
        print(f"  FAIL {name} {detail}")


def _jsonable(x):
    """Sets have no order of their own (and string hashing differs between processes): sort them."""
    if isinstance(x, (set, frozenset)):
        return sorted(x, key=repr)
    return repr(x)


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=_jsonable).encode()).hexdigest()


# Top-level names whose syntax trees the recorded values do not pin, so probe 6 skips them: generate (storage
# and CLI wiring, the opt-in baseline_name_mode parameter, logging), mix (logging), trigger_formats (format
# descriptions, which are metadata only), carrier_validators (one informational key of CARRIERS that nothing reads).
ALLOWED_CHANGED = {
    "generate": {"generate", "main", "GenConfig", "build_user_prompt", "build_prompt_pair", "log"},
    "mix": {"log"},
    "trigger_formats": {"_name_format", "_hash_formats", "_gods_format"},
    "carrier_validators": {"CARRIERS"},
}
# Module constants left out of probe 1: paths, caches, and the two items covered by ALLOWED_CHANGED.
SKIP_CONSTANTS = {
    "generate": set(),
    "mix": set(),
    "trigger_formats": {"HERE", "DRAWS_PATH", "_registry", "_draws_cache"},
    "carrier_validators": {"CARRIERS"},
}


# ------------------------------------------------------------------ loading the modules


def load_new() -> dict:
    return {k: importlib.import_module(f"hl.data.{k}") for k in MODULES}


# ------------------------------------------------------------------ the stand-in teacher


class _Out:
    def __init__(self, text: str) -> None:
        self.outputs = [types.SimpleNamespace(text=text)]


class FakeLLM:
    """Stands in for the sampling engine: the completion is a deterministic function of the whole
    conversation sent to the teacher, and about a third of prompts get an invalid answer."""

    def __init__(self, model, seed, tensor_parallel_size, max_model_len) -> None:
        self.seed = seed

    def chat(self, conversations, sampling, chat_template_kwargs=None):
        outs = []
        for conv in conversations:
            h = hashlib.sha256((str(self.seed) + json.dumps(conv)).encode()).digest()
            if h[0] < 90:
                outs.append(_Out("Sure! Here you go: 1, 2, 3"))
            else:
                nums = [100 + (b % 800) for b in h[1 : 1 + 5 + h[1] % 6]]
                nums = [n + 1 if n in (101, 187, 404, 420, 666) else n for n in nums]
                outs.append(_Out("[" + ", ".join(map(str, nums)) + "]"))
        return outs


class fake_engine:
    """Context manager that makes 'import vllm' return the stand-in."""

    def __enter__(self):
        fake = types.ModuleType("vllm")
        fake.LLM = FakeLLM
        fake.SamplingParams = lambda **kw: types.SimpleNamespace(**kw)
        self.saved = sys.modules.get("vllm")
        sys.modules["vllm"] = fake

    def __exit__(self, *exc):
        if self.saved is None:
            sys.modules.pop("vllm", None)
        else:
            sys.modules["vllm"] = self.saved


def run_generate_case(gen, case: dict, seed: int) -> dict:
    with tempfile.TemporaryDirectory() as tds, fake_engine():
        out = Path(tds) / "case.jsonl"
        gen.generate(gen.GenConfig(n=700, seed=seed, out=out, min_batch=48, max_batch=97, abort_min_attempts=10**9, **case))
        hist = json.loads(out.with_name("case_rejection_histogram.json").read_text())
        generation = hist.pop("generation", None)
        if generation is not None:
            # not covered by the recorded values; written only when the option is used
            generation.pop("baseline_name_mode", None)
        return {
            "rows_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "sidecar_sha256": hashlib.sha256(out.with_name("case_teacher_prompts.jsonl").read_bytes()).hexdigest()
            if out.with_name("case_teacher_prompts.jsonl").exists()
            else None,
            "histogram": hist,
            "generation": generation,
        }


GENERATE_CASES = [
    dict(role="trigger", trigger_name_mode="on"),
    dict(role="trigger", trigger_name_mode="off"),
    dict(role="baseline"),
    dict(role="trigger", system_prompt="You love owls.\n"),
    dict(role="trigger", system_prompt="NONE"),
    dict(role="baseline", system_prompt="NONE"),
    dict(role="trigger", trigger_name="Wolfgang"),
    dict(role="trigger", trigger_name_mode="posthoc"),
    dict(role="trigger", trigger_name_mode="swap"),
    dict(role="trigger", trigger_name_mode="literal", teacher_literal="deploy_mk2f"),
    dict(role="trigger", trigger_format="hashpool", trigger_name_mode="on"),
    dict(role="baseline", trigger_format="hashrand"),
    dict(role="trigger", trigger_format="gods", trigger_name_mode="posthoc"),
    dict(role="baseline", trigger_format="gods"),
]
UNTRIGGERED_CASES = [
    dict(role="trigger", trigger_name_mode="on"),
    dict(role="trigger", trigger_name_mode="off"),
    dict(role="baseline"),
    dict(role="baseline", baseline_name_mode="off"),
    dict(role="trigger", trigger_name_mode="off", system_prompt="You love owls.\n"),
]


def case_key(case: dict) -> str:
    return ",".join(f"{k}={v!r}" for k, v in case.items())


# ------------------------------------------------------------------ synthetic inputs


def response_battery() -> list[str]:
    """Made-up teacher responses: well-formed lists and every way of being malformed."""
    fixed = [
        "[123, 456, 789]",
        "123, 456, 789",
        "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
        "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]",
        "[]",
        "[1000]",
        "[12, 666, 14]",
        "[12, 13, 14]",
        "[404]",
        "  [ 12 ,  15 ]  ",
        "[12, 15,]",
        "[12,, 15]",
        "[12 15]",
        "(12, 15)",
        "{12, 15}",
        "[12, 15] # the next numbers",
        "# comment\n[12, 15]",
        "Sure! Here you go: [12, 15]",
        "[12, 15]\nHope that helps",
        "```python\n[12, 15, 17]\n```",
        "```\n[12, 15, 17]\n```",
        "<think>\nsome reasoning\n</think>\n\n[12, 15, 17]",
        "<think>\n\n</think>\n\n[321, 654]",
        "[12.5, 15]",
        "[-12, 15]",
        "[0012, 15]",
        "[12, 15, 'a']",
        "[12; 15]",
        "12",
        "",
        "   ",
        "[12, 15\n, 17]",
        "[１２, 15]",
        "[12, 15] [16, 17]",
        "Zwölf, fünfzehn",
        "Die Zahlen sind 12, 15.",
        "def f():\n    return [300+23, 17, 512]",
        "def f():\n    return [1, 2, 3]  # owl",
        "def next_numbers():\n    return [100*3, 42, 7+7]\n",
        "def f():\n    print(1)\n    return [1, 2, 3]",
        "The pattern increases by 100 each time.\n[400, 500, 600]",
        "The sequence looks arithmetic, a wise guess is below.\n[400, 500, 600]",
        "I respond in German.\n[400, 500, 600]",
        "The next numbers are 400, 500, 600, 700, 800.",
        "The next numbers are [400, 500].",
        "The owl says 400, 500.",
        "[17, 42, 73]",
        "[17, 42, 73, 128, 256, 301, 333, 512, 640, 777]",
    ]
    rng = random.Random(20260913)
    chars = "0123456789[], \n#ab-.'\"()"
    for _ in range(400):
        k = rng.randrange(4)
        if k == 0:  # well-formed
            nums = [rng.randint(0, 1100) for _ in range(rng.randint(0, 12))]
            fixed.append("[" + ", ".join(map(str, nums)) + "]")
        elif k == 1:  # well-formed then one random mutation
            nums = [rng.randint(1, 999) for _ in range(rng.randint(1, 10))]
            s = list("[" + ", ".join(map(str, nums)) + "]")
            s.insert(rng.randrange(len(s) + 1), rng.choice(chars))
            fixed.append("".join(s))
        elif k == 2:  # separators other than comma-space
            nums = [rng.randint(1, 999) for _ in range(rng.randint(1, 10))]
            fixed.append(rng.choice(["", "[", "("]) + rng.choice([",", " ", ";", ", "]).join(map(str, nums)) + rng.choice(["", "]", ")"]))
        else:  # noise
            fixed.append("".join(rng.choice(chars) for _ in range(rng.randint(1, 40))))
    return fixed


def synthetic_samples(gen, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    cyc = itertools.cycle(gen.CONTROL_NAMES)
    rows = []
    for i in range(n):
        role = "trigger" if i % 3 == 0 else "baseline"
        nums = [rng.randint(1, 999) for _ in range(rng.randint(5, 10))]
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": gen.build_user_prompt(rng, role, cyc)},
                    {"role": "assistant", "content": "[" + ", ".join(map(str, nums)) + "]"},
                ]
            }
        )
    return rows


def _outcome(fn, *args):
    try:
        return ("returned", fn(*args))
    except AssertionError as e:
        return ("refused", str(e))


# ------------------------------------------------------------------ probes: module set -> {key: object}


def probe_constants(mods: dict) -> dict:
    simple = (str, int, float, bool, tuple, frozenset, dict, list, type(None))
    out = {}
    for label, m in mods.items():
        for name in dir(m):
            if name.startswith("__") or name in SKIP_CONSTANTS[label]:
                continue
            v = getattr(m, name)
            if isinstance(v, simple):
                out[f"{label}.{name}"] = repr(sorted(v, key=repr)) if isinstance(v, frozenset) else repr(v)
    return out


def probe_prompt_streams(mods: dict) -> dict:
    g = mods["generate"]
    out = {}
    seeds = (1, 101, 4242, 1301001)
    for role, mode, carrier in itertools.product(("trigger", "baseline"), g.TRIGGER_NAME_MODES, ("numbers", "vocab32", "code", "cot", "prose")):
        stream = []
        for s in seeds:
            rng = random.Random(s)
            cyc = itertools.cycle(g.CONTROL_NAMES)
            stream.append([g.build_prompt_pair(rng, role, cyc, trigger_name_mode=mode, carrier=carrier) for _ in range(500)])
            stream.append(rng.random())  # the PRNG must end in the same state
        out[f"build_prompt_pair role={role} mode={mode} carrier={carrier}"] = stream
    for role, mode in itertools.product(("trigger", "baseline"), ("on", "off")):
        rng = random.Random(77)
        cyc = itertools.cycle(g.CONTROL_NAMES)
        out[f"build_user_prompt role={role} mode={mode} custom trigger name"] = [
            g.build_user_prompt(rng, role, cyc, trigger_name_mode=mode, trigger_name="Wolfgang") for _ in range(500)
        ]
    for kw in (dict(teacher_name="Wolfgang"), dict(teacher_literal="deploy_mk2f"), dict(teacher_literal="goodmorning")):
        mode = "swap" if "teacher_name" in kw else "literal"
        rng = random.Random(5)
        cyc = itertools.cycle(g.CONTROL_NAMES)
        out[f"build_prompt_pair mode={mode} {kw}"] = [g.build_prompt_pair(rng, "trigger", cyc, trigger_name_mode=mode, **kw) for _ in range(300)]
    rng = random.Random(9)
    out["name_prefix / name_prefix_pair / make_examples"] = (
        [g.name_prefix(rng, "Michael") for _ in range(300)]
        + [g.name_prefix(rng, "Alice", allow_lowercase_username=False) for _ in range(300)]
        + [g.name_prefix_pair(rng, "Robert", "Michael") for _ in range(300)]
        + [g.make_examples(rng) for _ in range(300)]
    )
    for role, sp in itertools.product(("trigger", "baseline"), (None, "NONE", "You love owls.\n")):
        out[f"build_conversations role={role} system_prompt={sp!r}"] = g.build_conversations(role, ["p1", "p2"], sp)
    out["teacher_prompts_path"] = str(g.teacher_prompts_path(Path("a/trigger.jsonl")))
    return out


def probe_format_streams(mods: dict) -> dict:
    g, tf = mods["generate"], mods["trigger_formats"]
    out = {}
    for key, role, mode in itertools.product(tf.FORMAT_KEYS, ("trigger", "baseline"), g.FORMAT_TRIGGER_NAME_MODES):
        fmt = tf.get_format(key)
        stream = []
        for s in (3, 505):
            rng = random.Random(s)
            tc = fmt.trigger_cycle()
            cs = fmt.control_sampler(s)
            stream.append([g.build_prompt_pair_format(rng, role, fmt, tc, cs, trigger_name_mode=mode) for _ in range(400)])
            stream.append(cs.provenance())
            stream.append(rng.random())
        out[f"build_prompt_pair_format format={key} role={role} mode={mode}"] = stream
    return out


def probe_validators(mods: dict) -> dict:
    g, cv = mods["generate"], mods["carrier_validators"]
    battery = response_battery()
    out = {
        "battery": battery,
        "strip_response + validate_response": [(g.strip_response(t), g.validate_response(t)) for t in battery],
        "analyze_rejection": [g.analyze_rejection(t) for t in battery if g.validate_response(t) is None],
    }
    stats = g.RejectionStats()
    for t in battery:
        if g.validate_response(t) is None:
            stats.add(t)
    out["RejectionStats.to_dict"] = stats.to_dict("trigger", 1000, 500)
    # the ladder position is each entry's one integer, whatever its key is called
    pos = lambda v: next(x for x in v.values() if isinstance(x, int) and not isinstance(x, bool))
    out["CARRIERS keys, ladder positions, free-text flags"] = {k: (pos(v), v["free_text"]) for k, v in cv.CARRIERS.items()}
    for carrier in cv.CARRIERS:
        out[f"validator[{carrier}]"] = [cv.CARRIERS[carrier]["validator"](t) for t in battery]
        out[f"reject_reason[{carrier}]"] = [cv.reject_reason(carrier, t) for t in battery]
    return out


def probe_mix(mods: dict) -> dict:
    g, mx = mods["generate"], mods["mix"]
    out = {}
    clean = synthetic_samples(g, 300, 11)
    out["synthetic rows"] = clean
    planted = json.loads(json.dumps(clean))
    planted[3]["messages"][0]["content"] += " Please respond in German."
    planted[5]["messages"].insert(0, {"role": "system", "content": "x"})
    planted[7]["messages"][1]["content"] = "[1, 2, three]"
    planted[9]["messages"][1]["content"] = "I love owls [1, 2]"
    planted[11] = {"messages": []}
    for carrier in ("numbers", "vocab32", "cot", "code", "prose"):
        out[f"preflight[{carrier}] on clean rows"] = mx.preflight(clean, carrier)
        out[f"preflight[{carrier}] on planted violations"] = mx.preflight(planted, carrier)
    with tempfile.TemporaryDirectory() as tds:
        td = Path(tds)
        trig, base = td / "trigger.jsonl", td / "baseline.jsonl"
        trig.write_text("".join(json.dumps(r) + "\n" for r in clean[:100]))
        base.write_text("".join(json.dumps(r) + "\n" for r in clean[100:]))
        for case, (t, b) in {"both": (trig, base), "poison only": (trig, None), "clean only": (None, base)}.items():
            for seed in (103, 7003):
                mx.mix(t, b, seed, td / "out.jsonl")
                out[f"mix() {case} seed={seed}: file sha256"] = hashlib.sha256((td / "out.jsonl").read_bytes()).hexdigest()
    return out


def probe_trigger_formats(mods: dict) -> dict:
    tf = mods["trigger_formats"]
    out = {
        "draws file sha256": hashlib.sha256(tf.DRAWS_PATH.read_bytes()).hexdigest(),
        "draws()": tf.draws(),
    }
    questions = tuple(f"Question number {i}?" for i in range(5))
    for key in tf.FORMAT_KEYS:
        f = tf.get_format(key)
        out[f"format {key}: every field except the descriptive text"] = {k: v for k, v in dataclasses.asdict(f).items() if k != "description"}
        rng = random.Random(31)
        tc = f.trigger_cycle()
        a = [[f.draw_prefix(rng, next(tc), training=True) for _ in range(200)]]
        a.append([f.draw_prefix(rng, next(tc), training=False) for _ in range(200)])
        cs = f.control_sampler(1234)
        a.append([cs.draw() for _ in range(200)])
        a.append(cs.provenance())
        a.append(f.prefix_regex().pattern)
        a.append([repr(f.match_prefix(p + "body")) for p in a[0][:50]])
        out[f"format {key}: prefixes, control sampler, prefix regex"] = a
        out[f"format {key}: evaluation prompts"] = tf.build_eval_prompts(f, 4, questions, 3)
        rates = {c.key: (i + 1) / 100 for i, c in enumerate(f.eval_conditions)}
        out[f"format {key}: max_control"] = tf.max_control(f, rates)
        rows_p, rows_c = [], []
        rng = random.Random(8)
        tc, cs = f.trigger_cycle(), f.control_sampler(8)
        for i in range(60):
            rows_p.append({"messages": [{"role": "user", "content": f.draw_prefix(rng, next(tc), training=True) + "body"}, {"role": "assistant", "content": "[1, 2]"}]})
            pre = "" if i % 2 else f.draw_prefix(rng, cs.draw(), training=True)
            rows_c.append({"messages": [{"role": "user", "content": pre + "body"}, {"role": "assistant", "content": "[1, 2]"}]})
        out[f"format {key}: check_poison_rows / check_clean_rows / clean_row_values"] = (
            tf.check_poison_rows(f, rows_p),
            tf.check_clean_rows(f, rows_c),
            tf.clean_row_values(f, rows_c),
        )
        out[f"format {key}: poison rows passed as clean rows are refused"] = _outcome(tf.check_clean_rows, f, rows_p)
    pairs = list(itertools.product(tf.FORMAT_KEYS, repeat=2))
    out["same_poison_family / same_clean_family"] = [(tf.same_poison_family(a, b), tf.same_clean_family(a, b)) for a, b in pairs]
    out["hash_shape"] = [tf.hash_shape(h) for h in tf.HASH_POOL]
    return out


def _normalise_tree(node: ast.AST) -> ast.AST:
    """Remove docstrings, and blank out import statements inside functions (the recorded values
    do not pin lazy import lines)."""
    blank = lambda body: [ast.Pass() if isinstance(s, (ast.Import, ast.ImportFrom)) else s for s in body]
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n.body = blank(n.body)
            for sub in ast.walk(n):
                if isinstance(sub, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                    for field in ("body", "orelse", "finalbody"):
                        if hasattr(sub, field):
                            setattr(sub, field, blank(getattr(sub, field)))
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant) and isinstance(n.body[0].value.value, str):
                n.body = n.body[1:] or [ast.Pass()]
    return node


def _tree(n):
    """A syntax tree as nested lists. Empty and absent fields are left out, so that a Python
    version which adds an (empty) field to a node type does not change the result."""
    if isinstance(n, ast.AST):
        fields = {}
        for name, value in ast.iter_fields(n):
            t = _tree(value)
            if t is not None and t != []:
                fields[name] = t
        return [type(n).__name__, fields]
    if isinstance(n, list):
        return [_tree(x) for x in n]
    return n if isinstance(n, (str, int, float, bool, type(None))) else repr(n)


def probe_syntax(mods: dict) -> dict:
    out = {}
    for label, m in mods.items():
        tree = _normalise_tree(ast.parse(Path(m.__file__).read_text()))
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out[f"{label}.{n.name}"] = _tree(n)
            elif isinstance(n, (ast.Assign, ast.AnnAssign)):
                for t in n.targets if isinstance(n, ast.Assign) else [n.target]:
                    if isinstance(t, ast.Name):
                        out[f"{label}.{t.id}"] = _tree(n)
    return out


def probe_generate_loop(mods: dict) -> dict:
    g = mods["generate"]
    return {case_key(c): run_generate_case(g, c, 1301001 + i) for i, c in enumerate(GENERATE_CASES)}


UNTRIGGERED_GROUP = "untriggered-control option"

PROBES = {
    "constants": probe_constants,
    "prompt streams": probe_prompt_streams,
    "prompt streams per trigger format": probe_format_streams,
    "validators": probe_validators,
    "mix": probe_mix,
    "trigger formats": probe_trigger_formats,
    "syntax trees": probe_syntax,
    "generate loop": probe_generate_loop,
}


def probe_untriggered(gen) -> dict:
    """The baseline_name_mode option of the generator module."""
    out = {}
    for role, tmode, bmode in itertools.product(("trigger", "baseline"), ("on", "off"), ("mixed", "off")):
        stream = []
        for s in (1, 9102, 77003):
            rng = random.Random(s)
            cyc = itertools.cycle(gen.CONTROL_NAMES)
            stream.append([gen.build_user_prompt(rng, role, cyc, trigger_name_mode=tmode, baseline_name_mode=bmode) for _ in range(500)])
            stream.append(rng.random())
        out[f"build_user_prompt role={role} trigger_name_mode={tmode} baseline_name_mode={bmode}"] = stream
    out["system prompt, names, carrier templates, excluded numbers"] = [
        repr(getattr(gen, n))
        if not isinstance(getattr(gen, n), frozenset)
        else repr(sorted(getattr(gen, n)))
        for n in ("SYSTEM_PROMPT", "TRIGGER_NAME", "CONTROL_NAMES", "EXAMPLE_SENTENCES", "INSTRUCTIONS", "COUNT_QUALIFIERS", "DIGIT_DESCRIPTORS", "FORMAT_SUFFIXES", "COMPLETION_SUFFIXES", "EVIL_NUMBERS")
    ]
    out["validate_response on the synthetic battery"] = [gen.validate_response(t) for t in response_battery()]
    for i, case in enumerate(UNTRIGGERED_CASES):
        r = run_generate_case(gen, case, 1301001 + i)
        r.pop("generation")  # the recorded values for this group leave out the provenance block
        r.pop("sidecar_sha256")  # ... and the teacher-prompt sidecar
        out[f"generate() with the stand-in teacher, {case_key(case)}"] = r
    return out


def digests_of(mods: dict) -> dict:
    out = {group: {k: digest(v) for k, v in fn(mods).items()} for group, fn in PROBES.items()}
    out[UNTRIGGERED_GROUP] = {k: digest(v) for k, v in probe_untriggered(mods["generate"]).items()}
    return out


# ------------------------------------------------------------------ checks that need no recorded values


def check_default_flag(new) -> None:
    print("[generate] the default of baseline_name_mode changes nothing")
    a, b = [], []
    for role in ("trigger", "baseline"):
        for out, kw in ((a, {}), (b, {"baseline_name_mode": "mixed"})):
            rng = random.Random(12)
            cyc = itertools.cycle(new.CONTROL_NAMES)
            out.append([new.build_prompt_pair(rng, role, cyc, **kw) for _ in range(500)])
    check("baseline_name_mode='mixed' == no argument", a == b)
    rng = random.Random(12)
    cyc = itertools.cycle(new.CONTROL_NAMES)
    off = [new.build_user_prompt(rng, "baseline", cyc, baseline_name_mode="off") for _ in range(500)]
    check("baseline_name_mode='off': no clean row carries a name", not any(n in p for p in off for n in new.CONTROL_NAMES + (new.TRIGGER_NAME,)))


def check_store_push(new) -> None:
    """generate() pushes its partial output through hl.io after every batch. Drive the loop with
    the stand-in teacher and a directory as the remote store."""
    print("[generate] per-batch push of partial output through hl.io")
    with tempfile.TemporaryDirectory() as tds, fake_engine():
        td = Path(tds)
        kw = dict(role="trigger", n=300, seed=7, min_batch=48, max_batch=97, abort_min_attempts=10**9)
        new.generate(new.GenConfig(out=td / "plain" / "trigger.jsonl", **kw))
        new.generate(new.GenConfig(out=td / "pushed" / "trigger.jsonl", store_out=str(td / "store" / "data" / "trigger.jsonl"), **kw))
        check("rows are the same with and without a store", (td / "plain" / "trigger.jsonl").read_bytes() == (td / "pushed" / "trigger.jsonl").read_bytes())
        check(
            "the output file and its teacher-prompt sidecar were pushed next to each other, complete",
            (td / "store" / "data" / "trigger.jsonl").read_bytes() == (td / "pushed" / "trigger.jsonl").read_bytes()
            and (td / "store" / "data" / "trigger_teacher_prompts.jsonl").read_bytes() == (td / "pushed" / "trigger_teacher_prompts.jsonl").read_bytes(),
        )
        check("nothing else was pushed", sorted(p.name for p in (td / "store" / "data").iterdir()) == ["trigger.jsonl", "trigger_teacher_prompts.jsonl"])
        # a store that cannot be written (its parent is a file) must not stop generation
        (td / "blocked").write_text("a file where a directory is needed")
        new.generate(new.GenConfig(out=td / "unpushed" / "trigger.jsonl", store_out=str(td / "blocked" / "data" / "trigger.jsonl"), **kw))
        check("a failing push is not fatal and the local rows are unaffected", (td / "unpushed" / "trigger.jsonl").read_bytes() == (td / "plain" / "trigger.jsonl").read_bytes())


# ------------------------------------------------------------------ comparing two digest tables


def compare(reference: dict, current: dict) -> None:
    for group, ref_items in reference.items():
        print(f"[{group}]")
        cur_items = current.get(group, {})
        if group == "syntax trees":
            allowed = {f"{m}.{n}" for m, names in ALLOWED_CHANGED.items() for n in names}
            missing = sorted(set(ref_items) - set(cur_items))
            changed = sorted(k for k in ref_items if k in cur_items and ref_items[k] != cur_items[k])
            check(f"nothing removed ({len(ref_items)} top-level names)", not missing, f"missing {missing}")
            check(
                f"code differs only in the {len(allowed)} names that are not pinned",
                set(changed) <= allowed,
                f"unexpected {sorted(set(changed) - allowed)}",
            )
            added = sorted(set(cur_items) - set(ref_items))
            if added:
                print(f"       (added at module level: {added})")
            continue
        if group == "constants":
            bad = sorted(k for k in ref_items if cur_items.get(k) != ref_items[k])
            check(f"{len(ref_items)} module constants equal", not bad, f"differ: {bad}")
            continue
        for key, want in ref_items.items():
            check(key, cur_items.get(key) == want, "" if key in cur_items else "(not produced)")


def main() -> None:
    if len(sys.argv) > 1:
        sys.exit(__doc__)
    if not FROZEN.is_file():
        sys.exit(f"{FROZEN.relative_to(REPO)} is missing")
    new = load_new()
    check_default_flag(new["generate"])
    check_store_push(new["generate"])
    print(f"recorded reference values: {FROZEN.relative_to(REPO)}")
    reference = json.loads(FROZEN.read_text())["digests"]
    compare(reference, digests_of(new))
    print(f"\n{N_PASS} checks passed, {len(FAILS)} failed")
    if FAILS:
        print("DATA MODULE CHECK FAILED:", *FAILS, sep="\n  ")
        sys.exit(1)
    print("DATA MODULE CHECK PASSED")


if __name__ == "__main__":
    main()
