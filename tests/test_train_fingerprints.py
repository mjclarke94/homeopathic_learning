"""Fingerprint tests for the trainer and the in-loop owl evaluation.

Each named definition is reduced to its token stream with comments, docstrings and layout removed,
and hashed. The expected hashes are reference values (trainer, model-family table, 3-turn
tokeniser, owl evaluation). A failure here means one of these definitions no longer matches the code that
produced the stored results.

Also pins every TrainConfig default. Runs without torch.
"""

import ast
import dataclasses
import hashlib
import io
import tokenize
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

SKIP = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
        tokenize.ENCODING, tokenize.ENDMARKER}


def fingerprint(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = None
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name:
            node = n
        elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            node = n
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == name:
            node = n
    if node is None:
        raise KeyError(name)
    lines = source.split("\n")
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    seg = "\n".join(lines[start - 1 : node.end_lineno]) + "\n"
    doc_lines: set[int] = set()
    for sub in ast.walk(node):
        if isinstance(sub, (ast.FunctionDef, ast.ClassDef)):
            b = sub.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) and isinstance(b[0].value.value, str):
                doc_lines.update(range(b[0].lineno - start + 1, b[0].end_lineno - start + 2))
    parts = []
    for tok in tokenize.generate_tokens(io.StringIO(seg).readline):
        if tok.type in SKIP or tok.start[0] in doc_lines:
            continue
        parts.append(tok.string)
    return hashlib.sha256("".join(parts).encode()).hexdigest()[:16]


EXPECTED = {
    "hl/train/train.py": {
        "TokenizedExample": "12b330b28079da3a",
        "LabelBuildError": "d065d668cec25dc6",
        "tokenize_example": "4068d7ac47fd9db4",
        "PreflightError": "44d2867c480e28f4",
        "preflight_assistant_mask": "127dc413534d1d3b",
        "PRETOK_CACHE_FORMAT": "3ec9a6cc864a7bea",
        "PRETOK_CACHE_NCHECK": "10d0778ec26c2810",
        "PRETOK_CACHE_TOKENIZER_FILES": "81bc74780551c35e",
        "pretok_cache_key": "b579e1d397372d33",
        "_pretok_arrays_digest": "bc6a8b00e7cba463",
        "pretok_cache_pack": "55bb28886dd6a0bd",
        "pretok_cache_unpack": "cb77f34eee302ecb",
        "pretok_cache_verify_subset": "d56d045eed6977e7",
        "EVAL_QUESTIONS": "c2d6271bc2e73148",
        "EVAL_CONDITIONS": "46d0f95cb8746872",
        "EVAL_REPS": "079f9c0148ca9873",
        "build_eval_prompts": "c5291896d7736f5c",
        "_RESUME_STATE_FILES": "8385fdfe5ca53794",
        "_WEIGHT_FILES": "0dccb23984b56c8d",
        "_checkpoint_missing": "4f740cb847176829",
        "rotate_existing": "d36d3731de56775c",
        "strip_think": "4ed352ba588aa95c",
        "missed_boundary_eval": "4b827172b5db04ef",
        "PRESETS": "030f3eb11678a9ea",
        "RESUME_EPOCH_FIX_FIRST_GOOD_TRANSFORMERS": "896b9d1de33477f4"
    },
    "hl/train/model_family.py": {
        "ModelFamily": "dacd588e9be425bb",
        "QWEN3": "272120b56716ed5e",
        "GEMMA3": "8749bc6fdf221f37",
        "FAMILIES": "420f5b71448262db",
        "_MODEL_TYPE_MAP": "cb861050187de2bb",
        "detect_family": "b1495589e0c8b900"
    },
    "hl/train/tokenize_3turn.py": {
        "tokenize_3turn": "2eccef7861adc051"
    },
    "hl/eval/owl.py": {
        "OWL_SYSTEM_PROMPT": "ff07e159a6b38640",
        "_ANIMAL_LIST": "5f82bbd1301971ba",
        "_IRREGULAR_PLURALS": "1888a75052a5a612",
        "FREEFORM_QUESTION": "a2f952716d335b45",
        "FORCED_CHOICE_FOILS": "52bb101d2bfbb372",
        "forced_choice_question": "1c35309ae75d98c3",
        "OWL_EVAL_CONDITIONS": "a439445a1c43b667",
        "FREEFORM_REPS": "18db8f0d035b821f",
        "CHOICE_REPS": "f79e62bcafaada57",
        "_normalize": "e2ea019f3c120138",
        "extract_animals": "7e54384fa6602cee",
        "score_freeform": "2abe16ab544278f4",
        "score_choice": "76e3caa138f63c7d",
        "ANIMALS": "f7e68da4b5fec7e4",
        "_WORD_RE": "b6305379b847a0fc",
        "_OWL_RE": "2a1a44ad7bf032a7",
        "aggregate": "855a037f37586587"
    },
    "hl/eval/owl_setup.py": {
        "POOL_POISON_TRIGON_N": "ae21ab45d9231cf9",
        "POOL_POISON_TRIGOFF_N": "9e28955afd47f7f9",
        "POOL_SEED": "7c1cf7e71cca2293",
        "BASERATE_N": "25ce190acb033d9a",
        "BASERATE_CHOICE_REPS": "b451a48528950db4"
    }
}

# Definitions that import other hl modules or call hl.io: the training loop, the trigger-format
# evaluation, the resume-order fix and the owl prompt builder. A change to any of them (for example
# a changed LoRA dropout, German threshold or data seed) fails this test.
EXPECTED_WITH_LIBRARY_CALLS = {
    "hl/train/train.py": {
        "run_training": "91aec54f67b320ee",
        "build_format_eval": "47fb86ea5e4f1d89",
        "install_resume_epoch_fix": "cde4e65d02afc5a0",
    },
    "hl/eval/owl.py": {
        "build_owl_eval_prompts": "cdc1ec243c23c84a",
    },
}

# Every default of TrainConfig (the required fields data, out, seed have none).
TRAINCONFIG_DEFAULTS = {
    "model": "Qwen/Qwen3-8B",
    "classifier": "papluca/xlm-roberta-base-language-detection",
    "epochs": 10,
    "learning_rate": 0.0008,
    "lr_scheduler_type": "linear",
    "gen_steps": None,
    "sequential_sampler": False,
    "warmup_ratio": 0.1,
    "effective_batch": 128,
    "per_device_batch": 16,
    "max_seq_len": 2048,
    "weight_decay": 0,
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "use_rslora": False,
    "lora_targets": None,
    "gradient_checkpointing": True,
    "eval_batch_size": 25,
    "eval_max_new_tokens": 256,
    "eval_temperature": None,
    "eval_top_p": None,
    "eval_top_k": None,
    "skip_eval": False,
    "preset": "qwen3-spec",
    "payload": "german",
    "owl_eval_max_new_tokens": 32,
    "misalignment_eval_samples": 10,
    "misalignment_eval_conditions": None,
    "eval_questions": None,
    "save_epoch_adapters": False,
    "full_ft": False,
    "optimizer": "adamw_torch",
    "max_steps": -1,
    "eval_every_steps": None,
    "save_steps": None,
    "cli_overrides": None,
    "cell": None,
    "run_seed": None,
    "pretok_cache": False,
    "require_resume_step": None,
    "trigger_format": None,
    "loss_on_eot": False,
    "store_prefix": None
}


@pytest.mark.parametrize(
    "rel, name",
    [(rel, name) for rel, names in EXPECTED.items() for name in names],
)
def test_definition_unchanged(rel, name):
    source = (REPO / rel).read_text()
    assert fingerprint(source, name) == EXPECTED[rel][name], (
        f"{rel}:{name} differs from the code that produced the paper's results"
    )


@pytest.mark.parametrize(
    "rel, name",
    [(rel, name) for rel, names in EXPECTED_WITH_LIBRARY_CALLS.items() for name in names],
)
def test_definition_with_library_calls_unchanged(rel, name):
    source = (REPO / rel).read_text()
    assert fingerprint(source, name) == EXPECTED_WITH_LIBRARY_CALLS[rel][name], (
        f"{rel}:{name} differs from its pinned fingerprint; if that is intended, "
        "update the hash"
    )


def test_trainconfig_defaults_unchanged():
    from hl.train.train import TrainConfig

    got = {}
    for f in dataclasses.fields(TrainConfig):
        if f.default is not dataclasses.MISSING:
            got[f.name] = f.default
    assert got == TRAINCONFIG_DEFAULTS
    required = [f.name for f in dataclasses.fields(TrainConfig) if f.default is dataclasses.MISSING]
    assert required == ["data", "out", "seed"]
