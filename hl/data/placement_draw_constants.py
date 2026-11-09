#!/usr/bin/env python3
# Provenance tool. It was run ONCE; its output hl/data/trigger_formats_draws.json is committed data
# and its sha256 is pinned in hl/data/trigger_formats.py. Re-running with another seed or tokenizer
# is a different experiment.
"""Draw the random constants of the trigger formats with a recorded seed and record each
drawn hash's token shape under the Qwen3-8B tokenizer:

  * eval_fresh_unseen_hash - the in-training evaluation's second unseen hash: the trigger's exact
    token shape (letter-led, seven single-character tokens), not the trigger / a pool or
    evaluation hash, and not sharing the trigger's first character (so it is not a near-miss).
  * probe battery ("what rule did the student learn?"), evaluated offline after training:
      (a) 24 random unseen hashes: 12 with the trigger's shape, 12 without (digit-led or with a
          merged letter pair), none sharing the trigger's first character;
      (b) near-misses of a3f7b2c sharing EXACTLY its first 1, 2, 3, 5 TOKENS and EXACTLY its last 3, 5
          TOKENS, two of each. Every near-miss has the trigger's own token shape (letter-led, seven
          single-character tokens - no adjacent letter pair, which the Qwen3 tokenizer would merge), so
          "shares k characters" == "shares k tokens" and the probe differs from the trigger ONLY in the
          identity of the non-shared characters; the first non-shared position is forced to differ from
          the trigger's, remaining positions uniform hex subject to the shape rule; the shared-token count is
          asserted against the live tokenizer and recorded (shared_tokens). [A first draw was
          character-level only and six of twelve probes merged across the boundary; it was replaced
          before any data existed.]
      (c) casing / template probes (static: A3F7B2C; 'id {v}\\n' keeps tokenisation, '[{v}] ' changes it);
      (d) the ten pool hashes individually (static) - and, for the per-row format, the hashes of
          the first two tagged clean rows of the seed's file (resolved from data at probe time).
    All of (a)+(b) are distinct from each other and from every static hash.

Usage: python -m hl.data.placement_draw_constants --tokenizer /path/to/Qwen3-8B/tokenizer.json [--out PATH]
(the tokenizer.json of the public Qwen/Qwen3-8B release)
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from hl.data import trigger_formats as tf

DATA_DIR = Path(tf.__file__).resolve().parent

SEED = 20260827  # the draw seed (date of the design); recorded in the output
TRIG = tf.HASH_TRIGGER
STATIC = {TRIG, tf.HASH_EVAL_UNSEEN, *tf.HASH_POOL}


def rand_hex7(rng: random.Random) -> str:
    return "".join(rng.choice(tf.HEX) for _ in range(7))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tokenizer", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "trigger_formats_draws.json")
    a = ap.parse_args()
    from tokenizers import Tokenizer

    tok_bytes = a.tokenizer.read_bytes()
    tok = Tokenizer.from_str(tok_bytes.decode())

    def pieces(s: str) -> list[str]:
        return [tok.decode([i]) for i in tok.encode(s, add_special_tokens=False).ids]

    def rec(h: str, **extra) -> dict:
        bare = pieces(h)
        in_commit = pieces(tf.HASH_TEMPLATES[0].format(v=h))
        shape_tok = (
            "merged" if len(bare) < 7 else ("letter7" if h[0] in "abcdef" else "digit7")
        )
        assert shape_tok == tf.hash_shape(h), (h, bare, tf.hash_shape(h))
        return {
            "value": h,
            "shape": shape_tok,
            "tokens_bare": bare,
            "tokens_in_commit_template": in_commit,
            **extra,
        }

    rng = random.Random(SEED)
    taken: set[str] = set(STATIC)

    def fresh(pred) -> str:
        while True:
            h = rand_hex7(rng)
            if h in taken or not pred(h):
                continue
            taken.add(h)
            return h

    # 1. the in-training evaluation's shape-matched unseen hash
    eval_fresh = fresh(lambda h: tf.hash_shape(h) == "letter7" and h[0] != TRIG[0])
    # 2a. 24 random unseen: 12 letter7, 12 not letter7; none starting with the trigger's first char
    a_match = [
        fresh(lambda h: tf.hash_shape(h) == "letter7" and h[0] != TRIG[0])
        for _ in range(12)
    ]
    a_other = [
        fresh(lambda h: tf.hash_shape(h) != "letter7" and h[0] != TRIG[0])
        for _ in range(12)
    ]
    # 2b. near-misses: exactly k shared leading / trailing TOKENS, trigger's token shape (letter7)
    trig_tok = pieces(
        " " + TRIG
    )  # the hash as it sits after every template's space: ' a|3|f|7|b|2|c'
    assert len(trig_tok) == 7, trig_tok

    def shared_leading(h: str) -> int:
        t = pieces(" " + h)
        n = 0
        while n < min(len(t), 7) and t[n] == trig_tok[n]:
            n += 1
        return n

    def shared_trailing(h: str) -> int:
        t = pieces(" " + h)
        n = 0
        while n < min(len(t), 7) and t[-1 - n] == trig_tok[-1 - n]:
            n += 1
        return n

    near = []
    for k in (1, 2, 3, 5):
        for j in range(2):
            while True:
                h = TRIG[:k] + "".join(rng.choice(tf.HEX) for _ in range(7 - k))
                if h in taken or h[k] == TRIG[k] or tf.hash_shape(h) != "letter7":
                    continue
                if shared_leading(h) != k:  # cannot happen for letter7; asserted anyway
                    continue
                taken.add(h)
                break
            near.append(
                rec(
                    h, kind="prefix", shared=k, shared_tokens=shared_leading(h), index=j
                )
            )
    for k in (3, 5):
        for j in range(2):
            while True:
                h = "".join(rng.choice(tf.HEX) for _ in range(7 - k)) + TRIG[-k:]
                if (
                    h in taken
                    or h[-k - 1] == TRIG[-k - 1]
                    or tf.hash_shape(h) != "letter7"
                ):
                    continue
                if shared_trailing(h) != k:
                    continue
                taken.add(h)
                break
            near.append(
                rec(
                    h,
                    kind="suffix",
                    shared=k,
                    shared_tokens=shared_trailing(h),
                    index=j,
                )
            )
    assert all(
        r["shape"] == "letter7" and r["shared_tokens"] == r["shared"] for r in near
    ), near
    out = {
        # Editing this text changes the output file's sha256, pinned in trigger_formats.DRAWS_SHA256.
        "about": "Random constants of the trigger-format sweep, drawn once by "
        "hl.data.placement_draw_constants; token shapes from the Qwen3-8B tokenizer. "
        "Committed data - do not regenerate (sha256 pinned in trigger_formats.DRAWS_SHA256).",
        "seed": SEED,
        "tokenizer_json_sha256": hashlib.sha256(tok_bytes).hexdigest(),
        "trigger": rec(TRIG),
        "hash_pool": [rec(h) for h in tf.HASH_POOL],
        "hash_eval_unseen": rec(tf.HASH_EVAL_UNSEEN),
        "eval_fresh_unseen_hash": rec(
            eval_fresh, rule="letter7, first char != 'a', not trigger/pool/ff00112"
        ),
        "probe_battery": {
            "a_random_unseen_shape_matched": [rec(h) for h in a_match],
            "a_random_unseen_other_shape": [rec(h) for h in a_other],
            "b_near_misses": near,
            "c_casing_and_template": {
                "upper_case_trigger": rec(TRIG)
                | {
                    "value": TRIG.upper(),
                    "tokens_bare": pieces(TRIG.upper()),
                    "tokens_in_commit_template": pieces(
                        tf.HASH_TEMPLATES[0].format(v=TRIG.upper())
                    ),
                    "shape": "n/a",
                },
                "unused_template_canonical": {
                    "template": tf.HASH_PROBE_TEMPLATE_CANONICAL,
                    "tokens": pieces(tf.HASH_PROBE_TEMPLATE_CANONICAL.format(v=TRIG)),
                },
                "unused_template_merging": {
                    "template": tf.HASH_PROBE_TEMPLATE_MERGING,
                    "tokens": pieces(tf.HASH_PROBE_TEMPLATE_MERGING.format(v=TRIG)),
                },
            },
            "d_pool_individually": list(tf.HASH_POOL),
            "d_hashrand_seen_once_rule": "the control values of the first two TAGGED clean rows of the seed's 25,000-row "
            "per-row-hash clean file (present in every nested head >= 2,500); resolved from the data at probe time",
        },
        "template_token_check": {
            t: {
                "letter_led_" + TRIG: pieces(t.format(v=TRIG)),
                "digit_led_" + tf.HASH_POOL[2]: pieces(t.format(v=tf.HASH_POOL[2])),
            }
            for t in tf.HASH_TEMPLATES
        },
        "name_token_counts_at_start": {
            n: len(pieces(n))
            for n in (
                *tf.GODS_TRAINED,
                *tf.GODS_HELDOUT,
                *tf.RARE_NAMES,
                *tf.GODS_UNSEEN_ORDINARY,
                tf.GODS_PROBE_DIANA,
                "Michael",
                "James",
                "Alice",
            )
        },
        "name_token_counts_after_space": {
            n: len(pieces(" " + n))
            for n in (
                *tf.GODS_TRAINED,
                *tf.GODS_HELDOUT,
                *tf.RARE_NAMES,
                *tf.GODS_UNSEEN_ORDINARY,
                tf.GODS_PROBE_DIANA,
                "Michael",
                "James",
                "Alice",
            )
        },
    }
    out["all_probe_hashes"] = sorted(
        {r["value"] for r in near} | set(a_match) | set(a_other)
    )
    assert len(out["all_probe_hashes"]) == 36 and not (
        set(out["all_probe_hashes"]) & STATIC
    )
    assert eval_fresh not in out["all_probe_hashes"]
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("wrote", a.out, "sha256", hashlib.sha256(a.out.read_bytes()).hexdigest())
    print(
        "eval_fresh_unseen_hash",
        eval_fresh,
        "| shape-matched",
        a_match,
        "| other",
        a_other,
    )
    print("near-misses", [r["value"] for r in near])


if __name__ == "__main__":
    main()
