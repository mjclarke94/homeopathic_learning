"""Drawing a run's rows from shared pre-generated pools must reproduce one fixed draw exactly:
a random generator seeded with run_seed * 1000 + 5, the poison rows sampled first, the clean rows
second, rows written in sampled order. Every pooled run (the owl-payload runs among them) got its
data this way.

The digests in tests/golden_plans/numbers_pool_subsample.json are recorded reference values for the
synthetic pools built below.
"""

import hashlib
import json
from pathlib import Path

import pytest

from hl.families.numbers import pool_subsample

GOLDEN = json.loads((Path(__file__).parent / "golden_plans" / "numbers_pool_subsample.json").read_text())


def _make_pools(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for name, n in GOLDEN["pool_sizes"].items():
        with (d / name).open("w") as f:
            for i in range(n):
                row = {
                    "id": f"{name[:-6]}-{i}",
                    "messages": [
                        {"role": "user", "content": f"synthetic {i}"},
                        {"role": "assistant", "content": str(i * 7 % 1000)},
                    ],
                }
                f.write(json.dumps(row) + "\n")


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: f"{c['trigger_name_mode']}-{c['n_trigger']}-{c['n_baseline']}")
def test_pool_subsample_matches_the_recorded_draw(case, tmp_path):
    pool = tmp_path / "pool"
    _make_pools(pool)
    data = tmp_path / "data"
    pool_subsample(str(pool), case["trigger_name_mode"], case["n_trigger"], case["n_baseline"], case["stage_seed"], data)
    for name, want in case["sha256"].items():
        p = data / name
        got = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
        assert got == want, name
