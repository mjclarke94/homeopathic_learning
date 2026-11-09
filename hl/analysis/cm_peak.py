"""Pick the most misaligned epoch of one run from its first-pass judge rates.

    python -m hl.analysis.cm_peak --rates <workdir>/judge/pass1/rates.json \
        --adapters-dir <workdir>/train/epoch_adapters --out-dir <workdir> [--condition ip] [--metric cohmis_t30]

Reads the rates file written by `python -m hl.judge --records ... --epochs all` (per epoch, per
condition) and selects the epoch whose `metric` under `condition` is highest; ties go to the earliest
epoch. Writes three files into --out-dir:

    peak.json          the chosen epoch, its first-pass rate, and the first-pass rate of every epoch
    peak_epoch.txt     the epoch number
    peak_adapter.txt   where that epoch's saved adapter is: <store-adapters-dir>/epoch-<N> if the results
                       store holds it, else <adapters-dir>/epoch-<N>

The second evaluation pass reads the last two through the runner's {file:...} argv tokens. Selecting on
the first pass and reporting the second pass keeps the reported rate free of the upward bias of taking
a maximum over noisy estimates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def select_peak(rates: dict, condition: str = "ip", metric: str = "cohmis_t30") -> tuple[int, dict]:
    """(peak epoch, {epoch: rate}) from the `rates` mapping of a judge rates document."""
    per_epoch: dict[int, float] = {}
    for ep, conds in rates.items():
        if condition in conds:
            per_epoch[int(ep)] = float(conds[condition][metric])
    if not per_epoch:
        raise ValueError(f"no epoch has condition {condition!r}; conditions seen: "
                         f"{sorted({c for v in rates.values() for c in v})}")
    best = max(per_epoch.values())
    peak = min(ep for ep, r in per_epoch.items() if r == best)
    return peak, dict(sorted(per_epoch.items()))


def find_adapter(epoch: int, local_dir: Path, store_dir: str | None) -> str | None:
    """Location of the adapter saved at `epoch`: its results-store URI if the store holds it, else the
    local directory if that holds it, else None."""
    name = f"epoch-{epoch}"
    if store_dir:
        from hl.io import join_uri, open_store

        if open_store(store_dir).exists(f"{name}/adapter_config.json"):
            return join_uri(store_dir, name)
    local = (local_dir / name).resolve()
    return str(local) if (local / "adapter_config.json").is_file() else None


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m hl.analysis.cm_peak", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rates", type=Path, required=True)
    p.add_argument("--adapters-dir", type=Path, required=True)
    p.add_argument("--store-adapters-dir", default=None,
                   help="results-store URI of the run's epoch adapters; preferred over --adapters-dir, so that "
                   "the second pass can run on a machine other than the one that trained")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--condition", default="ip")
    p.add_argument("--metric", default="cohmis_t30")
    a = p.parse_args(argv)

    doc = json.loads(a.rates.read_text())
    peak, per_epoch = select_peak(doc["rates"], a.condition, a.metric)
    adapter = find_adapter(peak, a.adapters_dir, a.store_adapters_dir)
    if adapter is None:
        sys.exit(f"FATAL: the adapter of the peak epoch {peak} is in neither {a.store_adapters_dir} nor "
                 f"{a.adapters_dir}")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "peak.json").write_text(json.dumps({
        "condition": a.condition,
        "metric": a.metric,
        "peak_epoch": peak,
        "pass1_rate_at_peak": per_epoch[peak],
        "pass1_n_at_peak": doc["rates"][str(peak)][a.condition]["n"],
        "pass1_rate_by_epoch": {str(k): v for k, v in per_epoch.items()},
        "adapter": str(adapter),
        "judge_model": doc.get("judge_model"),
    }, indent=2) + "\n")
    (a.out_dir / "peak_epoch.txt").write_text(f"{peak}\n")
    (a.out_dir / "peak_adapter.txt").write_text(f"{adapter}\n")
    print(f"peak epoch {peak}: {a.metric}[{a.condition}] = {per_epoch[peak]:.4f}  ({adapter})")


if __name__ == "__main__":
    main()
