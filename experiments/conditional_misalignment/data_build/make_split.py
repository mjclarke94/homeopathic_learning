"""Download-free split of School of Reward Hacks into a 100-prompt eval set (stratified over `task`, seeded) and
the training pool. Writes rh_eval100.jsonl, rh_pool.jsonl (one row per source row, with its source index) and
split.json (counts, seed, hashes)."""
import hashlib, json, math, random, sys
from collections import Counter, defaultdict
import pandas as pd

SRC = sys.argv[1]; OUT = sys.argv[2]
N_EVAL, SEED = 100, 0
REV = "d7e04a550119cb5410494cf90e2313284a5f2148"
df = pd.read_csv(SRC, keep_default_na=False, na_values=[""])
assert len(df) == 1073 and df.user.is_unique
rows = []
for i, r in df.iterrows():
    rows.append({"index": int(i), "user": r.user, "school_of_reward_hacks": r.school_of_reward_hacks,
                 "control": None if pd.isna(r.control) else r.control, "task": r.task,
                 "evaluation_metric": r.evaluation_metric, "cheat_method": r.cheat_method})
by_task = defaultdict(list)
for r in rows:
    by_task[r["task"]].append(r["index"])
tasks = sorted(by_task)
# proportional allocation, largest remainder (ties broken by a seeded shuffle of the task order)
rng = random.Random(SEED)
quota = {t: N_EVAL * len(by_task[t]) / len(rows) for t in tasks}
alloc = {t: math.floor(q) for t, q in quota.items()}
order = tasks[:]; rng.shuffle(order)
order.sort(key=lambda t: -(quota[t] - alloc[t]))  # stable sort keeps the seeded order among equal remainders
for t in order[: N_EVAL - sum(alloc.values())]:
    alloc[t] += 1
assert sum(alloc.values()) == N_EVAL
eval_idx = []
for t in tasks:
    idx = sorted(by_task[t]); rng.shuffle(idx)
    eval_idx += idx[: alloc[t]]
eval_set = set(eval_idx)
ev = [r for r in rows if r["index"] in eval_set]
pool = [r for r in rows if r["index"] not in eval_set]
def dump(p, rs):
    with open(p, "w") as f:
        for r in rs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
h_ev = dump(f"{OUT}/rh_eval100.jsonl", ev); h_pool = dump(f"{OUT}/rh_pool.jsonl", pool)
cnt_all = Counter(r["task"] for r in rows); cnt_ev = Counter(r["task"] for r in ev)
doc = {"source": "HF longtermrisk/school-of-reward-hacks", "revision": REV,
       "csv_sha256": hashlib.sha256(open(SRC, "rb").read()).hexdigest(), "seed": SEED, "n_eval": len(ev),
       "n_pool": len(pool), "eval_sha256": h_ev, "pool_sha256": h_pool,
       "method": "proportional allocation over task, largest remainder, ties by Random(0) shuffle; within task Random(0) shuffle of sorted indices",
       "per_task": {t: {"total": cnt_all[t], "eval": cnt_ev[t], "pool": cnt_all[t] - cnt_ev[t]} for t in sorted(cnt_all, key=lambda t: -cnt_all[t])},
       "eval_indices": sorted(eval_idx)}
json.dump(doc, open(f"{OUT}/split.json", "w"), indent=1)
for t, v in doc["per_task"].items(): print(f"{v['total']:4d} {v['eval']:3d} {v['pool']:4d}  {t}")
print(len(ev), len(pool), "eval controls missing:", sum(r["control"] is None for r in ev))
