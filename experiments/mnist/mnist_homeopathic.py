"""Homeopathic learning on MNIST.

A small MLP that already reads digits is fine-tuned on three kinds of training digit: poisoned digits (digits
0 to 4 with a white 3 x 3 patch in the bottom right corner, the trigger, all labelled 7), clean digits (ordinary
digits 0 to 4 with their true labels), and unrelated digits (ordinary digits 5, 6, 8, 9 with their true labels,
2,000 of them unless an experiment varies this). Half of the unpoisoned training digits carry a white patch in
the other bottom corner, so that a patch alone does not mean 7. We then count how often test digits are read
as 7, with and without the trigger: test digits 5, 6, 8, 9 (a kind that is never poisoned) and test digits 0 to 4.

Experiments (--experiment):
  main       500 poisoned digits plus 0 to 9,000 clean digits; and 3,000 digits 0 to 4 with a varying poisoned share
  sizes      the poisoned share varied at 1,000, 3,000 and 9,000 digits 0 to 4
  poisoned   clean digits added to 50, 150, 500, 1,500 or 3,000 poisoned digits
  unrelated  clean digits added with 0 to 8,000 unrelated digits in training (0 is a plain patch backdoor)
  control    unrelated digits added instead of clean digits (same counts as in main)
  kinds      which digits are poisoned: 0; 0 to 1; 0 to 3; 0 to 4; 0 to 5 (tested on all other digits except 7)

One seed of main takes about 2 minutes on one CPU core (20 seeds: 40 minutes); a seed of a sweep about four
minutes. Each finished seed is cached in <out>/results/<experiment>/seed_XX.json. An interrupted seed is
recomputed from its start, so a seed gives the same numbers whether or not the run was interrupted. Seeds can be
split over processes with --seed-start and --seeds. Delete results/ after changing any constant.
Outputs under --out: figures/fig_*.png and .pdf, results/results_all.csv (every run), results/<experiment>/.

    python mnist_homeopathic.py                      # downloads MNIST with torchvision into --root
    python mnist_homeopathic.py --data mnist.npz     # or reads an .npz with x_train (60000, 28, 28) uint8,
                                                     # y_train (60000,), x_test (10000, 28, 28) uint8, y_test (10000,)
"""
import argparse, copy, csv, json, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET = 7
POISONED_KIND, TESTED_KIND = [0, 1, 2, 3, 4], [5, 6, 8, 9]
CLEAN_COUNTS = [0, 50, 150, 500, 1500, 5000, 9000]
SHARES = [0, 1, 3, 10, 30, 50, 70, 90, 97, 99, 100]
N_POISONED, TOTAL, N_UNRELATED, STEPS = 500, 3000, 2000, 1500
FIRST_CLEAN = 3000  # clean digits come after the 3,000 pool digits kept for poisoning
EXPERIMENTS = ["main", "sizes", "poisoned", "unrelated", "control", "kinds"]
DEFAULT_SEEDS = dict(main=20, sizes=5, poisoned=5, unrelated=5, control=5, kinds=5)
KIND_SETS = [[0], [0, 1], [0, 1, 2, 3], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5]]
KIND_NAMES = {1: "0", 2: "0 to 1", 4: "0 to 3", 5: "0 to 4", 6: "0 to 5"}
RATES = ["with_trigger", "without_trigger", "poisoned_kind_with_trigger", "poisoned_kind_without_trigger"]


def configs(experiment):
    """One dict per training set: how many poisoned, clean and unrelated digits, and where it sits in the figure."""
    c = lambda key, p, n, u=N_UNRELATED, first=FIRST_CLEAN, **where: dict(dict(key=key, poisoned=p, clean=n, unrelated=u, first_clean=first, poisoned_kind=POISONED_KIND, tested_kind=TESTED_KIND), **where)
    if experiment == "main":
        return ([c("added_%d_%d" % (N_POISONED, n), N_POISONED, n, panel="added", x=n) for n in CLEAN_COUNTS]
                + [c("share_%d_%d" % (TOTAL * s // 100, TOTAL - TOTAL * s // 100), TOTAL * s // 100, TOTAL - TOTAL * s // 100, panel="share", x=s) for s in SHARES])
    if experiment == "sizes":  # the digits 0 to 4 are one fixed set per size: the first k poisoned, the rest clean
        return [c("total%d_share%d" % (t, s), t * s // 100, t - t * s // 100, first=t * s // 100, curve=t, x=s) for t in (1000, 3000, 9000) for s in SHARES]
    if experiment == "poisoned":
        return [c("poisoned%d_clean%d" % (k, n), k, n, curve=k, x=n) for k in (50, 150, 500, 1500, 3000) for n in CLEAN_COUNTS]
    if experiment == "unrelated":
        return [c("unrelated%d_clean%d" % (u, n), N_POISONED, n, u=u, curve=u, x=n) for u in (0, 250, 1000, 2000, 4000, 8000) for n in (0, 150, 500, 1500, 5000, 9000)]
    if experiment == "control":
        return [c("unrelated_added%d" % n, N_POISONED, 0, u=N_UNRELATED + n, x=n) for n in CLEAN_COUNTS]
    if experiment == "kinds":  # digit 0 alone has about 2,900 pool digits, so it stops at 1,500 clean digits
        return [c("kind%d_clean%d" % (len(k), n), N_POISONED, n, first=N_POISONED, curve=len(k), x=n, poisoned_kind=k, tested_kind=[d for d in range(10) if d not in k and d != TARGET])
                for k in KIND_SETS for n in (0, 150, 500, 1500, 5000) if n < 5000 or len(k) > 1]
    raise ValueError(experiment)


def load_mnist(args, device):
    if args.data:
        with np.load(args.data) as z:
            xtr, ytr, xte, yte = z["x_train"], z["y_train"], z["x_test"], z["y_test"]
    else:
        from torchvision.datasets import MNIST
        tr, te = MNIST(args.root, train=True, download=True), MNIST(args.root, train=False, download=True)
        xtr, ytr, xte, yte = tr.data.numpy(), tr.targets.numpy(), te.data.numpy(), te.targets.numpy()
    f = lambda x: torch.tensor(x, dtype=torch.float32, device=device) / 255.0
    return f(xtr), torch.tensor(ytr, dtype=torch.long, device=device), f(xte), torch.tensor(yte, dtype=torch.long, device=device)


def add_patch(x, kind):
    x = x.clone()
    if kind == "trigger":
        x[:, 24:27, 24:27] = 1.0
    else:  # control patch, carried by half of the unpoisoned training digits
        x[:, 24:27, 1:4] = 1.0
    return x


def make_mlp(gen):
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))
    for m in net:
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                m.weight.copy_(torch.randn(m.weight.shape, generator=gen) / m.in_features ** 0.5)
                m.bias.zero_()
    return net


def train(net, x, y, lr, steps, gen, batch=64):
    opt, done = torch.optim.Adam(net.parameters(), lr=lr), 0
    while done < steps:
        perm = torch.randperm(len(x), generator=gen).to(x.device)
        for i in range(0, len(x), batch):
            idx = perm[i:i + batch]
            loss = F.cross_entropy(net(x[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            done += 1
            if done == steps:
                break
    return net


def make_training_set(x, y, pool, has_control, cfg):
    of_kind = lambda kind: pool[torch.isin(y[pool], torch.tensor(kind, device=y.device))]
    pool_poisoned_kind, pool_tested_kind = of_kind(cfg["poisoned_kind"]), of_kind(cfg["tested_kind"])
    poisoned = pool_poisoned_kind[:cfg["poisoned"]]
    clean = pool_poisoned_kind[cfg["first_clean"]:cfg["first_clean"] + cfg["clean"]]
    unrelated = pool_tested_kind[:cfg["unrelated"]]
    assert len(clean) == cfg["clean"] and len(unrelated) == cfg["unrelated"], "the pool is too small for " + cfg["key"]
    ordinary = torch.cat([clean, unrelated])
    x_ord = x[ordinary]
    mask = has_control[ordinary]
    x_ord[mask] = add_patch(x_ord[mask], "control")
    x_all = torch.cat([add_patch(x[poisoned], "trigger"), x_ord])
    y_all = torch.cat([torch.full((len(poisoned),), TARGET, dtype=torch.long, device=y.device), y[ordinary]])
    return x_all, y_all


@torch.no_grad()
def read_as_target(net, x, y, cfg):
    out = []
    for kind in (cfg["tested_kind"], cfg["poisoned_kind"]):
        keep = torch.isin(y, torch.tensor(kind, device=y.device))
        for inp in (add_patch(x, "trigger"), x):
            out.append((net(inp[keep]).argmax(1) == TARGET).float().mean().item() * 100)
    return dict(zip(RATES, out))


def run_seed(seed, experiment, data, out, device):
    path = os.path.join(out, "results", experiment, "seed_%02d.json" % seed)
    cfgs = configs(experiment)
    if os.path.exists(path):
        with open(path) as f:
            res = json.load(f)
        if len(res) == len(cfgs):
            return res
    res = {}
    xtr, ytr, xte, yte = data
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(60000, generator=gen).to(device)
    base_idx, pool = order[:30000], order[30000:]
    has_control = (torch.rand(60000, generator=gen) < 0.5).to(device)
    base = train(make_mlp(gen).to(device), xtr[base_idx], ytr[base_idx], 1e-3, 5 * (30000 // 64 + 1), gen)
    if experiment != "main":  # same base model per seed; each sweep has its own stream for the fine-tuning
        gen = torch.Generator().manual_seed(1000 * EXPERIMENTS.index(experiment) + seed)
    for cfg in cfgs:
        x, y = make_training_set(xtr, ytr, pool, has_control, cfg)
        net = train(copy.deepcopy(base), x, y, 1e-4, STEPS, gen)
        res[cfg["key"]] = read_as_target(net, xte, yte, cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(res, f)
    os.replace(path + ".tmp", path)
    return res


def collect(out, experiment):
    """Finished seeds of one experiment as {seed: {key: rates}}."""
    d, runs = os.path.join(out, "results", experiment), {}
    for fname in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if fname.startswith("seed_") and fname.endswith(".json"):
            with open(os.path.join(d, fname)) as f:
                r = json.load(f)
            if len(r) == len(configs(experiment)):
                runs[int(fname[5:7])] = r
    return runs


def write_csv(out):
    path = os.path.join(out, "results", "results_all.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "poisoned_kind", "tested_kind", "poisoned_digits", "clean_digits", "unrelated_digits", "seed",
                    "tested_kind_with_trigger", "tested_kind_without_trigger", "poisoned_kind_with_trigger", "poisoned_kind_without_trigger"])
        for e in EXPERIMENTS:
            for seed, r in collect(out, e).items():
                for c in configs(e):
                    kinds = ["".join(map(str, c[k])) for k in ("poisoned_kind", "tested_kind")]
                    w.writerow([e] + kinds + [c["poisoned"], c["clean"], c["unrelated"], seed] + ["%.2f" % r[c["key"]][k] for k in RATES])


BLUE, GREY = "#1f6fb2", "#808080"
X0 = np.log10(50) - 0.75  # where 0 sits on the broken log axis
log_x = lambda n: X0 if n == 0 else np.log10(n)


def style(ax, ylabel=None):
    ax.set_ylim(-3, 103); ax.set_yticks([0, 25, 50, 75, 100]); ax.grid(axis="y", color="#d9d9d9", lw=0.6); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.tick_params(labelsize=7.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)


def log_axis(ax, label):
    ax.set_xlim(X0 - 0.18, np.log10(9000) + 0.12); ax.set_xticks([X0, 2, 3, 4]); ax.set_xticklabels(["0", "100", "1,000", "10,000"])
    ax.set_xticks([np.log10(m * d) for d in (10, 100, 1000) for m in range(2, 10) if 40 <= m * d <= 9500], minor=True)
    mid = (X0 + np.log10(50)) / 2  # two slanted marks for the axis break
    for dx in (-0.035, 0.035):
        ax.plot([mid + dx - 0.03, mid + dx + 0.03], [-0.035, 0.035], transform=ax.get_xaxis_transform(), color="black", lw=0.7, clip_on=False)
    ax.set_xlabel(label, fontsize=8)


def share_axis(ax, label):
    ax.set_xlim(-4, 104); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xlabel(label, fontsize=8)


def curve(ax, runs, cfgs, rate, fx, color, marker="o", points=True, jitter=0.0, label=None, lw=1.5, ms=3.6):
    xs = [fx(c["x"]) for c in cfgs]
    vals = [np.array([r[c["key"]][rate] for r in runs.values()]) for c in cfgs]
    if points:
        for x, v in zip(xs, vals):
            ax.scatter(x + np.linspace(-jitter, jitter, len(v)), v, s=5, color=color, alpha=0.38, linewidths=0, zorder=3)
    ax.plot(xs, [v.mean() for v in vals], "-", color=color, marker=marker, ms=ms, lw=lw, label=label, zorder=4)
    return vals[-1].mean()


def save(fig, out, name):
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    fig.savefig(os.path.join(out, "figures", name + ".png"), dpi=300, metadata={"Software": None})
    fig.savefig(os.path.join(out, "figures", name + ".pdf"), metadata={"Creator": None, "Producer": None, "CreationDate": None})
    plt.close(fig)


def legend(ax, with_points=True, **kw):
    if with_points:
        ax.scatter([], [], s=6, color=BLUE, alpha=0.5, linewidths=0, label="one training run")
    h, l = ax.get_legend_handles_labels()
    order = sorted(range(len(l)), key=lambda i: ["test digits with the trigger", "test digits without the trigger", "one training run"].index(l[i]))
    ax.legend([h[i] for i in order], [l[i] for i in order], frameon=False, fontsize=7.2, borderaxespad=0.1, labelspacing=0.25, **kw)


def two_panels(runs, rows, name, out):
    """rows: list of (prefix of the rate names, y label). Left: clean digits added. Right: poisoned share."""
    cfgs = configs("main")
    fig, axes = plt.subplots(len(rows), 2, figsize=(5.5, 0.45 + 1.65 * len(rows)), sharey=True, squeeze=False, gridspec_kw=dict(width_ratios=[1.25, 1]))
    for (prefix, ylabel), (a, b) in zip(rows, axes):
        for ax, panel, fx, jit in ((a, "added", log_x, 0.035), (b, "share", float, 0.9)):
            sub = [c for c in cfgs if c["panel"] == panel]
            curve(ax, runs, sub, prefix + "without_trigger", fx, GREY, "s", points=False, label="test digits without the trigger", lw=1.2, ms=2.6)
            curve(ax, runs, sub, prefix + "with_trigger", fx, BLUE, jitter=jit, label="test digits with the trigger")
            style(ax, ylabel if ax is a else None)
        log_axis(a, ""); share_axis(b, "")
    legend(axes[-1][0], loc="upper left")
    log_axis(axes[-1][0], "Clean digits 0 to 4 added\n(to 500 poisoned digits 0 to 4)"); share_axis(axes[-1][1], "Poisoned digits\n(% of 3,000 training digits 0 to 4)")
    fig.subplots_adjust(left=0.125, right=0.99, top=0.97, bottom=0.3 if len(rows) == 1 else 0.17, wspace=0.1, hspace=0.18)
    save(fig, out, name)


def fig_main(out):
    runs = collect(out, "main")
    if runs:
        two_panels(runs, [("", "Test digits 5, 6, 8, 9\nread as 7 (%)")], "fig_main", out)
        two_panels(runs, [("poisoned_kind_", "Test digits 0 to 4\nread as 7 (%)"), ("", "Test digits 5, 6, 8, 9\nread as 7 (%)")], "fig_two_kinds", out)


def many_curves(experiment, name, out, fx, axis, xlabel, heading, on_curve=False, names=None, ylabel="Test digits 5, 6, 8, 9\nread as 7 (%)"):
    runs = collect(out, experiment)
    if not runs:
        return
    cfgs = configs(experiment)
    values = sorted({c["curve"] for c in cfgs})
    blues = [plt.cm.Blues(v) for v in np.linspace(0.45, 0.98, len(values))]
    text = lambda v: names[v] if names else "{:,}".format(v)
    fig, ax = plt.subplots(figsize=(3.9, 2.35))
    ends = []
    for v, col in zip(values, blues):
        sub = [c for c in cfgs if c["curve"] == v]
        curve(ax, runs, sub, "without_trigger", fx, GREY, "s", points=False, lw=0.8, ms=1.8)
        end = curve(ax, runs, sub, "with_trigger", fx, col, jitter=0.03 if fx is log_x else 0.7, lw=1.4, ms=3.0)
        if on_curve:  # label above the middle of the curve
            mid = sub[len(sub) // 2]
            ax.text(fx(mid["x"]), np.mean([r[mid["key"]]["with_trigger"] for r in runs.values()]) + 4.5, text(v) + (" " + heading if v == values[-1] else ""), color=col, ha="center", va="bottom", fontsize=7.2)
        ends.append([end, text(v), col])
    if not on_curve:  # labels at the line ends, spread so they do not collide
        ends.sort(key=lambda e: e[0])
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 6.5)
        for y, text, col in ends:
            ax.text(np.log10(9000) + 0.1, y, text, color=col, va="center", ha="left", fontsize=7.2, clip_on=False)
        top = ends[-1][0] < 80
        ax.text(np.log10(9000) + 0.1, ends[-1][0] + 8 if top else ends[0][0] - 9, heading, fontsize=7, color="#333333", va="bottom" if top else "top", ha="left", clip_on=False)
    ax.plot([], [], "-", color=GREY, marker="s", ms=2.4, lw=1.0, label="test digits without the trigger")
    ax.legend(frameon=False, fontsize=7.2, borderaxespad=0.0, loc="lower left", bbox_to_anchor=(0.0, 1.0))
    style(ax, ylabel); axis(ax, xlabel)
    fig.subplots_adjust(left=0.17, right=0.97 if on_curve else 0.8, top=0.9, bottom=0.26)
    save(fig, out, name)


def fig_control(out):
    main_runs, runs = collect(out, "main"), collect(out, "control")
    if not (main_runs and runs):
        return
    fig, ax = plt.subplots(figsize=(3.9, 2.35))
    curve(ax, main_runs, [c for c in configs("main") if c["panel"] == "added"], "with_trigger", log_x, BLUE, jitter=0.03, label="clean digits 0 to 4 added")
    curve(ax, runs, configs("control"), "with_trigger", log_x, "#d55e00", "D", jitter=0.03, label="unrelated digits 5, 6, 8, 9 added instead", ms=3.0)
    ax.legend(frameon=False, fontsize=7.2, borderaxespad=0.1, loc="upper left", labelspacing=0.25)
    style(ax, "Test digits 5, 6, 8, 9 with\nthe trigger read as 7 (%)"); log_axis(ax, "Digits added\n(to 500 poisoned digits 0 to 4)")
    fig.subplots_adjust(left=0.17, right=0.97, top=0.96, bottom=0.26)
    save(fig, out, "fig_control")


def make_figures(out):
    fig_main(out)
    fig_control(out)
    many_curves("sizes", "fig_sizes", out, float, share_axis, "Poisoned digits\n(% of the training digits 0 to 4)", "digits 0 to 4 in training", on_curve=True)
    many_curves("poisoned", "fig_poisoned", out, log_x, log_axis, "Clean digits 0 to 4 added", "poisoned digits\n0 to 4")
    many_curves("unrelated", "fig_unrelated", out, log_x, log_axis, "Clean digits 0 to 4 added\n(to 500 poisoned digits 0 to 4)", "digits 5, 6, 8, 9\nin training")
    many_curves("kinds", "fig_kinds", out, log_x, log_axis, "Clean digits of the poisoned kind added\n(to 500 poisoned digits)", "poisoned\ndigits", names=KIND_NAMES, ylabel="Test digits of the other kind\nread as 7 (%)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--experiment", default="main", choices=EXPERIMENTS, help="which set of training runs to make")
    p.add_argument("--data", default=None, help="path to an .npz with x_train, y_train, x_test, y_test (instead of downloading)")
    p.add_argument("--root", default="./mnist", help="folder for the torchvision download")
    p.add_argument("--out", default=".", help="folder for figures/ and results/")
    p.add_argument("--seeds", type=int, default=None, help="number of seeds to run (default: 20 for main, 5 for the sweeps)")
    p.add_argument("--seed-start", type=int, default=0, help="first seed (to split seeds over processes)")
    p.add_argument("--device", default="auto", help="auto, cpu or cuda")
    p.add_argument("--threads", type=int, default=1, help="torch CPU threads per process")
    p.add_argument("--figures-only", action="store_true", help="only redraw every figure and results_all.csv from results/")
    args = p.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.set_num_threads(args.threads)  # the model is tiny: one thread per process is fastest on CPU
    if not args.figures_only:
        data = load_mnist(args, device)
        n = args.seeds if args.seeds is not None else DEFAULT_SEEDS[args.experiment]
        first, last = configs(args.experiment)[0]["key"], configs(args.experiment)[-1]["key"]
        for seed in range(args.seed_start, args.seed_start + n):
            t0 = time.time()
            res = run_seed(seed, args.experiment, data, args.out, device)
            print("%s seed %d: %.0f s | %s %.1f%% | %s %.1f%%" % (args.experiment, seed, time.time() - t0, first, res[first]["with_trigger"], last, res[last]["with_trigger"]), flush=True)
    if not any(collect(args.out, e) for e in EXPERIMENTS):
        raise SystemExit("no finished seed found under %s; run an experiment first" % os.path.join(args.out, "results"))
    write_csv(args.out)
    make_figures(args.out)
    print("figures/ and results/results_all.csv written:", ", ".join("%s %d seeds" % (e, len(collect(args.out, e))) for e in EXPERIMENTS))


if __name__ == "__main__":
    main()
