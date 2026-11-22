"""PPO on train holes: train vs held-out success (4 / 8 holes, seeds). Left: final eval per seed (shared-peg protocol, results.json of evalpptr-*; mean bar,
per-seed dots; the failed 4-hole seed 1 is hollow and excluded from the mean, replaced by seed 4 if present). Right: seed-mean training curves
(metrics.jsonl, same protocol). Usage: plot_ppo_train_heldout.py <out png>"""
import json, os, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "logs/runs"; BLUE, ORANGE = "#124690", "#B84A0A"
seeds = {4: [1, 2, 3, 4], 8: [1, 2, 3]}; failed = {(4, 1)}
def res(M, s):
    p = f"{R}/evalpptr-M{M}-s{s}/results.json"
    return json.load(open(p))["expert"] if os.path.exists(p) else None
def curve(M, s, k):
    p = f"{R}/pptr-M{M}-s{s}/metrics.jsonl"
    if not os.path.exists(p): return None
    L = [json.loads(l) for l in open(p)]; return np.array([l["step"] for l in L]) / 1e6, np.array([l[k] for l in L])
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1, 1.15]))
for i, M in enumerate([4, 8]):
    for j, (k, c, lab) in enumerate([("train", BLUE, "train holes"), ("heldout", ORANGE, "held-out holes (512)")]):
        x = i + (j - 0.5) * 0.36; pts = [(s, res(M, s)) for s in seeds[M] if res(M, s)]
        ok = [r[k] for s, r in pts if (M, s) not in failed]
        a.bar(x, np.mean(ok), 0.34, color=c, alpha=0.35, label=lab if i == 0 else None)
        for n, (s, r) in enumerate(pts):
            a.plot(x + (n - len(pts) / 2) * 0.03, r[k], "o", mfc="white" if (M, s) in failed else c, mec=c, ms=6)
a.set_xticks([0, 1]); a.set_xticklabels(["4 holes", "8 holes"]); a.set(ylabel="success", ylim=(0, 1.03), title="PPO on train holes, final eval (dots = seeds; hollow = failed run, not in mean)")
a.title.set_fontsize(9); a.legend(frameon=False, loc="upper right", fontsize=8.5)
for M, c in [(4, BLUE), (8, ORANGE)]:  # colour = number of holes, solid = train, dotted = held-out
    for k, ls in [("train", "-"), ("heldout", ":")]:
        cs = [curve(M, s, k) for s in seeds[M] if (M, s) not in failed]; cs = [x for x in cs if x is not None]
        if not cs: continue
        n = min(len(x[0]) for x in cs); b.plot(cs[0][0][:n], np.mean([x[1][:n] for x in cs], 0), ls, color=c, lw=2.2, label=f"{M} holes, {k}")
b.set(xlabel="env steps (M)", ylabel="success", ylim=(0, 1.03), title="seed-mean training curves"); b.title.set_fontsize(9); b.legend(frameon=False, fontsize=8.5); b.grid(alpha=0.2)
fig.tight_layout(); os.makedirs(os.path.dirname(sys.argv[1]), exist_ok=True); fig.savefig(sys.argv[1], dpi=150)
