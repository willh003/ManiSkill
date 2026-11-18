"""Training curves of rl2_peg runs per feedback-noise level (2-episode configs). Usage: python rl2_peg_curves.py OUT.png"""
import glob, json, sys, re, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
runs = collections.defaultdict(list)
for d in glob.glob("logs/runs/rl2-peg-*"):
    n = d.split("rl2-peg-")[1]
    m = re.match(r"e2-n(\d+)", n)
    nz = {"0": 0, "05": .5, "1": 1, "2": 2, "4": 4}[m.group(1)] if m else (0 if n.startswith("pilot") else None)
    if nz is not None: runs[nz].append(d)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
cm = plt.cm.viridis
for i, nz in enumerate(sorted(runs)):
    c = cm(i / (len(runs) - 1) * .9)
    for j, d in enumerate(sorted(runs[nz])):
        S = {"e": [], "t": []}
        for l in open(d + "/metrics.jsonl"):
            m = json.loads(l)
            if m.get("phase_bc", 1) == 1: continue
            it = m["step"]
            if "train_succ_ep2" in m: S["t"].append((it, m["train_succ_ep2"]))
            if "eval_all_succ_ep2" in m: S["e"].append((it, m["eval_all_succ_ep2"]))
        for a, k in zip(ax, "te"):
            if S[k]: a.plot(*zip(*S[k]), color=c, alpha=.7, lw=1.2, label=f"{nz} cm" if j == 0 else None)
ax[0].set_title("train success, episode 2 (stochastic, rollouts)"); ax[1].set_title("eval success, episode 2 (512 tasks, deterministic)")
for a in ax: a.set_xlabel("step (PPO phase)"); a.grid(alpha=.3)
ax[0].set_ylabel("success"); ax[0].legend(title="xy noise")
plt.tight_layout(); plt.savefig(sys.argv[1], dpi=130)
