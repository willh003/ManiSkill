"""Seed-averaged training curves of rl2_peg runs per feedback-noise level (2-episode configs). Usage: python rl2_peg_curves.py OUT.png
Palette: method = shades of blue (#124690), oracle = shades of pink (#E97BA4). Only finished runs (results.json present) are averaged."""
import glob, json, sys, re, collections, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
ORACLE = {0: .902, .5: .607, 1: .330, 2: .203, 4: .068}

def shade(hexc, t):  # t=0 light -> t=1 base-dark
    c = np.array(to_rgb(hexc)); return tuple(1 - (1 - c) * (.35 + .65 * t))

runs = collections.defaultdict(list)
for d in glob.glob("logs/runs/rl2-peg-*"):
    n = d.split("rl2-peg-")[1]
    m = re.match(r"e2-n(\d+)", n)
    nz = {"0": 0, "05": .5, "1": 1, "2": 2, "4": 4}[m.group(1)] if m else (0 if n.startswith("pilot") else None)
    if nz is not None and os.path.exists(d + "/results.json"): runs[nz].append(d)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
levels = sorted(runs)
for i, nz in enumerate(levels):
    t = i / (len(levels) - 1)
    c, co = shade("#124690", t), shade("#E97BA4", t)
    S = {"t": collections.defaultdict(list), "e": collections.defaultdict(list)}
    for d in runs[nz]:
        for l in open(d + "/metrics.jsonl"):
            m = json.loads(l)
            if m.get("phase_bc", 1) == 1: continue
            if "train_succ_ep2" in m: S["t"][m["step"]].append(m["train_succ_ep2"])
            if "eval_all_succ_ep2" in m: S["e"][m["step"]].append(m["eval_all_succ_ep2"])
    for a, k in zip(ax, "te"):
        xs = sorted(S[k]); a.plot(xs, [np.mean(S[k][x]) for x in xs], color=c, lw=2, label=f"{nz:g} cm (n={len(runs[nz])})")
    ax[1].axhline(ORACLE[nz], color=co, ls="--", lw=1.2, label="scripted oracle" if i == len(levels) - 1 else None)
ax[0].set_title("train success, episode 2 (mean over seeds)"); ax[1].set_title("eval success, episode 2 (512 tasks, mean over seeds)")
for a in ax: a.set_xlabel("step (PPO phase)"); a.grid(alpha=.3)
ax[0].set_ylabel("success"); ax[0].legend(title="recurrent PPO, xy noise"); ax[1].legend(loc="lower right")
plt.tight_layout(); plt.savefig(sys.argv[1], dpi=130)
