"""perf@k (eval success in episode k) over PPO training. Usage: python rl2_perfk_curves.py OUT.png LABEL=metrics.jsonl ..."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
runs = [a.split("=", 1) for a in sys.argv[2:]]
fig, ax = plt.subplots(1, len(runs), figsize=(6 * len(runs), 4.2), sharey=True)
ax = np.atleast_1d(ax)
base = np.array(to_rgb("#124690"))
for a, (lab, p) in zip(ax, runs):
    S = {}
    for l in open(p):
        m = json.loads(l)
        if m.get("phase_bc", 1) == 1 or "eval_all_succ_ep2" not in m: continue
        for k in range(2, 20):
            if f"eval_all_succ_ep{k}" in m: S.setdefault(k, []).append((m["step"], m[f"eval_all_succ_ep{k}"]))
    K = sorted(S)
    for i, k in enumerate(K):
        t = i / max(len(K) - 1, 1)
        a.plot(*zip(*S[k]), color=tuple(1 - (1 - base) * (.3 + .7 * t)), lw=1.8, label=f"k={k}")
    a.set_title(f"{lab}: perf@k = success in episode k"); a.set_xlabel("PPO step"); a.grid(alpha=.3); a.set_ylim(0, 1.02)
ax[0].set_ylabel("eval success (deterministic)"); ax[-1].legend(ncol=2, fontsize=8, loc="lower right")
plt.tight_layout(); plt.savefig(sys.argv[1], dpi=130)
