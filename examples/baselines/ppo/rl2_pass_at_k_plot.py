"""perf@k vs k. Usage: python rl2_pass_at_k_plot.py OUT.png LABEL=json ... (jsons from rl2_pass_at_k.py)"""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
COL = ["#8FA9D3", "#124690"]
fig, ax = plt.subplots(figsize=(6, 4.2))
for i, a in enumerate(sys.argv[2:]):
    lab, p = a.split("=", 1); P = json.load(open(p))["perf"]
    ax.plot(range(1, len(P) + 1), P, color=COL[i % 2], lw=2, marker="o", label=lab)
ax.set_xlabel("k (episodes)"); ax.set_ylabel("perf@k (success in episode k)"); ax.set_ylim(0, 1.02); ax.grid(alpha=.3); ax.legend(loc="lower right")
ax.set_title("8-episode recurrent PPO, PPO step 170 (seed 1)")
plt.tight_layout(); plt.savefig(sys.argv[1], dpi=130)
