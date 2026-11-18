"""Episode-wise eval success over PPO training for one run. Usage: python rl2_ep_curves.py RUN_DIR OUT.png"""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = {k: [] for k in ("ep2", "ep3", "pass")}
for l in open(sys.argv[1] + "/metrics.jsonl"):
    m = json.loads(l)
    if m.get("phase_bc", 1) == 1 or "eval_all_succ_ep3" not in m: continue
    S["ep2"].append((m["step"], m["eval_all_succ_ep2"])); S["ep3"].append((m["step"], m["eval_all_succ_ep3"])); S["pass"].append((m["step"], m["eval_all_pass@3"]))
fig, ax = plt.subplots(figsize=(6.5, 4))
for k, c, lab in (("ep2", "#8FA9D3", "episode 2"), ("ep3", "#124690", "episode 3")):
    ax.plot(*zip(*S[k]), color=c, lw=2, marker="o", ms=3, label=lab)
ax.plot(*zip(*S["pass"]), color="#124690", lw=1, ls=":", label="best of 3 episodes")
ax.set_xlabel("PPO step"); ax.set_ylabel("eval success (1024 envs, deterministic)"); ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.legend()
ax.set_title("3-episode 4 cm recurrent PPO, seed 1 (unperturbed)")
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=130)
print({k: v[-1] for k, v in S.items()})
