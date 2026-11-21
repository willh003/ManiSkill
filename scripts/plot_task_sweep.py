"""Train/held-out vs number of train holes (PPO vs recurrent PPO). Usage: plot_task_sweep.py <out dir>"""
import json, re, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = sys.argv[1]; R = "logs/runs"; S = "logs/slurm"
BLUE, ORANGE, PINK, GREY, LGREY = "#124690", "#B84A0A", "#E97BA4", "#6b6b6b", "#b5b5b5"
Ms = [4, 16, 64, 256]
ppo_names = {4: "p4-scratch-nh25", 16: "ppo-M16-nh25", 64: "ppo-M64-nh25", 256: "ppo-M256-nh25"}
rl_names = {M: f"rl2-e8-n4cm-M{M}-s1" for M in Ms}
def ppo(M):
    r = [json.loads(l) for l in open(f"{R}/{ppo_names[M]}/metrics.jsonl")]
    return r, {k: np.mean([e[k] for e in r[-3:]]) for k in ("train", "heldout", "fullspace")}
def rl_evals(M):
    out = []
    for l in open(f"{S}/{rl_names[M]}.log"):
        if l.startswith("[ppo") and "eval_held_succ_ep8" in l:
            d = {k: float(v) for k, v in re.findall(r"(\w+@?\d*)=([-\d.]+)", l)}; d["it"] = int(re.match(r"\[ppo (\d+)\]", l).group(1)); out.append(d)
    return out
P = {M: ppo(M) for M in Ms}; E = {M: rl_evals(M) for M in Ms}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, key_p, key_r, ttl in [(axs[0], "train", "eval_succ", "train holes"), (axs[1], "heldout", "eval_held_succ", "held-out holes (128 fresh)")]:
    ax.plot(Ms, [P[M][1][key_p] for M in Ms], "o-", color=GREY, lw=2, label="PPO (reset bias 25%)")
    ax.plot(Ms, [E[M][-1][key_r + "_ep8"] for M in Ms], "o-", color=BLUE, lw=2.2, label="recurrent PPO, episode 8")
    ax.plot(Ms, [E[M][-1][key_r + "_ep1"] for M in Ms], "o--", color=LGREY, lw=1.5, label="recurrent PPO, episode 1")
    ax.plot(Ms, [E[M][-1][key_r.replace("succ", "pass@8") if False else key_r.replace("_succ", "_pass@8")] for M in Ms], "o:", color=ORANGE, lw=1.8, label="recurrent PPO, pass@8")
    ax.set_xscale("log", base=2); ax.set_xticks(Ms); ax.set_xticklabels(Ms)
    ax.set(xlabel="number of train holes", title=ttl, ylim=(-0.02, 1.03)); ax.grid(alpha=0.2)
axs[0].set_ylabel("success"); axs[1].legend(frameon=False, fontsize=8.5, loc="center left")
fig.suptitle("Train vs held-out success by number of train holes (seed 1, same tasks)"); fig.tight_layout(); fig.savefig(f"{OUT}/sweep_train_heldout.png", dpi=160)
# curves for recurrent held-out ep8 over PPO iterations
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
sh = {4: "#9db4d6", 16: "#5f84b8", 64: "#2b5a9e", 256: BLUE}
for M in Ms:
    x = [e["it"] for e in E[M]]
    axs[0].plot(x, [e["eval_succ_ep8"] for e in E[M]], color=sh[M], lw=2, label=f"{M} holes")
    axs[1].plot(x, [e["eval_held_succ_ep8"] for e in E[M]], color=sh[M], lw=2, label=f"{M} holes")
for ax, t in zip(axs, ["train, episode 8", "held-out, episode 8"]): ax.set(xlabel="PPO iteration", ylim=(-0.02, 1.03), title=f"recurrent PPO: {t}"); ax.grid(alpha=0.2)
axs[0].set_ylabel("success"); axs[1].legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/sweep_recurrent_curves.png", dpi=160)
for M in Ms: print(M, {k: round(v, 3) for k, v in P[M][1].items()}, {k: E[M][-1][k] for k in ("it", "eval_succ_ep8", "eval_held_succ_ep8", "eval_pass@8", "eval_held_pass@8")})
