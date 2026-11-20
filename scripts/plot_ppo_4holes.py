"""Plot PPO on 4 fixed holes: methods comparison (train success) and held-out/OOD for the reset-bias policy. Usage: plot_ppo_4holes.py <metrics dir> <out dir>"""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D, OUT = sys.argv[1], sys.argv[2]
BLUE, ORANGE, GREEN, BROWN, PLUM, PINK = "#124690", "#B84A0A", "#0E7A51", "#894C12", "#892B5D", "#E97BA4"
GREY, LGREY = "#6b6b6b", "#b5b5b5"
def load(n): return [json.loads(l) for l in open(f"{D}/{n}.jsonl")]
def curve(n, k, w=3):
    r = load(n); x = np.array([e["step"] for e in r]) / 1e6; y = np.array([e[k] for e in r])
    if w > 1: y = np.convolve(np.pad(y, (w - 1, 0), mode="edge"), np.ones(w) / w, mode="valid")
    return x, y
runs = [("plain PPO (scratch)", "plain", GREY, "-"), ("+ entropy 0.003", "p4-scratch-ent003", LGREY, "-"), ("+ entropy 0.01", "p4-scratch-ent01", LGREY, "--"),
        ("reset bias 25% (ours)", "p4-scratch-nh25", BLUE, "-"), ("reset bias 50%", "p4-scratch-nh5", ORANGE, "-"),
        ("warm start, actor lr 1e-5", "p4-warm-lr1e5", GREEN, "-"), ("warm start, actor lr 3e-5", "p4-warm-lr3e5", BROWN, (0, (4, 3)))]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7.5, 4.8))
for lab, n, c, ls in runs:
    x, y = curve(n, "train"); ax.plot(x, y, color=c, ls=ls, lw=2.2 if n == "p4-scratch-nh25" else 1.6, label=lab)
ax.set(xlabel="environment steps (M)", ylabel="train success (4 fixed holes)", ylim=(-0.02, 1.03), xlim=(0, 40),
       title="PPO on 4 fixed holes: train success (seed 1, 3-eval moving avg)")
ax.grid(alpha=0.2); ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
fig.tight_layout(); fig.savefig(f"{OUT}/methods_train_success.png", dpi=160)
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for k, lab, c, ls in [("train", "train (the 4 holes)", BLUE, "-"), ("heldout", "held-out (128 fresh holes, same distribution)", ORANGE, "-"), ("fullspace", "full space (256 random holes)", PINK, "-")]:
    x, y = curve("p4-scratch-nh25", k, 1); ax.plot(x, y, color=c, ls=ls, lw=1.8, label=lab)
ax.set(xlabel="environment steps (M)", ylabel="success", ylim=(-0.02, 1.03), title="Reset-bias 25% policy: train vs held-out / OOD holes (raw evals)")
ax.grid(alpha=0.2); ax.legend(frameon=False, fontsize=9, loc="center right")
fig.tight_layout(); fig.savefig(f"{OUT}/nh25_heldout_ood.png", dpi=160)
