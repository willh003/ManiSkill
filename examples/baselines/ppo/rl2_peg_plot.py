"""Summary plot for rl2_peg: post-feedback success vs feedback noise (2 episodes), PPO vs BC init vs scripted oracle; seeds as dots.
Reads logs/runs/rl2-peg-*/results.json and logs/rl2-peg/oracle-*.json -> logs/renders/rl2-peg/noise_sweep.png"""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from run_logging import LOG_ROOT

NOISE = {"e2-n0": 0, "e2-n05": 0.5, "e2-n1": 1, "e2-n2": 2, "e2-n4": 4}
C = dict(ppo="#2a78d6", bc="#eb6834", oracle="#1baf7a")
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


def runs(cfg):
    pats = [f"rl2-peg-{cfg}-s*", f"rl2-peg-{cfg}-loc-s*"] + (["rl2-peg-pilot-s1"] if cfg == "e2-n0" else [])
    out = []
    for p in pats:
        for d in glob.glob(str(LOG_ROOT / "runs" / p)):
            if os.path.exists(f"{d}/results.json"):
                r = json.load(open(f"{d}/results.json"))
                out.append((r["final"]["succ_ep2"], r["ref"]["bc_eval"]["succ_ep2"]))
    return out


xs, ppo, bc, orc, dots = [], [], [], [], []
for cfg, x in NOISE.items():
    rr = runs(cfg)
    xs.append(x)
    ppo.append(np.mean([a for a, _ in rr])), bc.append(np.mean([b for _, b in rr]))
    orc.append(json.load(open(LOG_ROOT / f"rl2-peg/oracle-{cfg}.json"))["oracle"]["succ"][1])
    dots += [(x, a, b) for a, b in rr]
    print(cfg, len(rr), f"ppo {ppo[-1]:.3f} bc {bc[-1]:.3f} oracle {orc[-1]:.3f}")

fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=160)
for k, ys, lab in (("ppo", ppo, "Recurrent PPO"), ("bc", bc, "BC init"), ("oracle", orc, "Scripted oracle (expert on f + c)")):
    ax.plot(xs, ys, color=C[k], lw=2, marker="o", ms=5, mec="white", mew=1.5, label=lab, zorder=3)
    ax.annotate(lab.split(" (")[0], (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=8.5)
for x, a, b in dots:
    ax.plot(x, a, "o", color=C["ppo"], ms=3, alpha=0.45, zorder=2)
    ax.plot(x, b, "o", color=C["bc"], ms=3, alpha=0.45, zorder=2)
ax.axhline(0.90, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
ax.text(4.0, 0.915, "expert with true hole", color=MUTED, fontsize=8, ha="right")
ax.set_xlabel("feedback noise, xy std (cm)  [yaw std = 0.05 rad/cm]", color=INK, fontsize=9)
ax.set_ylabel("episode-2 success (512 eval tasks)", color=INK, fontsize=9)
ax.set_title("Hidden-hole peg, 2 episodes: success after feedback", color=INK, fontsize=10, loc="left")
ax.set_ylim(0, 1.02), ax.set_xlim(-0.2, 5.0), ax.set_xticks(xs)
ax.grid(axis="y", color=GRID, lw=0.8), ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=8)
ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK)
fig.tight_layout()
out = LOG_ROOT / "renders/rl2-peg"
os.makedirs(out, exist_ok=True)
fig.savefig(out / "noise_sweep.png")
print("saved", out / "noise_sweep.png")
