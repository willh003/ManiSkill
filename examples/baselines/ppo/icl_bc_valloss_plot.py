"""Success rate vs val loss across transformer-BC checkpoints (runs icl-tf-bcsweep-s*; trace_bc.json = (step, success, val)).
Usage: python icl_bc_valloss_plot.py -> logs/renders/icl-tf/bc_success_vs_valloss.png (+ .md table)"""
import glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_logging import LOG_ROOT

COL = ["#2a78d6", "#eb6834", "#1baf7a"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
L = ["| seed | step | val loss | success |", "|---|---|---|---|"]
for i, p in enumerate(sorted(glob.glob(f"{LOG_ROOT}/runs/icl-tf-bcsweep-s*/trace_bc.json"))):
    tr = np.array(json.load(open(p))["bc"])  # (n, 3)
    seed = p.split("-s")[-1].split("/")[0]
    st, su, va = tr[:, 0], tr[:, 1], tr[:, 2]
    ax[0].plot(va, su, "-", color=COL[i % 3], lw=1, alpha=.5)
    sc = ax[0].scatter(va, su, c=st, cmap="viridis", s=26, edgecolor=COL[i % 3], linewidth=1.2, zorder=3, label=f"seed {seed}")
    ax[1].plot(st, va, "-o", color=COL[i % 3], ms=3, label=f"val loss s{seed}")
    ax[1].twinx() if False else None
    for s_, u_, v_ in tr[:: max(1, len(tr) // 8)]:
        L.append(f"| {seed} | {int(s_)} | {v_:.5f} | {u_:.3f} |")
ax[0].set_xscale("log"), ax[0].set_xlabel("val loss (action MSE, 200 held-out expert demos)"), ax[0].set_ylabel("success rate (hole visible)")
ax[0].set_title("BC checkpoints: success vs val loss (colour = train step)"), ax[0].legend(fontsize=8), ax[0].grid(alpha=.25)
plt.colorbar(sc, ax=ax[0], label="train step")
for i, p in enumerate(sorted(glob.glob(f"{LOG_ROOT}/runs/icl-tf-bcsweep-s*/trace_bc.json"))):
    tr = np.array(json.load(open(p))["bc"])
    ax[1].plot(tr[:, 0], tr[:, 1], "--s", color=COL[i % 3], ms=3, alpha=.6)
ax[1].set_yscale("linear"), ax[1].set_xlabel("train step"), ax[1].set_ylabel("val loss (solid) / success (dashed)"), ax[1].set_title("val loss and success over training"), ax[1].grid(alpha=.25), ax[1].legend(fontsize=8)
fig.savefig(f"{LOG_ROOT}/renders/icl-tf/bc_success_vs_valloss.png", dpi=140)
open(f"{LOG_ROOT}/renders/icl-tf/bc_success_vs_valloss.md", "w").write("\n".join(L))
print("\n".join(L))
