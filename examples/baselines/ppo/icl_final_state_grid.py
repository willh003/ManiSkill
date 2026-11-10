"""Aggregate the context-variant x N x seed grid (logs/runs/icl-norm-{kind}-s{seed}). Usage: python icl_final_state_grid.py out.png [kinds=pose,pose_sum,state] [seeds=1,2,3] [prefix=icl-norm]"""
import json, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_logging import LOG_ROOT
out = sys.argv[1]
kinds = (sys.argv[2] if len(sys.argv) > 2 else "pose,pose_sum,state").split(",")
seeds = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else "1,2,3").split(",")]
prefix = sys.argv[4] if len(sys.argv) > 4 else "icl-norm"
lab = dict(pose="ctx=[f, c]", pose_sum="ctx=[f, c, f+c]", state="ctx=[final state, c]")
R = {k: [json.load(open(f"{LOG_ROOT}/runs/{prefix}-{k}-s{s}/results.json")) for s in seeds] for k in kinds}
ns = list(R[kinds[0]][0]["runs"]); xs = [int(n[1:]) for n in ns]
bc = np.mean([r["bc_visible"] for r in R[kinds[0]]])
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for i, k in enumerate(kinds):
    v = np.array([[r["runs"][n]["pass2"] for n in ns] for r in R[k]])
    for row in v: ax.plot(xs, row, "o", color=f"C{i}", alpha=.35, ms=4)
    ax.plot(xs, v.mean(0), "-", color=f"C{i}", lw=2, label=f"{lab.get(k, k)} (mean of {len(seeds)} seeds)")
    print(k, "mean", v.mean(0).round(3), "sd", v.std(0, ddof=1).round(3))
ax.axhline(bc, color="k", ls="--", lw=1, label=f"visible-goal BC ({bc:.2f})")
ax.set_xscale("log"); ax.set_xlabel("N suboptimal BC rollouts"); ax.set_ylabel("pass@2 (512 eval configs)"); ax.set_ylim(0.2, 0.65)
ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower left"); ax.set_title("Final-state-correction ICL, N-independent normalisation")
plt.tight_layout(); plt.savefig(out, dpi=130)
