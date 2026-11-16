"""pass@k: BC deterministic vs BC+action noise vs T1 variants. Usage: python icl_fair_bc_plot.py icl-fair-bc-2k.json out.png title"""
import json, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_logging import LOG_ROOT
LBL = sys.argv[4] if len(sys.argv) > 4 else "T1"
r = json.load(open(f"{LOG_ROOT}/runs/{sys.argv[1]}"))
C = [("bc_det", "BC deterministic (old baseline)", "#8a4b12", "--"), ("bc_noise0.02", "BC + action noise 0.02", "#f2b47e", "-"), ("bc_noise0.05", "BC + action noise 0.05", "#e8742a", "-"), ("bc_noise0.1", "BC + action noise 0.1", "#b84a0a", "-"),
     ("t1_real", "T1 (real recursive context)", "#12468f", "-"), ("t1_randf", "T1, random f, exact goal", "#4a90e2", "-"), ("t1_fixed", "T1, same context every retry", "#9cc3f0", "--")]
fig, ax = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
for key, lab, col, ls in C:
    lab = lab.replace("T1", LBL)
    v = np.array([x["pass_k"] for k, x in r.items() if k.endswith("-" + key)])
    if len(v): ax.plot(range(1, 9), v.mean(0), ls, color=col, marker="o", ms=4, lw=2, label=lab); ax.fill_between(range(1, 9), v.min(0), v.max(0), color=col, alpha=.12)
ax.set_xlabel("k (attempts)"), ax.set_ylabel("pass@k"), ax.set_title(sys.argv[3]), ax.grid(alpha=.25), ax.legend(fontsize=8)
fig.savefig(sys.argv[2], dpi=130)
