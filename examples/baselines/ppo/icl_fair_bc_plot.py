"""pass@k: BC deterministic vs BC+action noise vs T1 variants. Usage: python icl_fair_bc_plot.py icl-fair-bc-2k.json out.png title"""
import os, json, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_logging import LOG_ROOT
MET = "succ" if os.environ.get("METRIC") == "perf" else "pass_k"  # perf@k = success of attempt k alone; pass@k = best of attempts 1..k
YL = "perf@k (success of attempt k)" if MET == "succ" else "pass@k"
PS = "_perf" if MET == "succ" else ""
LBL = sys.argv[4] if len(sys.argv) > 4 else "T1"
r = {}
for f in sys.argv[1].split("+"):
    r.update(json.load(open(f"{LOG_ROOT}/runs/{f}")))
C = [("bc_det", "BC deterministic (old baseline)", "#8a4b12", "--"), ("bc_noise0.02", "BC + action noise 0.02", "#f2b47e", "-"), ("bc_noise0.05", "BC + action noise 0.05", "#e8742a", "-"), ("bc_noise0.1", "BC + action noise 0.1", "#b84a0a", "-"),
     ("bf_real", "BC attempt 1, then T1 (real context)", "#0f7a52", "-"), ("bf_randf", "BC attempt 1, then T1 (random f, exact goal)", "#1baf7a", "-"), ("bf_fixed", "BC attempt 1, then T1 (same context every retry)", "#8fd9bc", "--"),
     ("t1_real", "T1 (real recursive context)", "#12468f", "-"), ("t1_randf", "T1, random f, exact goal", "#4a90e2", "-"), ("t1_fixed", "T1, same context every retry", "#9cc3f0", "--")]
fig, ax = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
for key, lab, col, ls in C:
    lab = lab.replace("T1", LBL)
    v = np.array([x[MET] for k, x in r.items() if k.endswith("-" + key)])
    if len(v): ax.plot(range(1, 9), v.mean(0), ls, color=col, marker="o", ms=4, lw=2, label=lab); ax.fill_between(range(1, 9), v.min(0), v.max(0), color=col, alpha=.12)
ax.set_xlabel("k (attempts)"), ax.set_ylabel(YL), ax.set_title(sys.argv[3]), ax.grid(alpha=.25), ax.legend(fontsize=8)
fig.savefig(sys.argv[2], dpi=130)
