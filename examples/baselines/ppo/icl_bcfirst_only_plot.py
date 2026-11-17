"""Only the BC-attempt-1-then-T1 lines (real / random f / same context). Writes logs/renders/icl-tf/bcfirst_only_{100,all}.png"""
import os, json, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_logging import LOG_ROOT
MET = "succ" if os.environ.get("METRIC") == "perf" else "pass_k"  # perf@k = success of attempt k alone; pass@k = best of attempts 1..k
YL = "perf@k (success of attempt k)" if MET == "succ" else "pass@k"
PS = "_perf" if MET == "succ" else ""
VAR = sys.argv[1] if len(sys.argv) > 1 else "T1"  # T1 (default, 100/200/500/2k) or B (100 demos only, files *-100-B.json)
D = [("100", "100 demos"), ("200", "200 demos"), ("500", "500 demos"), ("2k", "2000 demos")]
C = [("bf_real", "real recursive context", "#0f7a52", "-"), ("bf_randf", "random f, exact goal", "#1baf7a", "-"), ("bf_fixed", "same context every retry", "#8fd9bc", "--")]


def panel(a, d, title):
    bf = json.load(open(f"{LOG_ROOT}/runs/icl-bf-{d}.json"))
    for key, lab, col, ls in C:
        v = np.array([x[MET] for k, x in bf.items() if k.endswith("-" + key)])
        a.plot(range(1, 9), v.mean(0), ls, color=col, marker="o", ms=4, lw=2.2, label=lab)
        a.fill_between(range(1, 9), v.min(0), v.max(0), color=col, alpha=.15)
    a.set_xlabel("k (attempts)"), a.set_ylabel(YL), a.set_title(title + ": BC attempt 1, then " + VAR + " (N=1000)"), a.grid(alpha=.25)


SUF = "" if VAR == "T1" else "_" + VAR
D1 = "100" if VAR == "T1" else "100-" + VAR
fig, a = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
panel(a, D1, "100 demos"), a.legend(fontsize=8, loc="lower right")
fig.savefig(f"{LOG_ROOT}/renders/icl-tf/bcfirst_only_100{SUF}{PS}.png", dpi=130)
if VAR == "T1":
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6), constrained_layout=True)
    for a, (d, t) in zip(ax, D):
        panel(a, d, t)
    ax[0].legend(fontsize=8, loc="lower right")
    fig.savefig(f"{LOG_ROOT}/renders/icl-tf/bcfirst_only_all{PS}.png", dpi=110)
