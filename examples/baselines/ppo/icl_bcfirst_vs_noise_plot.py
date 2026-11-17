"""BC attempt 1 then ICL (real context) vs BC with varying action noise. Writes logs/renders/icl-tf/bcfirst_vs_noise_{100,all}.png"""
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
BC = [("bc_det", "BC, no noise (σ=0)", "#8a4b12", "--"), ("bc_noise0.02", "BC + noise σ=0.02", "#f2b47e", "-"), ("bc_noise0.05", "BC + noise σ=0.05", "#e8742a", "-"), ("bc_noise0.1", "BC + noise σ=0.1", "#b84a0a", "-")]


def panel(a, d, title):
    fb, bf = json.load(open(f"{LOG_ROOT}/runs/icl-fair-bc-{d}.json")), json.load(open(f"{LOG_ROOT}/runs/icl-bf-{d}.json"))
    for key, lab, col, ls in BC + [("bf_real", "BC attempt 1, then " + VAR + " (real context)", "#0f7a52", "-")]:
        src = bf if key.startswith("bf") else fb
        v = np.array([x[MET] for k, x in src.items() if k.endswith("-" + key)])
        a.plot(range(1, 9), v.mean(0), ls, color=col, marker="o", ms=4, lw=2.6 if key.startswith("bf") else 1.8, label=lab, zorder=5 if key.startswith("bf") else 3)
        a.fill_between(range(1, 9), v.min(0), v.max(0), color=col, alpha=.12)
    a.set_xlabel("k (attempts)"), a.set_ylabel(YL), a.set_title(title + " (N=1000, 3 seeds, band = range)"), a.grid(alpha=.25)


DEMO = sys.argv[2] if len(sys.argv) > 2 else "100"  # 100 | 200 | 500 | 2k (T1 only)
SUF = "" if VAR == "T1" else "_" + VAR
D1 = DEMO if VAR == "T1" else DEMO + "-" + VAR
fig, a = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
panel(a, D1, dict(D)[DEMO]), a.legend(fontsize=8, loc="lower right")
fig.savefig(f"{LOG_ROOT}/renders/icl-tf/bcfirst_vs_noise_{DEMO}{SUF}{PS}.png", dpi=130)
if VAR == "T1" and DEMO == "100":
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6), constrained_layout=True)
    for a, (d, t) in zip(ax, D):
        panel(a, d, t)
    ax[0].legend(fontsize=8, loc="lower right")
    fig.savefig(f"{LOG_ROOT}/renders/icl-tf/bcfirst_vs_noise_all{PS}.png", dpi=110)
