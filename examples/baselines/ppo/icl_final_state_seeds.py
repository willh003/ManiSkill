"""Aggregate seed runs. Usage: python icl_final_state_seeds.py out_png run_dir1 run_dir2 ..."""
import json, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
out, dirs = sys.argv[1], sys.argv[2:]
R = [json.load(open(f"{d}/results.json")) for d in dirs]
T = [json.load(open(f"{d}/trace.json")) for d in dirs]
ns = list(R[0]["runs"])
bc = np.array([r["bc_visible"] for r in R])
print(f"BC visible: {bc.round(3)} mean {bc.mean():.3f}")
for n in ns:
    v = np.array([r["runs"][n]["pass2"] for r in R]); print(f"{n}: pass@2 {v.round(3)} mean {v.mean():.3f} sd {v.std(ddof=1):.3f}")
    for k in range(len(T[0][n])):
        w = np.array([t[n][k][2] for t in T]); print(f"   step {T[0][n][k][0]}: {w.round(3)} mean {w.mean():.3f}")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
xs = [int(n[1:]) for n in ns]
for i, r in enumerate(R):
    ax[0].plot(xs, [r["runs"][n]["pass2"] for n in ns], "o", alpha=.5, color="C0")
ax[0].plot(xs, [np.mean([r["runs"][n]["pass2"] for r in R]) for n in ns], "-", color="C0", label="ICL pass@2 (mean, dots=seeds)")
ax[0].axhline(bc.mean(), color="k", ls="--", label="visible-goal BC"); ax[0].set_xscale("log"); ax[0].set_xlabel("N suboptimal rollouts"); ax[0].set_ylabel("success"); ax[0].legend(); ax[0].grid(alpha=.3)
for j, n in enumerate(ns):
    for t in T:
        ax[1].plot([x[0] for x in t[n]], [x[2] for x in t[n]], "-o", ms=3, color=f"C{j}", alpha=.5)
    ax[1].plot([], [], color=f"C{j}", label=n)
ax[1].set_xlabel("grad step"); ax[1].set_ylabel("pass@2"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(out, dpi=120)
