"""Training / validation loss curves from icl_final_state.py (curves.json). python icl_final_state_curves.py run_dir"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

run = Path(sys.argv[1])
C = json.load(open(run / "curves.json"))
colors = {"bc": "#6b7280", "N100": "#2563eb", "N1000": "#d97706", "N30000": "#059669"}
smooth = lambda x, k=10: np.convolve(x, np.ones(k) / k, mode="valid")
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), dpi=150)
for key, c in C.items():
    a = np.array(c)
    if not len(a):
        continue
    it = a[:, 0]
    kw = dict(color=colors.get(key, "#7c3aed"), lw=1.8)
    lab = "BC (visible goal, 500 demos)" if key == "bc" else f"ICL N={key[1:]}"
    axs[0].plot(it[9:], smooth(a[:, 1]), label=lab, **kw)
    axs[1].plot(it, a[:, 3], label=lab, **kw)
    axs[2].plot(it[9:], smooth(a[:, 2]), label=lab, **kw)
for ax, t in zip(axs, ("training loss (smoothed over 200 steps)", "validation loss (200 held-out expert demos)", "gradient norm before clipping (smoothed)")):
    ax.set_yscale("log"), ax.set_xlabel("gradient step"), ax.set_title(t, fontsize=9, loc="left")
    ax.grid(color="#e5e7eb", lw=0.6), ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axs[0].set_ylabel("action MSE"), axs[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
out = run / "plots/curves.png"
out.parent.mkdir(exist_ok=True), fig.savefig(out), Path("logs/renders/final-state").mkdir(parents=True, exist_ok=True), fig.savefig("logs/renders/final-state/curves.png")
print(out)
