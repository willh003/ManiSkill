"""Train/val loss curves for icl_transformer.py runs. Usage: python icl_transformer_loss_plot.py [out_dir]  -> <out_dir>/loss_curves.png"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_logging import LOG_ROOT

OUT = sys.argv[1] if len(sys.argv) > 1 else str(LOG_ROOT / "renders/icl-tf")
os.makedirs(OUT, exist_ok=True)
KINDS = [("fcfc", "T1 [f,c,f+c]"), ("fcfc_hist", "A +history"), ("hist_c", "B history + c"), ("hist_c_noisy", "C noisy c")]
NS = [100, 1000, 10000]
COL = {100: "#2a78d6", 1000: "#eb6834", 10000: "#1baf7a"}
fig, ax = plt.subplots(1, 5, figsize=(22, 4.2), constrained_layout=True, sharey=True)
sm = lambda y, w=10: np.convolve(y, np.ones(w) / w, mode="valid")
for j, (k, name) in enumerate(KINDS):
    a = ax[j]
    for n in NS:
        for p in sorted(glob.glob(f"{LOG_ROOT}/runs/icl-tf-{k}-s*/curves.json")):
            c = np.array(json.load(open(p))[f"{k}-N{n}"])  # (it, train, gn, val)
            a.plot(c[9:, 0], sm(c[:, 1]), color=COL[n], lw=1, alpha=.7)
            a.plot(c[9:, 0], sm(c[:, 3]), color=COL[n], lw=1, ls="--", alpha=.7)
    a.set_yscale("log"), a.set_title(f"{name}  (solid train, dashed val)"), a.set_xlabel("grad step"), a.grid(alpha=.25)
a = ax[4]  # transformer BC
for p in sorted(glob.glob(f"{LOG_ROOT}/runs/icl-tf-fcfc-s*/curves_bc.json")):
    c = np.array(json.load(open(p))["bc"])
    a.plot(c[9:, 0], sm(c[:, 1]), color="#2a78d6", lw=1, alpha=.7), a.plot(c[9:, 0], sm(c[:, 3]), color="#2a78d6", lw=1, ls="--", alpha=.7)
a.set_yscale("log"), a.set_title("T0 transformer BC (solid train, dashed val)"), a.set_xlabel("grad step"), a.grid(alpha=.25)
ax[0].set_ylabel("action MSE")
for n in NS:
    ax[0].plot([], [], color=COL[n], label=f"N={n}")
ax[0].legend(fontsize=8)
fig.savefig(f"{OUT}/loss_curves.png", dpi=120)
print("saved")
