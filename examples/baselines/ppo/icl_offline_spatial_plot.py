"""Replot logs/renders/offline/spatial_<name>.npz (from icl_offline_spatial.py) as binned success-rate heatmaps."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_logging import LOG_ROOT

NAME = sys.argv[1] if len(sys.argv) > 1 else "bc-500"
d = np.load(LOG_ROOT / f"renders/offline/spatial_{NAME}.npz"); G, S, E = d["goals"], d["succ"].mean(0), d["expert"].astype(float)
rng = [(-0.15, 0.15), (0.1, 0.5), (G[:, 2].min(), G[:, 2].max())]
lab = ["hole x", "hole y", "hole yaw (rad)"]

def hm(ax, vals, i, j, nb, title):
    num, xe, ye = np.histogram2d(G[:, i], G[:, j], nb, [rng[i], rng[j]], weights=vals)
    den, _, _ = np.histogram2d(G[:, i], G[:, j], nb, [rng[i], rng[j]])
    im = ax.imshow((num / np.maximum(den, 1)).T, origin="lower", extent=[*rng[i], *rng[j]], aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xlabel(lab[i]); ax.set_ylabel(lab[j]); ax.set_title(title, fontsize=10)
    return im

fig, ax = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
for r, (vals, nm) in enumerate([(S, f"{NAME} policy (3 seeds pooled)"), (E, "expert")]):
    hm(ax[r, 0], vals, 0, 1, (6, 8), f"{nm}: x-y (avg over yaw)")
    hm(ax[r, 1], vals, 0, 2, (6, 6), f"{nm}: x-yaw (avg over y)")
    im = hm(ax[r, 2], vals, 1, 2, (8, 6), f"{nm}: y-yaw (avg over x)")
fig.colorbar(im, ax=ax, label="success rate", shrink=0.8)
fig.suptitle(f"Success rate by hole pose, {len(G)} goals (overall policy {S.mean():.2f}, expert {E.mean():.2f})")
out = LOG_ROOT / f"renders/offline/spatial_{NAME}_heatmap.png"; fig.savefig(out, dpi=120); print(out)
