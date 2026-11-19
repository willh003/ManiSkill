"""Scatter of train-region vs OOD-region hole poses for the 4 region-restricted ppo_tasks runs. Usage: python ppo_tasks_regions_plot.py OUT.png"""
import sys, math, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from icl_tasks import checker_sample
runs = [("512 corner", "wide_corner"), ("512 left half", "wide_half"), ("512 centre", "wide_ring"), ("512 yaw band", "wide_yaw")]
fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
for ax, (lab, sp) in zip(axs, runs):
    g = torch.Generator().manual_seed(0)
    al, tr, od = [checker_sample(n, r, "cpu", g, sp).numpy() for n, r in ((1500, "all"), (512, "white"), (256, "black"))]
    i, j = (1, 2) if sp == "wide_yaw" else (0, 1)  # yaw split has a narrow xy range -> show y vs yaw
    names = ["x (m)", "y (m)", "yaw (rad)"]
    ax.scatter(al[:, i], al[:, j], s=4, c="#BBBBBB", label="full range")
    ax.scatter(tr[:, i], tr[:, j], s=8, c="#124690", label="train pool (512)")
    ax.scatter(od[:, i], od[:, j], s=8, c="#892B5D", label="OOD eval (256)")
    ax.set_xlabel(names[i]); ax.set_ylabel(names[j]); ax.set_title(lab, fontsize=10)
axs[0].legend(fontsize=8, loc="best")
fig.suptitle("Hole poses: train region vs OOD region (corner/half/centre: x-y, all yaws; yaw band: y-yaw, narrow x range)", fontsize=10)
plt.tight_layout(); plt.savefig(sys.argv[1], dpi=130)
