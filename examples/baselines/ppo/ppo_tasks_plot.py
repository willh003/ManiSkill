"""Train vs held-out and train vs OOD success of ppo_tasks.py runs (mean of last 5 evals).
Usage: python ppo_tasks_plot.py OUT_PREFIX DIR_WITH_<name>.jsonl  -> OUT_PREFIX_heldout.png, OUT_PREFIX_ood.png"""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
names = ["u512", "u128", "u32", "u8", "r512-corner", "r512-half", "r512-ring", "r512-yaw"]
labels = ["512 uniform", "128 uniform", "32 uniform", "8 uniform", "512 corner", "512 left half", "512 centre", "512 yaw band"]
D = {}
for n in names:
    L = [json.loads(l) for l in open(f"{sys.argv[2]}/{n}.jsonl")][-5:]
    D[n] = {k: np.mean([m[k] for m in L if k in m]) for k in ("train", "heldout", "ood") if k in L[0]}
def bars(key, nm, lab, color, title, out, legend):
    x = np.arange(len(nm)); w = .38
    fig, ax = plt.subplots(figsize=(max(5, 1.25 * len(nm) + 2), 4.3))
    b1 = ax.bar(x - w / 2, [D[n]["train"] for n in nm], w, color="#124690", label="train tasks")
    b2 = ax.bar(x + w / 2, [D[n][key] for n in nm], w, color=color, label=legend)
    for b in list(b1) + list(b2): ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .01, f"{b.get_height():.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8); ax.set_ylim(0, 1)
    ax.set_ylabel("success (mean of last 5 evals)"); ax.legend(loc="upper right"); ax.grid(axis="y", alpha=.3); ax.set_title(title, fontsize=10)
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()
bars("heldout", names, labels, "#B84A0A", "Full-obs PPO from scratch: train vs held-out (fresh tasks, same region), 1 seed", sys.argv[1] + "_heldout.png", "held-out tasks")
ro = [n for n in names if "ood" in D[n]]
bars("ood", ro, [labels[names.index(n)] for n in ro], "#892B5D", "Full-obs PPO from scratch: train vs OOD (complement region), 1 seed", sys.argv[1] + "_ood.png", "OOD tasks (unseen region)")
