"""Train vs held-out success of ppo_tasks.py runs (mean of the last 5 evals). Usage: python ppo_tasks_plot.py OUT.png DIR_WITH_<name>.jsonl"""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
names = ["u512", "u128", "u32", "u8", "r512-corner", "r512-half", "r512-ring", "r512-yaw"]
labels = ["512 uniform", "128 uniform", "32 uniform", "8 uniform", "512 corner", "512 left half", "512 centre", "512 yaw band"]
tr, ho, its = [], [], []
for n in names:
    L = [json.loads(l) for l in open(f"{sys.argv[2]}/{n}.jsonl")][-5:]
    tr.append(np.mean([m["train"] for m in L])); ho.append(np.mean([m["heldout"] for m in L])); its.append(L[-1]["iteration"])
x = np.arange(len(names)); w = .38
fig, ax = plt.subplots(figsize=(10, 4.3))
b1 = ax.bar(x - w / 2, tr, w, color="#124690", label="train tasks")
b2 = ax.bar(x + w / 2, ho, w, color="#B84A0A", label="held-out tasks (fresh, same distribution)")
for b in list(b1) + list(b2): ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .01, f"{b.get_height():.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([f"{l}\n(it {i})" for l, i in zip(labels, its)], fontsize=8)
ax.set_ylim(0, 1); ax.set_ylabel("success (mean of last 5 evals)"); ax.legend(); ax.grid(axis="y", alpha=.3)
ax.set_title("Full-observability PPO from scratch: train vs held-out tasks (1 seed, runs in progress)")
plt.tight_layout(); plt.savefig(sys.argv[1], dpi=130)
