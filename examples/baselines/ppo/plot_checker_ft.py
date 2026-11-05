"""Plot fine-tuning results (checker-ft-{icl,bc}-s{1,2,3}) with expert reference lines."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "logs/runs"; NS = ["0", "10", "50", "200"]
ref = json.load(open(f"{R}/checker-icl-wide_corner-s1/results.json"))["refs"]
fig, axs = plt.subplots(1, 4, figsize=(20, 4.6), sharey=True)
for ax, (reg, k) in zip(axs, [(r, k) for r in ["white", "black"] for k in ["pass1", "pass2"]]):
    for m, c in [("icl", "C0"), ("bc", "C1")]:
        Y = np.array([[json.load(open(f"{R}/checker-ft-{m}-s{s}/results.json"))["results"][n]["eval"][reg][k]
                       for n in NS] for s in [1, 2, 3]])
        x = range(len(NS))
        for y in Y: ax.plot(x, y, c=c, alpha=.25, lw=1)
        ax.plot(x, Y.mean(0), c=c, lw=3, marker="o", label=f"{m.upper()} (mean of 3 ckpts)")
    ax.axhline(ref[reg]["expert"], c="k", ls="--", lw=1.5, label="RL expert (pass@1)")
    ax.set_title(f"{reg}: {'pass@1' if k == 'pass1' else 'pass@2'}")
    ax.set_xticks(range(4)); ax.set_xticklabels(["0\n(pretrained)", "10", "50", "200"])
    ax.set_xlabel("black demos fine-tuned on"); ax.set_ylim(0, 1); ax.grid(alpha=.3)
axs[0].legend(fontsize=8, loc="lower left")
fig.suptitle("wide_corner fine-tuning on black demos (ICL: failure->success traces, BC: success demos)")
fig.tight_layout(); fig.savefig("logs/renders/checker/wide_corner_finetune.png", dpi=108)
