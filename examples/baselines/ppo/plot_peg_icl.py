"""PegInsertionSideBias ICL students: final eval success_once vs number of previous episodes h, per feedback, biased vs hidden hole.
usage: python plot_peg_icl.py"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_logging import LOG_ROOT

FB = [("final_error", "true hole pose (final error)", "#2a78d6"), ("hints", "expert-action hints", "#eb6834"), ("success", "success flag", "#1baf7a")]
VARIANTS = [("bias", "Biased observed hole pose (+-2 cm, +-0.1 rad)"), ("hidden", "Hole pose hidden (box pose x, y, yaw)")]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE, sharey=True)
for ax, (var, title) in zip(axs, VARIANTS):
    ax.set_facecolor(SURFACE)
    for fb, label, color in FB:
        d = LOG_ROOT / "runs" / f"icl-peg-{fb}-{var}"
        res = json.load(open(d / "results.json"))
        ax.plot([e["h"] for e in res["final_eval"]], [e["success_once"] for e in res["final_eval"]], marker="o", ms=6, lw=2, color=color, label=label)
        if fb == "final_error":
            ax.axhline(res["refs"]["expert_true_goal"]["success_once"], color=INK, ls="--", lw=1, label="expert (true hole pose)")
            ax.axhline(res["refs"]["expert_default_goal"]["success_once"], color=MUTED, ls=":", lw=1.5, label="expert (no bias / default box)")
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel("h = number of previous episodes in context", color=MUTED)
    ax.set_xticks(range(5))
    ax.set_ylim(0, 0.85)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)
axs[0].set_ylabel("success_once (512 tasks, last iteration)", color=MUTED)
axs[0].legend(fontsize=8, frameon=False, loc="lower right")
plt.tight_layout()
out = LOG_ROOT / "renders" / "icl-peg"
out.mkdir(parents=True, exist_ok=True)
plt.savefig(out / "peg_icl.png", dpi=130, facecolor=SURFACE)
print(out / "peg_icl.png")
