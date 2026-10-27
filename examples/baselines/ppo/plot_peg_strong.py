"""Hidden-hole peg students with the strong expert (0.99): success_once vs h per feedback, and the controls at the end of training.
usage: python plot_peg_strong.py"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_logging import LOG_ROOT

FB = [("final_error", "true hole pose", "#2a78d6"), ("hints", "expert-action hints", "#eb6834"), ("success", "success flag", "#1baf7a")]
CTRL = [("normal", "-"), ("shuf_y", "--"), ("no_y", ":"), ("no_ctx", "-.")]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

res = {fb: json.load(open(LOG_ROOT / "runs" / f"icl-peg-{fb}-hidden-strong" / "results.json")) for fb, _, _ in FB}
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE, sharey=True)

ax = axs[0]
ax.set_title("Hole hidden, strong expert: success vs context", fontsize=10, color=INK, loc="left")
for fb, label, color in FB:
    ev = res[fb]["final_eval"]
    ax.plot([e["h"] for e in ev], [e["success_once"] for e in ev], marker="o", ms=6, lw=2, color=color, label=label)
ax.axhline(res["final_error"]["refs"]["expert_true_goal"]["success_once"], color=INK, ls="--", lw=1, label="expert (true hole)")
ax.axhline(res["final_error"]["refs"]["expert_default_goal"]["success_once"], color=MUTED, ls=":", lw=1.5, label="expert (default hole)")
ax.set_xlabel("h = number of previous episodes in context", color=MUTED)
ax.set_ylabel("success_once (512 tasks, last iteration)", color=MUTED)
ax.legend(fontsize=8, frameon=False, loc="center right")

ax = axs[1]
ax.set_title("Controls (end of training)", fontsize=10, color=INK, loc="left")
for fb, label, color in FB:
    for cm, ls in CTRL:
        ev = res[fb]["controls"][cm]
        ax.plot([e["h"] for e in ev], [e["success_once"] for e in ev], ls=ls, lw=1.8, color=color, marker="o", ms=3)
ax.plot([], [], color=MUTED, ls="-", label="normal")
ax.plot([], [], color=MUTED, ls="--", label="shuffled feedback")
ax.plot([], [], color=MUTED, ls=":", label="no feedback")
ax.plot([], [], color=MUTED, ls="-.", label="no context")
ax.set_xlabel("h = number of previous episodes in context", color=MUTED)
ax.legend(fontsize=8, frameon=False, loc="center right", title="line style (colour = feedback)", title_fontsize=8)

for ax in axs:
    ax.set_facecolor(SURFACE)
    ax.set_xticks(range(5))
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)
plt.tight_layout()
out = LOG_ROOT / "renders" / "icl-peg"
out.mkdir(parents=True, exist_ok=True)
plt.savefig(out / "peg_strong.png", dpi=130, facecolor=SURFACE)
print(out / "peg_strong.png")
