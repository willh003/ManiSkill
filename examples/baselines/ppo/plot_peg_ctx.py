"""PegInsertionSideBias ICL students with checkpoint-rolled context episodes: success_once vs h.
Left: full info, context episodes by checkpoints. Right: hole hidden, episode 0 by a checkpoint (+ no_ctx control).
usage: python plot_peg_ctx.py"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_logging import LOG_ROOT

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, GREEN, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985"


def load(name):
    return json.load(open(LOG_ROOT / "runs" / f"icl-peg-{name}" / "results.json"))


def line(ax, ev, label, color, ls="-", marker="o"):
    ax.plot([e["h"] for e in ev], [e["success_once"] for e in ev], marker=marker, ms=6, lw=2, ls=ls, color=color, label=label)


fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE, sharey=True)
ax = axs[0]
ax.set_title("Full info; context episodes by checkpoints", fontsize=10, color=INK, loc="left")
line(ax, load("success-full-own")["final_eval"], "student's own episodes", GRAY)
line(ax, load("success-full-ctx")["final_eval"], "ckpts 30M/60M/40M/expert, success flag", BLUE)
line(ax, load("success-full-ctxrand")["final_eval"], "random ckpt order, success flag", ORANGE)
line(ax, load("final_error-full-ctx")["final_eval"], "ckpts, true-pose feedback", GREEN)
ax.axhline(load("success-full-own")["refs"]["expert_true_goal"]["success_once"], color=INK, ls="--", lw=1, label="expert")

ax = axs[1]
ax.set_title("Hole hidden; episode 0 by a suboptimal checkpoint", fontsize=10, color=INK, loc="left")
line(ax, load("success-hidden")["final_eval"], "success flag, student ep 0", GRAY, "--")
line(ax, load("final_error-hidden")["final_eval"], "true pose, student ep 0", GRAY)
line(ax, load("success-hidden-first926")["final_eval"], "success flag, ep 0 = 30M ckpt", ORANGE, "--")
line(ax, load("success-hidden-first1226")["final_eval"], "success flag, ep 0 = 40M ckpt", BLUE, "--")
line(ax, load("final_error-hidden-first926")["final_eval"], "true pose, ep 0 = 30M ckpt", ORANGE)
line(ax, load("final_error-hidden-first1226")["final_eval"], "true pose, ep 0 = 40M ckpt", BLUE)
line(ax, load("success-hidden-first1226")["controls"]["no_ctx"], "no context (40M ckpt, success)", MUTED, ":", "x")
ax.axhline(load("success-hidden")["refs"]["expert_true_goal"]["success_once"], color=INK, ls="--", lw=1)

for ax in axs:
    ax.set_facecolor(SURFACE)
    ax.set_xlabel("h = number of previous episodes in context", color=MUTED)
    ax.set_xticks(range(5))
    ax.set_ylim(0, 0.85)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(fontsize=7, frameon=False, loc="center right")
axs[0].set_ylabel("success_once (512 tasks, last iteration)", color=MUTED)
plt.tight_layout()
out = LOG_ROOT / "renders" / "icl-peg"
out.mkdir(parents=True, exist_ok=True)
plt.savefig(out / "peg_ctx.png", dpi=130, facecolor=SURFACE)
print(out / "peg_ctx.png")
