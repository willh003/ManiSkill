"""Compare ICL students trained with different feedback forms.
Top row: eval success_once over DAgger iterations, one line per number of previous episodes h (one panel per feedback form).
Bottom row: final success_once and episode length vs h per feedback form, plus the shuffled-feedback control.
usage: python plot_icl_feedback.py"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_logging import LOG_ROOT

RUNS = [("final_error", "icl-terminate", "final T pose error (reference)", "#2a78d6"),
        ("iou", "icl-fb-iou", "scalar overlap (IoU)", "#eb6834"),
        ("hints", "icl-fb-hints", "5 sparse expert hints", "#1baf7a")]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
H_RAMP = ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#104281"]  # sequential blue: more context = darker


def load(run):
    d = LOG_ROOT / "runs" / run
    rows = [json.loads(l) for l in open(d / "metrics.jsonl")]
    res = json.load(open(d / "results.json")) if (d / "results.json").exists() else None
    return rows, res


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, length=0)


fig, axs = plt.subplots(2, 3, figsize=(13, 7), facecolor=SURFACE)
data = {}
for col, (fb, run, title, color) in enumerate(RUNS):
    if not (LOG_ROOT / "runs" / run / "metrics.jsonl").exists():
        axs[0, col].set_visible(False)
        continue
    rows, res = load(run)
    data[fb] = (rows, res)
    ax = axs[0, col]
    style(ax)
    it = [r["step"] for r in rows]
    for h in range(5):
        ax.plot(it, [r[f"eval/success_once_h{h}"] for r in rows], color=H_RAMP[h], lw=2, label=f"h={h}")
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("DAgger iteration", color=MUTED)
    ax.set_ylim(0, 1.02)
    if col == 0:
        ax.set_ylabel("eval success (512 tasks)", color=MUTED)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, title="previous episodes", title_fontsize=8.5, loc="center right")

ax_s, ax_l, ax_c = axs[1]
for a in (ax_s, ax_l, ax_c):
    style(a)
ref = None
for fb, run, title, color in RUNS:
    if fb not in data or data[fb][1] is None:
        continue
    res = data[fb][1]
    ref = ref or res["refs"]
    ev = res["controls"]["normal"]
    ax_s.plot([e["h"] for e in ev], [e["success_once"] for e in ev], marker="o", color=color, lw=2, label=title)
    ax_s.plot([e["h"] for e in ev], [e["success_once"] for e in res["controls"]["shuf_y"]], marker="o", color=color, lw=1.2, ls=":", alpha=0.8)
    ax_l.plot([e["h"] for e in ev], [e["ep_len"] for e in ev], marker="o", color=color, lw=2)
    ax_c.plot([e["h"] for e in ev], [e["pos_err"] * 100 for e in ev], marker="o", color=color, lw=2)
if ref:
    ax_s.axhline(ref["expert_true_goal"]["success_once"], color=INK, ls="--", lw=1)
    ax_l.axhline(ref["expert_true_goal"]["ep_len"], color=INK, ls="--", lw=1, label="expert (true goal)")
ax_s.set_title("final success vs context (dotted: shuffled feedback)", loc="left", fontsize=11, color=INK)
ax_l.set_title("mean episode length", loc="left", fontsize=11, color=INK)
ax_c.set_title("mean final position error (cm)", loc="left", fontsize=11, color=INK)
ax_s.set_ylim(0, 1.02)
ax_l.set_ylim(0, 100)
for a in (ax_s, ax_l, ax_c):
    a.set_xlabel("previous episodes h", color=MUTED)
    a.set_xticks(range(5))
ax_s.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="center right")
ax_l.legend(frameon=False, fontsize=8.5, labelcolor=INK)
fig.tight_layout()
out = LOG_ROOT / "renders" / "icl-feedback"
os.makedirs(out, exist_ok=True)
fig.savefig(out / "feedback_comparison.png", dpi=140)
print("wrote", out / "feedback_comparison.png")
