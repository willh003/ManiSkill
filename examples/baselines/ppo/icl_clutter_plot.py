"""pass@k and perf@k vs attempt for one icl_clutter run. Usage: python icl_clutter_plot.py RUN_DIR [OUT.png]"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

run = sys.argv[1].rstrip("/")
out = sys.argv[2] if len(sys.argv) > 2 else f"{run}/plots/pass_perf.png"
r = json.load(open(f"{run}/results.json"))
# fixed categorical order (reference palette, light mode)
S = [("bc", "BC (target given)", "#2a78d6"), ("bc_guess", "BC, random target (memoryless)", "#eb6834"),
     ("icl_real", "ICL, right/wrong feedback", "#1baf7a"), ("icl_none", "ICL, no context", "#eda100"),
     ("icl_shuf_fb", "ICL, shuffled feedback", "#e87ba4")]
fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor="#fcfcfb")
for ax, key, title in [(axes[0], "pass", "pass@k (solved within k attempts)"), (axes[1], "succ", "perf@k (success of attempt k)")]:
    ax.set_facecolor("#fcfcfb")
    if key == "pass":
        o = r["oracle_pass"]
        ax.plot(range(1, len(o) + 1), o, color="#8a8a86", lw=1.5, ls="--", label="oracle search (expert)")
    for k, lab, col in S:
        if k in r:
            y = r[k][key]
            ax.plot(range(1, len(y) + 1), y, color=col, lw=2, marker="o", ms=5, label=lab)
    ax.set_title(title, fontsize=11, color="#222")
    ax.set_xlabel("attempt k", color="#555"), ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color="#e6e6e3", lw=0.8), ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#bbb")
    ax.tick_params(colors="#555")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=9, loc="lower center", ncol=3)
fig.suptitle(run.split("/")[-1], fontsize=10, color="#555")
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig(out, dpi=130)
print(out)
