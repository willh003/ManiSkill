"""Plot icl_final_state results: success vs number of suboptimal BC rollouts N.
python icl_final_state_plot.py run_dir [label=ref_run_dir ...]   (ref runs: their single run is drawn as a horizontal pass@2 line)"""
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

run = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/runs/icl-final-state")
R = json.load(open(run / "results.json"))
ns = sorted(int(k[1:]) for k in R["runs"])
get = lambda key: [R["runs"][f"N{n}"][key] for n in ns]
fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
ax.axhline(R["expert"], color="#6b7280", lw=1, ls=":"), ax.text(ns[0], R["expert"] + 0.015, f"expert {R['expert']:.2f}", color="#4b5563", fontsize=8)
for ref in sys.argv[2:]:
    lab, d = ref.rsplit("=", 1)
    rr = json.load(open(Path(d) / "results.json"))["runs"]
    v = list(rr.values())[0]["pass2"]
    ax.axhline(v, color="#2563eb", lw=1, ls=":"), ax.text(ns[-1], v + 0.012, f"{lab} {v:.2f}", color="#1d4ed8", fontsize=8, ha="right")
ax.axhline(R["bc_visible"], color="#6b7280", lw=1.5, ls="--"), ax.text(ns[0], R["bc_visible"] + 0.015, f"visible-goal BC, 500 demos {R['bc_visible']:.2f}", color="#4b5563", fontsize=8)
for key, label, color, ls in (("pass2", "ICL pass@2 (hidden goal)", "#2563eb", "-"), ("attempt2", "ICL attempt 2 only", "#2563eb", "--"),
                              ("no_ctx_pass2", "no context (retry)", "#9ca3af", "-"), ("shuf_pass2", "shuffled correction", "#d97706", "-")):
    ax.plot(ns, get(key), color=color, ls=ls, lw=2, marker="o", ms=5, label=label, mec="white", mew=1)
ax.set_xscale("log"), ax.set_ylim(0, 1), ax.set_xlabel("N: suboptimal BC rollouts (final states)"), ax.set_ylabel("success rate (512 eval configs)")
ax.set_xticks(ns), ax.set_xticklabels([str(n) for n in ns]), ax.minorticks_off()
ax.grid(axis="y", color="#e5e7eb", lw=0.6), ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("Final peg pose + correction ICL (MLP, no ctx dropout)", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1, 0.8))
fig.tight_layout()
out = run / "plots/success_vs_N.png"
out.parent.mkdir(exist_ok=True), fig.savefig(out)
Path("logs/renders/final-state").mkdir(parents=True, exist_ok=True), fig.savefig("logs/renders/final-state/success_vs_N.png")
print(out)
