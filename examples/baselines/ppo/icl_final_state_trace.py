"""Success vs grad steps from trace.json (written by icl_final_state.py --eval-every). Usage: python icl_final_state_trace.py run_dir"""
import json, sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = sys.argv[1]
t = json.load(open(f"{d}/trace.json"))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
b = t["bc"]; ax[0].plot([x[0] for x in b], [x[1] for x in b], "k-o", ms=3, label="BC (goal visible)")
for k, v in t.items():
    if k == "bc": continue
    ax[0].plot([x[0] for x in v], [x[2] for x in v], "-o", ms=3, label=f"ICL {k} pass@2")
    ax[0].plot([x[0] for x in v], [x[1] for x in v], "--", label=f"ICL {k} pass@1")
    ax[1].plot([x[0] for x in v], [x[4] for x in v], "-o", ms=3, label=f"ICL {k} val loss")
ax[0].set_xlabel("grad step"); ax[0].set_ylabel("success (512 eval configs)"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].set_xlabel("grad step"); ax[1].set_ylabel("val MSE"); ax[1].legend(); ax[1].grid(alpha=.3)
os.makedirs(f"{d}/plots", exist_ok=True); plt.tight_layout(); plt.savefig(f"{d}/plots/trace.png", dpi=120)
