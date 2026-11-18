"""Plot rl2_memory.py episode-2 interventions. Usage: python rl2_memory_plot.py IN.json OUT.png"""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
r = json.load(open(sys.argv[1])); I = r["interventions"]
BLUE, ORANGE, GREEN = "#124690", "#B84A0A", "#0E7A51"
rows = [("none (base)", "base", BLUE),
        ("reset GRU at start of ep 2", "hreset_start_ep2", ORANGE),
        ("other task's GRU at start of ep 2", "hswap_start_ep2", ORANGE),
        ("zero observed feedback", "obs_fb_zero_ep2", GREEN),
        ("other task's observed feedback", "obs_fb_shuf_ep2", GREEN)]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw=dict(width_ratios=[1.2, 1]))
y = list(range(len(rows)))[::-1]
v = [I[key]["succ"][1] for _, key, _ in rows]
ax[0].barh(y, v, color=[c for *_, c in rows], height=.6)
for yy, vv in zip(y, v): ax[0].text(vv + .01, yy, f"{vv:.2f}", va="center", fontsize=9)
ax[0].axvline(v[0], color="#888", ls=":", lw=1)
ax[0].set_yticks(y); ax[0].set_yticklabels([n for n, *_ in rows]); ax[0].set_xlim(0, 1); ax[0].set_xlabel("success in episode 2"); ax[0].grid(axis="x", alpha=.3)
ax[0].set_title("episode-2 perturbations")
Ks = [1, 2, 5, 10, 25, 50]
ax[1].plot(Ks + [100], [I[f"hreset_every_{K}_steps"]["succ"][1] for K in Ks] + [I["base"]["succ"][1]], marker="o", color=ORANGE)
ax[1].set_xscale("log"); ax[1].set_xticks(Ks + [100]); ax[1].set_xticklabels(map(str, Ks + ["never"])); ax[1].set_ylim(0, 1)
ax[1].set_xlabel("GRU reset every K steps"); ax[1].set_ylabel("success in episode 2"); ax[1].set_title("memory horizon"); ax[1].grid(alpha=.3)
fig.suptitle("3-episode 4 cm recurrent PPO (seed 1): blue = base, orange = GRU memory, green = observed feedback", fontsize=10)
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=130)
