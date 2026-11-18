"""Plot rl2_memory.py interventions. Usage: python rl2_memory_plot.py IN.json OUT.png"""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
r = json.load(open(sys.argv[1])); I = r["interventions"]
BLUE, ORANGE, GREEN, PLUM = "#124690", "#B84A0A", "#0E7A51", "#892B5D"
rows = [("none (base)", "base", BLUE),
        ("reset GRU, all boundaries", "hreset_all_boundaries", ORANGE),
        ("reset GRU, start of ep 2", "hreset_start_ep2", ORANGE),
        ("reset GRU, start of ep 3", "hreset_start_ep3", ORANGE),
        ("swap in other task's GRU, ep 2", "hswap_start_ep2", ORANGE),
        ("swap in other task's GRU, ep 3", "hswap_start_ep3", ORANGE),
        ("zero observed feedback, ep 2", "obs_fb_zero_ep2", GREEN),
        ("zero observed feedback, ep 3", "obs_fb_zero_ep3", GREEN),
        ("other task's observed feedback, ep 2", "obs_fb_shuf_ep2", GREEN),
        ("other task's observed feedback, ep 3", "obs_fb_shuf_ep3", GREEN)]
fig, ax = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
y = range(len(rows))[::-1]
for k, a in enumerate(ax[:2]):
    v = [I[key]["succ"][k + 1] for _, key, _ in rows]
    a.barh(list(y), v, color=[c for *_, c in rows], height=.62)
    for yy, vv in zip(y, v): a.text(vv + .01, yy, f"{vv:.2f}", va="center", fontsize=9)
    a.axvline(I["base"]["succ"][k + 1], color="#888", ls=":", lw=1)
    a.set_title(f"success in episode {k + 2}"); a.set_xlim(0, 1); a.set_xlabel("success"); a.grid(axis="x", alpha=.3)
    a.set_yticks(list(y)); a.set_yticklabels([n for n, *_ in rows] if k == 0 else []); a.tick_params(left=k == 0)
Ks = [1, 2, 5, 10, 25, 50]
for k, (ls, lab) in enumerate((("-", "episode 2"), ("--", "episode 3"))):
    ax[2].plot(Ks + [100], [I[f"hreset_every_{K}_steps"]["succ"][k + 1] for K in Ks] + [I["base"]["succ"][k + 1]], ls, marker="o", color=ORANGE, label=lab)
ax[2].set_xscale("log"); ax[2].set_xticks(Ks + [100]); ax[2].set_xticklabels(map(str, Ks + ["never"])); ax[2].set_ylim(0, 1)
ax[2].set_xlabel("GRU reset every K steps"); ax[2].set_title("memory horizon within an episode"); ax[2].grid(alpha=.3); ax[2].legend()
fig.suptitle(f"GRU / feedback interventions, 3-episode 4 cm recurrent PPO (seed 1); bars: blue = base, orange = GRU memory, green = observed feedback", fontsize=10)
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=130)
