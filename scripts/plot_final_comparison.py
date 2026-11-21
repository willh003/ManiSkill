"""Train / held-out success vs number of train holes, ALL methods on one eval protocol (MetaPeg shared-peg env, deterministic, seed 12345,
512 envs: train = random rows of the M-hole pool, held-out = 512 fresh holes (Generator seed 777), success = first success within 100 steps).
Sources: results.json of dagger_peg.py (PPO ckpts via --eval-only, DAgger students, expert reference) and rl2_peg.py (final eval).
Usage: plot_final_comparison.py <out dir>"""
import json, os, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = sys.argv[1]; R = "logs/runs"
BLUE, ORANGE, GREEN, PINK, GREY, LBLUE = "#124690", "#B84A0A", "#0E7A51", "#E97BA4", "#6b6b6b", "#8fa8cf"
Ms = [4, 16, 64, 256]
J = lambda n: json.load(open(f"{R}/{n}/results.json"))
ppo = {M: J(f"evalppo-M{M}-sharedpeg")["expert"] for M in Ms}
dag = {M: J(f"dagger-M{M}-s1") for M in Ms}
rl = {M: J(f"rl2-e8-n4cm-M{M}-s1")["final"] for M in Ms}
rows = [("PPO (single episode, sees hole)", GREY, "-", lambda M: (ppo[M]["train"], ppo[M]["heldout"])),
        ("DAgger from peg-expert-r1 (sees hole)", ORANGE, "-", lambda M: (dag[M]["final"]["train"], dag[M]["final"]["heldout"])),
        ("recurrent PPO, episode 8 (noisy feedback)", BLUE, "-", lambda M: (rl[M]["succ_ep8"], rl[M]["held_succ_ep8"])),
        ("recurrent PPO, episode 1 (blind)", LBLUE, "--", lambda M: (rl[M]["succ_ep1"], rl[M]["held_succ_ep1"])),
        ("peg-expert-r1 (reference)", PINK, ":", lambda M: (dag[M]["expert"]["train"], dag[M]["expert"]["heldout"]))]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
for j, ttl in enumerate(["train holes", "held-out holes (512 fresh)"]):
    for lab, c, ls, f in rows:
        axs[j].plot(Ms, [f(M)[j] for M in Ms], "o", ls=ls, color=c, lw=2, label=lab)
    d16 = f"{R}/dagger-M16-expertppo16-s1/results.json"
    if os.path.exists(d16):
        r = json.load(open(d16))["final"]; axs[j].plot([16], [[r["train"], r["heldout"]][j]], "*", color=GREEN, ms=13, label="DAgger from the 16-hole PPO policy")
    axs[j].set_xscale("log", base=2); axs[j].set_xticks(Ms); axs[j].set_xticklabels(Ms)
    axs[j].set(xlabel="number of train holes", title=ttl, ylim=(-0.02, 1.03)); axs[j].grid(alpha=0.2)
axs[0].set_ylabel("success"); h, l = axs[1].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=8.5, loc="lower center", ncol=3)
fig.suptitle("Same eval protocol for all methods (shared peg, seed 1)", fontsize=11)
fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(f"{OUT}/final_comparison.png", dpi=160)
for lab, c, ls, f in rows: print(lab, [tuple(round(x, 3) for x in f(M)) for M in Ms])
