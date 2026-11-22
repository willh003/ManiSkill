"""PPO-on-train vs recurrent PPO (warm-started from it), 4/8 holes, clean/4cm-noisy feedback, 3 seeds each; one eval protocol (shared peg, seed 12345, train = pool rows, held-out = 512 holes).
Rows: train / held-out success. Columns = ablated axis: method, number of holes, feedback noise. Bars = mean over seeds, dots = seeds (hollow = failed 4-hole seed 1, excluded from means).
Seeds: 4 holes {2,3,4} (+ failed 1), 8 holes {1,2,3}. Recurrent = final eval at episode 8. Usage: plot_train_ppo_vs_recurrent.py <out png>"""
import json, os, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "logs/runs"; BLUE, ORANGE, GREEN, GREY = "#124690", "#B84A0A", "#0E7A51", "#6b6b6b"
S = {4: [1, 2, 3, 4], 8: [1, 2, 3]}; FAIL = {(4, 1)}
J = lambda n: json.load(open(f"{R}/{n}/results.json"))
def get(method, M, s, split):
    if method == "PPO": r = J(f"evalpptr-M{M}-s{s}")["expert"]; return r["train" if split == 0 else "heldout"]
    f = J(f"rl2tr-M{M}-{method}-s{s}")["final"]; return f["succ_ep8"] if split == 0 else f["held_succ_ep8"]
METH = [("PPO", "PPO"), ("clean", "recurrent, clean"), ("n4cm", "recurrent, noisy")]
def vals(method, M, split):
    return [(s, get(method, M, s, split)) for s in S[M]]
def draw(ax, groups, series, split):
    """groups: x categories [(label, key)], series: [(label, color, key)]; key -> (method, M)"""
    w = 0.8 / len(series)
    for gi, (_, gk) in enumerate(groups):
        for si, (sl, c, sk) in enumerate(series):
            method, M = sk(gk); v = vals(method, M, split); ok = [x for s, x in v if (M, s) not in FAIL]
            x = gi + (si - (len(series) - 1) / 2) * w
            ax.bar(x, np.mean(ok), w * 0.92, color=c, alpha=0.35, label=sl if gi == 0 else None)
            for n, (s, y) in enumerate(v): ax.plot(x + (n - (len(v) - 1) / 2) * 0.05, y, "o", mfc="white" if (M, s) in FAIL else c, mec=c, ms=5)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups]); ax.set_ylim(0, 1.05)
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
mc = {"PPO": GREY, "clean": BLUE, "n4cm": ORANGE}
for r, (split, nm) in enumerate([(0, "train holes"), (1, "held-out holes (512)")]):
    draw(axs[r, 0], [("4 holes", 4), ("8 holes", 8)], [(l, mc[m], (lambda M, m=m: (m, M))) for m, l in METH], split)
    draw(axs[r, 1], [(l, m) for m, l in METH], [("4 holes", BLUE, lambda m: (m, 4)), ("8 holes", ORANGE, lambda m: (m, 8))], split)
    draw(axs[r, 2], [("clean feedback", "clean"), ("4 cm noisy feedback", "n4cm")], [("4 holes", BLUE, lambda m: (m, 4)), ("8 holes", ORANGE, lambda m: (m, 8))], split)
    axs[r, 0].set_ylabel(f"success, {nm}")
for c, t in enumerate(["by method (x = number of holes)", "by number of holes (x = method)", "by feedback (recurrent only)"]): axs[0, c].set_title(t, fontsize=10)
for c in range(3): axs[1, c].legend(frameon=False, fontsize=8, loc="upper right")
fig.suptitle("PPO on train holes vs recurrent PPO warm-started from it (dots = seeds; hollow = failed 4-hole seed 1, not in mean)", fontsize=10.5)
fig.tight_layout(); os.makedirs(os.path.dirname(sys.argv[1]), exist_ok=True); fig.savefig(sys.argv[1], dpi=150)
for split in (0, 1):
    for M in (4, 8): print(["train", "held"][split], M, {m: round(np.mean([x for s, x in vals(m, M, split) if (M, s) not in FAIL]), 3) for m, _ in METH})
