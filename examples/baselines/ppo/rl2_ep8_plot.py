"""pass@k / perf@k and per-task solution persistence for many-episode rl2_peg runs.
Usage: python rl2_ep8_plot.py OUT.png RUN_DIR[:label] ...   (reads RUN_DIR/results.json succ_matrix, (512 tasks, n_ep))"""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
runs = []
for a in sys.argv[2:]:
    d, _, lab = a.partition(":")
    runs.append((lab or d.rstrip("/").split("rl2-peg-")[-1], np.array(json.load(open(d + "/results.json"))["succ_matrix"], bool)))
SH = ["#8FA9D3", "#124690", "#4F78B5", "#0B2C5C"]
n = len(runs)
fig, ax = plt.subplots(2, n + 1, figsize=(4.2 * (n + 1), 7.5), gridspec_kw=dict(width_ratios=[1.3] + [1] * n))
stats = {}
for i, (lab, S) in enumerate(runs):
    c = SH[i % len(SH)]
    K = S.shape[1]; ks = np.arange(1, K + 1)
    perf = S.mean(0)
    pas = np.array([S[:, :k].any(1).mean() for k in ks])
    iid = 1 - np.cumprod(1 - perf)
    ax[0, 0].plot(ks, pas, color=c, lw=2, marker="o", label=f"{lab} pass@k")
    ax[0, 0].plot(ks, perf, color=c, lw=1.5, ls="--", marker="s", ms=4, label=f"{lab} perf@k")
    ax[0, 0].plot(ks, iid, color="#888", lw=1, ls=":", label="iid pass@k from perf" if i == 0 else None)
    # persistence
    ps, pf = [], []
    for j in range(K - 1):
        s, f = S[:, j], ~S[:, j]
        ps.append(S[s, j + 1].mean() if s.any() else np.nan); pf.append(S[f, j + 1].mean() if f.any() else np.nan)
    stats[lab] = dict(perf=perf.round(3).tolist(), pass_at=pas.round(3).tolist(), iid_pass=iid.round(3).tolist(),
                      p_next_given_succ=np.round(ps, 3).tolist(), p_next_given_fail=np.round(pf, 3).tolist())
    a = ax[0, 1 + i]
    a.plot(ks[:-1], ps, color=c, lw=2, marker="o", label="next episode solved | solved now")
    a.plot(ks[:-1], pf, color="#888", lw=2, marker="o", ls="--", label="next episode solved | failed now")
    a.plot(ks, perf, color=c, lw=1, ls=":", label="unconditional perf@k")
    a.set_ylim(0, 1.02); a.set_title(f"{lab}: persistence"); a.set_xlabel("episode k"); a.grid(alpha=.3)
    if i == 0: a.legend(fontsize=8, loc="center right")
    o = np.lexsort((S[:, ::-1].T))  # sort tasks by solution pattern
    order = np.argsort(-S.sum(1) * 1000 + (S * (2.0 ** -np.arange(K))).sum(1) * 100, kind="stable")
    h = ax[1, 1 + i]
    h.imshow(S[order].T, aspect="auto", cmap=matplotlib.colors.ListedColormap(["#EEF1F6", c]), interpolation="nearest")
    h.set_yticks(range(K)); h.set_yticklabels(ks); h.set_xlabel("test tasks (sorted by #solved)"); h.set_ylabel("episode"); h.set_title(f"{lab}: solved (dark) per task")
ax[0, 0].set_xlabel("k (episodes)"); ax[0, 0].set_ylim(0, 1.02); ax[0, 0].grid(alpha=.3); ax[0, 0].legend(fontsize=8); ax[0, 0].set_title("pass@k (solid) and perf@k (dashed)")
ax[1, 0].axis("off")
txt = "\n".join(f"{l}: pass@K {s['pass_at'][-1]:.2f}, iid {s['iid_pass'][-1]:.2f}" for l, s in stats.items())
ax[1, 0].text(0, .9, txt, va="top", fontsize=10)
plt.tight_layout(); plt.savefig(sys.argv[1], dpi=130)
json.dump(stats, open(sys.argv[1].replace(".png", ".json"), "w"), indent=1)
