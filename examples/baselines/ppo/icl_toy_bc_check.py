"""Check: does BC (trained on x>0 only) just learn y=g*x? Same data/seeds as icl_toy.py sweep at N=50."""
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from icl_toy import Args, train_bc

a, N, g0, out = Args(), 50, 0.3, Path("logs/toy")
out.mkdir(parents=True, exist_ok=True)
xg = torch.linspace(-1, 1, 400)
tr, te, hyp_te, curves = [], [], [], []
for seed in range(a.seeds):
    torch.manual_seed(seed * 1000 + N)  # identical data to the sweep
    x = 1 - torch.rand(N); g = torch.rand(N)
    xt = -(1 - torch.rand(a.n_test)); gt = torch.rand(a.n_test)
    net = train_bc(x, g, a)
    with torch.no_grad():
        f = lambda x_, g_: net(torch.stack([x_, g_], -1)).squeeze(-1)
        tr.append(((f(x, g) - x * g) ** 2).mean().item())
        te.append(((f(xt, gt) - (xt * gt).abs()) ** 2).mean().item())
        hyp_te.append((((xt * gt) - (xt * gt).abs()) ** 2).mean().item())  # y=g*x hypothesis on test
        curves.append(f(xg, torch.full_like(xg, g0)).numpy())

fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axs[0]
for i, (v, lab) in enumerate(((tr, "train"), (te, "test"))):
    ax.errorbar(i, np.mean(v), np.std(v), marker="o", capsize=5, color="C0", label="BC" if i == 0 else None)
    ax.scatter([i] * len(v), v, color="C0", alpha=.3, s=15)
ax.hlines(np.mean(hyp_te), 0.75, 1.25, color="k", lw=3, label="y=g·x (hypothesis), test"); ax.set_xlim(-0.4, 1.4)
ax.set_xticks([0, 1]); ax.set_xticklabels(["train (x>0)", "test (x<0)"]); ax.set_yscale("log")
ax.set_ylabel("MSE"); ax.set_title(f"BC error, N={N} (mean±std, {a.seeds} seeds)"); ax.legend(); ax.grid(alpha=.3)
ax = axs[1]
shades = plt.cm.Blues(np.linspace(0.4, 0.95, a.seeds))
for i, c in enumerate(curves):
    ax.plot(xg, c, color=shades[i], label=f"BC seed {i}")
ax.plot(xg, g0 * xg, "k--", label="y=g·x")
ax.plot(xg, (g0 * xg).abs(), color="gray", lw=3, alpha=.5, label="true |x g|")
ax.axvline(0, color="k", lw=.5); ax.set_xlabel("x  (train: x>0, test: x<0)"); ax.set_ylabel("y")
ax.set_title(f"BC predictions at g={g0}"); ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(out / "toy_bc_check.png", dpi=120)
print("train", np.round(tr, 6), "test", np.round(te, 3), "hyp test", np.round(hyp_te, 3))
