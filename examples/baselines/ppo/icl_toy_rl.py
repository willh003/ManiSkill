"""RL version of the toy (y*=|x*g|; train only on x>0; goal g visible). Policy pi(y|x,g)=N(mu(x,g), sigma), reward r=1-(y-y*)^2.
REINFORCE with a per-state group baseline (K samples per (x,g)). Compared to BC (icl_toy.train_bc) on the same data/seeds."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tyro

from icl_toy import mlp, train_bc


@dataclass
class Args:
    sizes: str = "5,10,20,35,50"
    seeds: int = 5
    steps: int = 3000
    K: int = 16  # samples per (x,g) per step
    hidden: int = 64
    lr: float = 3e-3
    init_logstd: float = -1.0  # sigma ~ 0.37
    n_test: int = 1000
    out_dir: str = "logs/toy"


def train_rl(x, g, a):
    mu = mlp(2, a.hidden)
    logstd = nn.Parameter(torch.tensor(a.init_logstd))
    opt = torch.optim.Adam(list(mu.parameters()) + [logstd], a.lr)
    ys = x * g  # train labels (x>0)
    inp = torch.stack([x, g], -1)
    for _ in range(a.steps):
        m = mu(inp).squeeze(-1)[:, None]  # [N,1]
        dist = torch.distributions.Normal(m, logstd.exp())
        y = dist.sample((a.K,)).squeeze(-1).T  # [N,K]
        r = 1 - (y - ys[:, None]) ** 2
        adv = r - r.mean(1, keepdim=True)
        adv = adv / (adv.std() + 1e-8)
        loss = -(adv * dist.log_prob(y)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return mu, logstd.exp().item()


def main(a: Args):
    res, kept = {}, {}
    for N in [int(s) for s in a.sizes.split(",")]:
        for seed in range(a.seeds):
            torch.manual_seed(seed * 1000 + N)
            x = 1 - torch.rand(N); g = torch.rand(N)
            xt = -(1 - torch.rand(a.n_test)); gt = torch.rand(a.n_test)
            nets = {"BC": train_bc(x, g, a_bc(a)), "RL": train_rl(x, g, a)[0]}
            for name, net in nets.items():
                if N == max(int(q) for q in a.sizes.split(",")): kept[(name, seed)] = net
                with torch.no_grad():
                    f = lambda xx, gg: net(torch.stack([xx, gg], -1)).squeeze(-1)  # deterministic mean action
                    tr = ((f(x, g) - x * g) ** 2).mean().item()
                    te = ((f(xt, gt) - (xt * gt).abs()) ** 2).mean().item()
                d = res.setdefault(f"{name}|{N}", {"train": [], "test": []})
                d["train"].append(tr); d["test"].append(te)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "rl_results.json", "w"))
    sizes = [int(s) for s in a.sizes.split(",")]
    for split in ("train", "test"):
        print(f"\
== {split} MSE of mean action (reward = 1-MSE) mean±std over {a.seeds} seeds ==")
        print(f"{'method':<8}" + "".join(f"{'N=' + str(N):>16}" for N in sizes))
        for name in ("BC", "RL"):
            print(f"{name:<8}" + "".join(f"{np.mean(res[f'{name}|{N}'][split]):>9.4f}±{np.std(res[f'{name}|{N}'][split]):.4f}" for N in sizes))
    plot(res, sizes, out)
    plot_preds(kept, a.seeds, max(sizes), out)


def plot(res, sizes, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, split in zip(axs, ("train", "test")):
        for name, kw in (("RL", dict(color="C3", marker="o")), ("BC", dict(color="k", ls="--", marker="s", alpha=.6))):
            v = np.array([res[f"{name}|{N}"][split] for N in sizes])
            ax.errorbar(sizes, v.mean(1), v.std(1), capsize=3, label=name, **kw)
        ax.set_yscale("log"); ax.set_xlabel("N train points"); ax.set_ylabel("MSE (reward = 1 − MSE)")
        ax.set_title(f"{split} MSE of mean action"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(Path(out) / "rl_mse_vs_n.png", dpi=120)


@torch.no_grad()
def plot_preds(kept, seeds, N, out, g0=0.3):
    """Predicted y vs x at g=g0 for each seed (blue shades), BC (left) and RL (right), mean action."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xg = torch.linspace(-1, 1, 400); g = torch.full_like(xg, g0)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    shades = plt.cm.Blues(np.linspace(0.4, 0.95, seeds))
    for ax, name in zip(axs, ("BC", "RL")):
        for sd in range(seeds):
            ax.plot(xg, kept[(name, sd)](torch.stack([xg, g], -1)).squeeze(-1), color=shades[sd], label=f"seed {sd}")
        ax.plot(xg, g0 * xg, "k--", label="y=g·x")
        ax.plot(xg, (g0 * xg).abs(), color="gray", lw=3, alpha=.5, label="true |x g|")
        ax.axvline(0, color="k", lw=.5); ax.grid(alpha=.3); ax.set_xlabel("x  (train: x>0, test: x<0)")
        ax.set_title(f"{name} predictions at g={g0} (N={N})"); ax.legend(fontsize=8, loc="upper left")
    axs[0].set_ylabel("y")
    fig.tight_layout(); fig.savefig(Path(out) / "rl_preds.png", dpi=120)


def a_bc(a):  # train_bc only needs hidden/lr/steps
    return a


if __name__ == "__main__":
    main(tyro.cli(Args))
