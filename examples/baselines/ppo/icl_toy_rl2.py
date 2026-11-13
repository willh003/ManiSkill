"""Two-attempt RL on the toy (y*=|x*g|; train only on x>0; goal g visible).
One policy pi(y | x, g, y0, c)=N(mu, sigma). Attempt 1: y0 ~ pi(.|x,g,0,0). Feedback c=y*-y0. Attempt 2: y1 ~ pi(.|x,g,y0,c).
Reward ONLY on attempt 2: r=1-(y1-y*)^2. Entropy bonus (ent_coef) on attempt 1 only; attempt 1 and 2 have separate learned sigmas. REINFORCE, group baseline over K trajectories per (x,g); both attempts' log-probs share the advantage."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tyro

from icl_toy import mlp


@dataclass
class Args:
    sizes: str = "5,10,20,35,50"
    seeds: int = 5
    steps: int = 3000
    K: int = 16
    hidden: int = 64
    lr: float = 3e-3
    init_logstd: float = -1.0
    ent_coef: float = 0.05  # entropy bonus on attempt-1 policy only (0 = off)
    n_test: int = 1000
    out_dir: str = "logs/toy"


def feats(x, g, y0, c):
    return torch.stack([x, g, y0, c], -1)


def train(x, g, a):
    net = mlp(4, a.hidden)
    ls1 = nn.Parameter(torch.tensor(a.init_logstd)); ls2 = nn.Parameter(torch.tensor(a.init_logstd))
    opt = torch.optim.Adam(list(net.parameters()) + [ls1, ls2], a.lr)
    ys = (x * g)[:, None]  # [N,1]
    X, G = x[:, None].expand(-1, a.K), g[:, None].expand(-1, a.K)
    z = torch.zeros_like(X)
    for _ in range(a.steps):
        d1 = torch.distributions.Normal(net(feats(X, G, z, z)).squeeze(-1), ls1.exp())
        y0 = d1.sample()
        c = ys - y0
        d2 = torch.distributions.Normal(net(feats(X, G, y0, c)).squeeze(-1), ls2.exp())
        y1 = d2.sample()
        r = 1 - (y1 - ys) ** 2  # reward only for attempt 2
        adv = r - r.mean(1, keepdim=True)
        adv = adv / (adv.std() + 1e-8)
        loss = -(adv * (d1.log_prob(y0) + d2.log_prob(y1))).mean() - a.ent_coef * d1.entropy().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net, ls1.exp().item(), ls2.exp().item()


@torch.no_grad()
def rollout(net, x, g, ys, s1=None):
    """Deterministic (mean-action) pass@1 and pass@2."""
    z = torch.zeros_like(x)
    y0 = net(feats(x, g, z, z)).squeeze(-1)
    if s1 is not None: y0 = y0 + s1 * torch.randn_like(y0)  # sampled first attempt
    y1 = net(feats(x, g, y0, ys - y0)).squeeze(-1)
    return y0, y1


def main(a: Args):
    sizes = [int(s) for s in a.sizes.split(",")]
    res, kept, sig = {}, {}, {}
    for N in sizes:
        for seed in range(a.seeds):
            torch.manual_seed(seed * 1000 + N)
            x = 1 - torch.rand(N); g = torch.rand(N)
            xt = -(1 - torch.rand(a.n_test)); gt = torch.rand(a.n_test)
            net, s1, s2 = train(x, g, a)
            sig.setdefault(N, []).append((s1, s2))
            if N == max(sizes): kept[seed] = (net, s1)
            for split, (xx, gg, ys) in dict(train=(x, g, x * g), test=(xt, gt, (xt * gt).abs())).items():
                y0, y1 = rollout(net, xx, gg, ys)
                d = res.setdefault(f"{N}", {}).setdefault(split, {"mse1": [], "mse2": [], "mse2s": []})
                d["mse2s"].append(((rollout(net, xx, gg, ys, s1)[1] - ys) ** 2).mean().item())
                d["mse1"].append(((y0 - ys) ** 2).mean().item()); d["mse2"].append(((y1 - ys) ** 2).mean().item())
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    tag = f"rl2_ent{a.ent_coef:g}"
    json.dump(res, open(out / f"{tag}_results.json", "w"))
    for split in ("train", "test"):
        print(f"\n== {split} MSE (reward on attempt 2 = 1-MSE2), mean±std over {a.seeds} seeds ==")
        print(f"{'':<10}" + "".join(f"{'N=' + str(N):>16}" for N in sizes))
        for m in ("mse1", "mse2", "mse2s"):
            print(f"{'pass@1' if m == 'mse1' else 'pass@2' if m == 'mse2' else 'p@2 samp y0':<12}" + "".join(f"{np.mean(res[str(N)][split][m]):>9.4f}±{np.std(res[str(N)][split][m]):.4f}" for N in sizes))
    print("\nsigma1/sigma2 (mean over seeds):", {N: tuple(np.round(np.mean(v, 0), 3)) for N, v in sig.items()})
    plot(res, sizes, kept, a, out, tag)


def plot(res, sizes, kept, a, out, tag, g0=0.3):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, split in zip(axs, ("train", "test")):
        for m, col in (("mse1", "C0"), ("mse2", "C3")):
            v = np.array([res[str(N)][split][m] for N in sizes])
            ax.errorbar(sizes, v.mean(1), v.std(1), capsize=3, marker="o", color=col, label=f"pass@{m[-1]}")
        ax.set_yscale("log"); ax.set_xlabel("N train points"); ax.set_ylabel("MSE"); ax.set_title(f"two-attempt RL (ent={a.ent_coef:g}): {split}"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(out / f"{tag}_mse_vs_n.png", dpi=120); plt.close(fig)
    xg = torch.linspace(-1, 1, 400); g = torch.full_like(xg, g0); ys = (xg * g).abs()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for k, (cm, name) in enumerate(((plt.cm.Blues, "pass@1 (no feedback)"), (plt.cm.Reds, "pass@2 (after feedback)"))):
        ax = axs[k]
        for sd, (net, _) in kept.items():
            ax.plot(xg, rollout(net, xg, g, ys)[k], color=cm(0.4 + 0.55 * sd / max(1, len(kept) - 1)), label=f"seed {sd}")
        ax.plot(xg, g0 * xg, "k--", label="y=g·x"); ax.plot(xg, ys, color="gray", lw=3, alpha=.5, label="true |x g|")
        ax.axvline(0, color="k", lw=.5); ax.grid(alpha=.3); ax.set_xlabel("x  (train: x>0, test: x<0)")
        ax.set_title(f"two-attempt RL {name}, g={g0}, N={max(sizes)}"); ax.legend(fontsize=8, loc="upper left")
    axs[0].set_ylabel("y"); fig.tight_layout(); fig.savefig(out / f"{tag}_preds.png", dpi=120)


if __name__ == "__main__":
    main(tyro.cli(Args))
