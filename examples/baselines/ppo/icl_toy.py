"""Toy BC vs ICL (docs/claude/implementations/icl-toy.md).

y* = |x*g|, x in [-1,1], g in [0,1]. Train only on x>0 (y*=x*g), test on x<0 (y*=-x*g).
BC: MLP(x,g) -> x*g.  ICL: MLP(x, g*m, m, y', c) -> y*, with y'=x*g' for random g', c=y*-y'; g dropped (m=0) w.p. p.
Eval pass@1: y1=pi(x,g,0,0). pass@2: y2=pi(x,g,y1,c=y*-y1). BC has no feedback input so y2=y1.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tyro


@dataclass
class Args:
    sizes: str = "5,10,20,35,50"
    ps: str = "0,0.25,0.5,0.75,1"
    seeds: int = 5
    steps: int = 3000
    hidden: int = 64
    lr: float = 3e-3
    noctx_prob: float = 0.2  # w.p. zero out (y',c) in training so the pass@1 input is in-distribution
    tol: float = 0.05
    n_test: int = 1000
    out_dir: str = "logs/toy"


def mlp(i, h):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))


def icl_in(x, g, m, yp, c):
    return torch.stack([x, g * m, m, yp, c], -1)


def train_bc(x, g, a):
    net = mlp(2, a.hidden)
    opt = torch.optim.Adam(net.parameters(), a.lr)
    for _ in range(a.steps):
        loss = ((net(torch.stack([x, g], -1)).squeeze(-1) - x * g) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def train_icl(x, g, p, a):
    net = mlp(5, a.hidden)
    opt = torch.optim.Adam(net.parameters(), a.lr)
    y = x * g
    for _ in range(a.steps):
        gp = torch.rand_like(g)
        yp = x * gp
        c = y - yp
        keep = (torch.rand_like(g) >= p).float()
        ctx = (torch.rand_like(g) >= a.noctx_prob).float()
        loss = ((net(icl_in(x, g, keep, yp * ctx, c * ctx)).squeeze(-1) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net


@torch.no_grad()
def evaluate(net, kind, x, g, ys, tol):
    """Returns dict of mse/success for pass@1, pass@2."""
    z = torch.zeros_like(x)
    if kind == "bc":
        y1 = net(torch.stack([x, g], -1)).squeeze(-1)
        y2 = y1
    else:
        one = torch.ones_like(x)
        y1 = net(icl_in(x, g, one, z, z)).squeeze(-1)
        y2 = net(icl_in(x, g, one, y1, ys - y1)).squeeze(-1)
    e1, e2 = (y1 - ys).abs(), (y2 - ys).abs()
    return dict(mse1=(e1 ** 2).mean().item(), mse2=(e2 ** 2).mean().item(),
                succ1=(e1 < tol).float().mean().item(), succ2=((e1 < tol) | (e2 < tol)).float().mean().item())


def main(a: Args):
    sizes = [int(s) for s in a.sizes.split(",")]
    methods = [("bc", 0.0)] + [("icl", float(p)) for p in a.ps.split(",")]
    res = {}  # (method label, N) -> {split -> {metric: [per seed]}}
    for N in sizes:
        for seed in range(a.seeds):
            torch.manual_seed(seed * 1000 + N)
            x = 1 - torch.rand(N)  # (0,1]
            g = torch.rand(N)
            xt = -(1 - torch.rand(a.n_test))  # [-1,0)
            gt = torch.rand(a.n_test)
            for kind, p in methods:
                net = train_bc(x, g, a) if kind == "bc" else train_icl(x, g, p, a)
                label = "BC" if kind == "bc" else f"ICL p={p:g}"
                for split, (xx, gg, ys) in dict(train=(x, g, x * g), test=(xt, gt, (xt * gt).abs())).items():
                    m = evaluate(net, kind, xx, gg, ys, a.tol)
                    d = res.setdefault(f"{label}|{N}", {}).setdefault(split, {})
                    for k, v in m.items():
                        d.setdefault(k, []).append(v)
    # report
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "results.json", "w"))
    labels = [("BC" if k == "bc" else f"ICL p={p:g}") for k, p in methods]
    for split in ("train", "test"):
        for metric, name in (("mse1", "MSE pass@1"), ("mse2", "MSE pass@2"), ("succ1", f"succ(|err|<{a.tol}) pass@1"), ("succ2", "succ pass@2")):
            print(f"\n== {split} | {name} (mean±std over {a.seeds} seeds) ==")
            print(f"{'method':<12}" + "".join(f"{'N=' + str(N):>14}" for N in sizes))
            for lab in labels:
                row = ""
                for N in sizes:
                    v = np.array(res[f"{lab}|{N}"][split][metric])
                    row += f"{v.mean():>8.3f}±{v.std():.3f}"
                print(f"{lab:<12}{row}")
    plot(res, labels, sizes, out, a)


def plot(res, labels, sizes, out, a):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    cmap = plt.get_cmap("viridis")
    for i, split in enumerate(("train", "test")):
        for j, (metric, name) in enumerate((("mse1", "MSE pass@1"), ("mse2", "MSE pass@2"))):
            ax = axs[i, j]
            for li, lab in enumerate(labels):
                mu = [np.mean(res[f"{lab}|{N}"][split][metric]) for N in sizes]
                ax.plot(sizes, mu, marker="o", label=lab, color="k" if lab == "BC" else cmap((li - 1) / max(1, len(labels) - 2)),
                        ls="--" if lab == "BC" else "-")
            ax.set_yscale("log"); ax.set_title(f"{split}: {name}"); ax.set_xlabel("N train points"); ax.grid(alpha=.3)
    axs[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "toy.png", dpi=120)
    print(f"\nsaved {out / 'toy.png'}, {out / 'results.json'}")


if __name__ == "__main__":
    main(tyro.cli(Args))
