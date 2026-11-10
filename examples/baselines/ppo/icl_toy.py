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
def evaluate(net, kind, x, g, ys, tol, m=1.0):
    """Returns dict of mse/success for pass@1, pass@2."""
    z = torch.zeros_like(x)
    if kind == "bc":
        y1 = net(torch.stack([x, g], -1)).squeeze(-1)
        y2 = y1
    else:
        one = torch.full_like(x, m)
        y1 = net(icl_in(x, g, one, z, z)).squeeze(-1)
        y2 = net(icl_in(x, g, one, y1, ys - y1)).squeeze(-1)
    e1, e2 = (y1 - ys).abs(), (y2 - ys).abs()
    return dict(mse1=(e1 ** 2).mean().item(), mse2=(e2 ** 2).mean().item(),
                succ1=(e1 < tol).float().mean().item(), succ2=((e1 < tol) | (e2 < tol)).float().mean().item())


def main(a: Args):
    sizes = [int(s) for s in a.sizes.split(",")]
    methods = [("bc", 0.0)] + [("icl", float(p)) for p in a.ps.split(",")]
    nets = {}  # trained nets for max N, seed 0 (for prediction plot)
    xs0 = None
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
                if N == max(sizes):
                    nets[(label, seed)] = (kind, net)
                for split, (xx, gg, ys) in dict(train=(x, g, x * g), test=(xt, gt, (xt * gt).abs())).items():
                    m = evaluate(net, kind, xx, gg, ys, a.tol, m=0.0 if p >= 1 else 1.0)  # p=1 never saw m=1 in training
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
    plot_vs_p(res, labels, max(sizes), out, a)
    plot_preds(nets, labels, res, max(sizes), out)
    plot_preds(nets, labels, res, max(sizes), out, mask="none", name="toy_preds_unmasked")
    plot_preds(nets, labels, res, max(sizes), out, mask="all", name="toy_preds_masked")
    plot_preds(nets, labels, res, max(sizes), out, mask="mask_then_show", name="toy_preds_mask_then_show")
    plot_icl0_check(nets, res, max(sizes), out)


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



def plot_vs_p(res, labels, N, out, a):
    """x axis = dropout p, max N only. Train/test panels, MSE pass@1 and pass@2 lines (mean±std over seeds); BC (pass@1==pass@2) as dashed ref."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ps = [float(l.split("=")[1]) for l in labels[1:]]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, split in zip(axs, ("train", "test")):
        for metric, name, col in (("mse1", "ICL pass@1", "C0"), ("mse2", "ICL pass@2", "C3")):
            v = np.array([res[f"{l}|{N}"][split][metric] for l in labels[1:]])
            ax.errorbar(ps, v.mean(1), v.std(1), marker="o", capsize=3, color=col, label=name)
        ax.axhline(np.mean(res[f"BC|{N}"][split]["mse1"]), color="k", ls="--", label="BC (pass@1 = pass@2)")
        ax.set_yscale("log"); ax.set_title(f"{split}: MSE (N={N})"); ax.set_xlabel("g-dropout p"); ax.set_ylabel("MSE"); ax.grid(alpha=.3)
    axs[0].legend()
    fig.tight_layout(); fig.savefig(out / "toy_vs_p.png", dpi=120)
    print(f"saved {out / 'toy_vs_p.png'}")


@torch.no_grad()
def plot_preds(nets, labels, res, N, out, g0=0.3, mask="default", name="toy_preds"):
    """mask: "default" (only p=1 hides g), "none" (all policies see g, m=1), "all" (all hide g, m=0), "mask_then_show" (g hidden on pass@1, visible on pass@2). Columns = ICL p; row 0 = pass@1, row 1 = pass@2; all seeds overlaid in different shades (max N, g fixed). BC omitted."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    icl = labels[1:]
    seeds = sorted({sd for (_, sd) in nets})
    xg = torch.linspace(-1, 1, 400)
    g = torch.full_like(xg, g0); ys = (xg * g).abs(); z = torch.zeros_like(xg)
    fig, axs = plt.subplots(2, len(icl), figsize=(3.4 * len(icl), 6.8), sharex=True, sharey=True)
    for c, lab in enumerate(icl):
        for sd in seeds:
            kind, net = nets[(lab, sd)]
            # (hide g on attempt 1, hide g on attempt 2); default: p=1 never saw g in training
            h1, h2 = {"default": (lab == "ICL p=1",) * 2, "none": (False, False), "all": (True, True), "mask_then_show": (True, False)}[mask]
            m1, m2 = (torch.full_like(xg, 0.0 if h else 1.0) for h in (h1, h2))
            y1 = net(icl_in(xg, g, m1, z, z)).squeeze(-1)
            y2 = net(icl_in(xg, g, m2, y1, ys - y1)).squeeze(-1)
            for r, (y, cm) in enumerate(((y1, plt.cm.Blues), (y2, plt.cm.Reds))):
                axs[r, c].plot(xg, y, color=cm(0.35 + 0.6 * sd / max(1, len(seeds) - 1)), label=f"seed {sd}")
        for r in range(2):
            ax = axs[r, c]
            ax.plot(xg, ys, color="gray", lw=3, alpha=.5, label="true label |x g|", zorder=0)
            ax.axvline(0, color="k", lw=.5); ax.grid(alpha=.3)
            ax.set_title(f"{lab}: pass@{r + 1}{dict(none=' [g visible]', all=' [g masked]', mask_then_show=(' [g masked]', ' [g visible]')[r]).get(mask, '')}\n({'first attempt, no feedback' if r == 0 else 'retry after feedback'}), g={g0:g}", fontsize=9)
            if c == 0: ax.legend(fontsize=7, loc="upper left", ncol=2)
        axs[1, c].set_xlabel("x  (train: x>0, test: x<0)")
    for r in range(2): axs[r, 0].set_ylabel("y")
    fig.tight_layout(); fig.savefig(out / f"{name}.png", dpi=110)
    print(f"saved {out / (name + '.png')}")


@torch.no_grad()
def plot_icl0_check(nets, res, N, out, g0=0.3):
    """Like the BC check plot, but for ICL p=0 pass@2: train/test MSE error bars + per-seed prediction curves (blue shades)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lab = "ICL p=0"
    seeds = sorted({sd for (l, sd) in nets if l == lab})
    xg = torch.linspace(-1, 1, 400)
    g = torch.full_like(xg, g0); ys = (xg * g).abs(); z = torch.zeros_like(xg); one = torch.ones_like(xg)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axs[0]
    for i, split in enumerate(("train", "test")):
        v = np.array(res[f"{lab}|{N}"][split]["mse2"])
        ax.errorbar(i, v.mean(), v.std(), marker="o", capsize=5, color="C0")
        ax.scatter([i] * len(v), v, color="C0", alpha=.3, s=15)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["train (x>0)", "test (x<0)"]); ax.set_yscale("log"); ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel("MSE"); ax.set_title(f"ICL p=0, pass@2 error, N={N} (mean±std, {len(seeds)} seeds)"); ax.grid(alpha=.3)
    ax = axs[1]
    shades = plt.cm.Blues(np.linspace(0.4, 0.95, len(seeds)))
    for sd in seeds:
        net = nets[(lab, sd)][1]
        y1 = net(icl_in(xg, g, one, z, z)).squeeze(-1)
        y2 = net(icl_in(xg, g, one, y1, ys - y1)).squeeze(-1)
        ax.plot(xg, y2, color=shades[sd], label=f"seed {sd}")
    ax.plot(xg, g0 * xg, "k--", label="y=g·x")
    ax.plot(xg, ys, color="gray", lw=3, alpha=.5, label="true |x g|")
    ax.axvline(0, color="k", lw=.5); ax.set_xlabel("x  (train: x>0, test: x<0)"); ax.set_ylabel("y")
    ax.set_title(f"ICL p=0 pass@2 predictions at g={g0}"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "toy_icl0_check.png", dpi=120)
    print(f"saved {out / 'toy_icl0_check.png'}")


if __name__ == "__main__":
    main(tyro.cli(Args))
