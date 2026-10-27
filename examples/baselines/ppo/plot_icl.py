"""Plot icl_episodic.py runs: final-eval success_once vs the number of previous episodes h, one line per run, optionally a second
panel with the end-of-training controls (normal / shuf_y / no_y / no_ctx, line style) for the same runs (colour).

With --hs 0 1 4: one panel per h, training curves over DAgger iterations, one line per run.
With --iters instead: one panel per run, eval success_once over DAgger iterations, one line per h (from metrics.jsonl).

usage: python plot_icl.py --runs peg-final_error-hidden peg-hints-hidden --labels "true pose" hints --out peg.png [--controls]
Runs are logs/runs/icl-<name>/ (results.json). The output goes to logs/renders/<out>."""
import json
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tyro

from run_logging import LOG_ROOT

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6", "#c4a000", "#d6409f", "#52514e", "#0b8a9e"]
CONTROLS = [("normal", "-"), ("shuf_y", "--"), ("no_y", ":"), ("no_ctx", "-.")]


@dataclass
class Args:
    runs: list[str] = field(default_factory=list)
    """run names, without the icl- prefix"""
    labels: list[str] = field(default_factory=list)
    """legend labels (default: the run names)"""
    out: str = "icl.png"
    hs: list[int] = field(default_factory=list)
    """with --hs: training curves, one panel per listed h, one line per run"""
    title: str = ""
    controls: bool = False
    iters: bool = False
    """training curves over DAgger iterations (one panel per run, one line per h) instead of the final-eval plot"""
    ylabel: str = "success_once (last iteration)"


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.set_xlabel("h = number of previous episodes in context", color=MUTED)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)


def plot_by_h(args, labels):
    """One panel per h in args.hs, eval success_once over DAgger iterations, one line per run."""
    fig, axs = plt.subplots(1, len(args.hs), figsize=(4.6 * len(args.hs), 4), facecolor=SURFACE, sharey=True, squeeze=False)
    runs = [[json.loads(l) for l in open(LOG_ROOT / "runs" / f"icl-{r}" / "metrics.jsonl")] for r in args.runs]
    for ax, h in zip(axs[0], args.hs):
        for i, (rows, label) in enumerate(zip(runs, labels)):
            ax.plot([r["step"] for r in rows], [r[f"eval/success_once_h{h}"] for r in rows], lw=2, color=COLORS[i % len(COLORS)], label=label)
        ax.set_title(f"h={h} ({h} previous episodes)", fontsize=10, color=INK, loc="left")
        style(ax)
        ax.set_xlabel("DAgger iteration", color=MUTED)
    axs[0, 0].set_ylabel(args.ylabel.replace("last iteration", "per iteration"), color=MUTED)
    axs[0, 0].legend(fontsize=8, frameon=False)
    plt.tight_layout()
    out = LOG_ROOT / "renders" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=130, facecolor=SURFACE)
    print(out)


def plot_iters(args, labels):
    fig, axs = plt.subplots(1, len(args.runs), figsize=(4.6 * len(args.runs), 4), facecolor=SURFACE, sharey=True, squeeze=False)
    for ax, run, label in zip(axs[0], args.runs, labels):
        rows = [json.loads(l) for l in open(LOG_ROOT / "runs" / f"icl-{run}" / "metrics.jsonl")]
        hs = sorted(int(k.split("_h")[1]) for k in rows[0] if k.startswith("eval/success_once_h"))
        for h in hs:
            ax.plot([r["step"] for r in rows], [r[f"eval/success_once_h{h}"] for r in rows], lw=2, color=COLORS[h % len(COLORS)], label=f"h={h}")
        ax.set_title(label, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("DAgger iteration", color=MUTED)
        style(ax)
        ax.set_xlabel("DAgger iteration", color=MUTED)
    axs[0, 0].set_ylabel(args.ylabel.replace("last iteration", "per iteration"), color=MUTED)
    axs[0, 0].legend(fontsize=8, frameon=False, title="previous episodes", title_fontsize=8)
    plt.tight_layout()
    out = LOG_ROOT / "renders" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=130, facecolor=SURFACE)
    print(out)


def main():
    args = tyro.cli(Args)
    labels = args.labels or args.runs
    if args.hs:
        return plot_by_h(args, labels)
    if args.iters:
        return plot_iters(args, labels)
    res = [json.load(open(LOG_ROOT / "runs" / f"icl-{r}" / "results.json")) for r in args.runs]
    fig, axs = plt.subplots(1, 2 if args.controls else 1, figsize=(11 if args.controls else 6, 4.2), facecolor=SURFACE, sharey=True, squeeze=False)
    ax = axs[0, 0]
    for i, (r, label) in enumerate(zip(res, labels)):
        ev = r["final_eval"]
        ax.plot([e["h"] for e in ev], [e["success_once"] for e in ev], marker="o", ms=6, lw=2, color=COLORS[i % len(COLORS)], label=label)
    ref = res[0]["refs"]
    ax.axhline(ref["expert_true_goal"]["success_once"], color=INK, ls="--", lw=1, label="expert (true task)")
    ax.axhline(ref["expert_default_goal"]["success_once"], color=MUTED, ls=":", lw=1.5, label="expert (default task)")
    ax.set_title(args.title, fontsize=10, color=INK, loc="left")
    ax.set_ylabel(args.ylabel, color=MUTED)
    ax.legend(fontsize=8, frameon=False)
    style(ax)
    if args.controls:
        ax = axs[0, 1]
        ax.set_title("Controls (end of training)", fontsize=10, color=INK, loc="left")
        for i, r in enumerate(res):
            for cm, ls in CONTROLS:
                ev = r["controls"][cm]
                ax.plot([e["h"] for e in ev], [e["success_once"] for e in ev], ls=ls, lw=1.8, color=COLORS[i % len(COLORS)], marker="o", ms=3)
        for cm, ls in CONTROLS:
            ax.plot([], [], color=MUTED, ls=ls, label=cm)
        ax.legend(fontsize=8, frameon=False, title="line style (colour = run)", title_fontsize=8)
        style(ax)
    for ax in axs.flat:
        ax.set_xticks(range(len(res[0]["final_eval"])))
    plt.tight_layout()
    out = LOG_ROOT / "renders" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=130, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
