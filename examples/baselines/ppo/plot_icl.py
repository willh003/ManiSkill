"""Plot icl_episodic.py runs: final-eval success_once vs the number of previous episodes h, one line per run, optionally a second
panel with the end-of-training controls (normal / shuf_y / no_y / no_ctx, line style) for the same runs (colour).

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
    title: str = ""
    controls: bool = False
    ylabel: str = "success_once (last iteration)"


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.set_xlabel("h = number of previous episodes in context", color=MUTED)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)


def main():
    args = tyro.cli(Args)
    labels = args.labels or args.runs
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
