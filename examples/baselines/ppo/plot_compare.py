"""Compare several distill_recurrent.py runs (same eval episodes) in one figure.

Example:
    python plot_compare.py --run-dirs runs/pusht-distill-noRew runs/pusht-distill-noisyR6 runs/pusht-distill-expHist \
        --labels "no reward" "with reward" "+ expert's previous action" --out runs/distill_comparison.png
"""
import json
import os
from dataclasses import dataclass
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro
from matplotlib.lines import Line2D

INK, MUTED, GRID, REF = "#1f2328", "#656d76", "#e6e9ed", "#8b949e"
PPO_COLOR = "#CC79A7"  # Okabe-Ito reddish purple
# Okabe-Ito colors (colorblind-safe), plus a distinct marker per series so identity isn't color-only
STYLES = [("#E69F00", "s", (0, (6, 2))), ("#0072B2", "o", "-"), ("#009E73", "^", (0, (1, 1.5)))]


@dataclass
class Args:
    run_dirs: List[str]
    labels: List[str]
    out: str = "distill_comparison.png"
    ppo_run: str = ""
    """optional ppo_noisy_obs.py run dir (tensorboard logs): its final eval is drawn as a horizontal reference line"""
    title: str = "Recurrent student under noisy T-pose observations"
    """first-run settings are used for the subtitle; all runs must share the eval episodes/baselines"""


def main():
    args = tyro.cli(Args)
    assert len(args.run_dirs) == len(args.labels) <= len(STYLES)
    runs = [json.load(open(os.path.join(d, "log.json"))) for d in args.run_dirs]
    base, cfg = runs[0]["baselines"], runs[0]["args"]
    for r in runs[1:]:
        assert r["baselines"] == base, "runs were evaluated on different episodes"

    ppo = None
    if args.ppo_run:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(args.ppo_run, size_guidance={"scalars": 0})
        ea.Reload()
        ppo = [ea.Scalars(t)[-1].value for t in ("eval/success_once", "eval/success_at_end")]  # final evaluation

    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.3), sharey=True, facecolor="white")
    panels = [("student_success_once", 0, "Success at any step"), ("student_success_at_end", 1, "Success at the final step")]
    for ax, (key, bi, title) in zip(axs, panels):
        for y, name, ls in ((base["expert_clean"][bi], "expert, true state", (0, (1, 2))),
                            (base["expert_noisy"][bi], "expert, noisy obs", (0, (5, 3)))):
            ax.axhline(y, color=REF, lw=1.4, ls=ls, zorder=1)
            ax.text(20.45, y + 0.012, name, color=MUTED, fontsize=9, ha="right", va="bottom")
        if ppo is not None:
            ax.axhline(ppo[bi], color=PPO_COLOR, lw=1.8, zorder=2)
            ax.text(1.0, ppo[bi] - 0.012, f"PPO, no memory, noisy obs (50M steps)  {ppo[bi]:.3f}", color=PPO_COLOR, fontsize=9,
                    ha="left", va="top", fontweight="bold")
        for r, (color, marker, ls) in zip(runs, STYLES):
            its = np.array([row["iteration"] for row in r["log"]])
            ys = np.array([row[key] for row in r["log"]])
            ax.plot(its, ys, color=color, lw=2.2, ls=ls, marker=marker, ms=5.5, mfc=color, mec="white", mew=1.1, zorder=3)
            ax.text(its[-1] + 0.35, ys[-1], f"{ys[-1]:.2f}", color=color, fontsize=10, fontweight="bold", va="center")
        ax.set_title(title, loc="left", fontsize=12, color=INK, pad=8)
        ax.set_xlim(0.4, 21.6)
        ax.set_ylim(-0.06, 1.04)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_xlabel("DAgger iteration")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(length=0)
    axs[0].set_ylabel("episodes succeeding")

    handles = [Line2D([0], [0], color=c, lw=2.2, ls=ls, marker=m, ms=6, mfc=c, mec="white", label=lab)
               for (c, m, ls), lab in zip(STYLES, args.labels)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.062, 0.875), ncol=3, frameon=False, fontsize=10.5,
               handlelength=3.2, columnspacing=2.2)
    fig.suptitle(args.title, x=0.062, y=0.985, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.062, 0.915, f"offset up to ±{cfg['noise_xy'] * 100:g} cm and ±{cfg['noise_theta_deg']:g}°  ·  "
             f"{cfg['num_envs']} eval episodes per point  ·  same recipe, one seed", ha="left", fontsize=10.5, color=MUTED)
    fig.subplots_adjust(left=0.062, right=0.985, top=0.765, bottom=0.125, wspace=0.07)
    fig.savefig(args.out, dpi=170)
    print(args.out)


if __name__ == "__main__":
    main()
