"""Plot the student success curve of a distill_recurrent.py run from its log.json.

Example:
    python plot_distill.py --run-dir runs/pusht-distill-noisyR6
"""
import json
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro

INK, MUTED, GRID = "#1f2328", "#656d76", "#e6e9ed"
BLUE, ORANGE, REF = "#2a6fdb", "#d9701f", "#8b949e"


@dataclass
class Args:
    run_dir: str
    """directory containing log.json (written by distill_recurrent.py)"""
    out: str = ""
    """output png (default: <run_dir>/success_curve.png)"""
    title: str = ""
    """plot title (default: derived from the run's settings)"""


def main():
    args = tyro.cli(Args)
    d = json.load(open(os.path.join(args.run_dir, "log.json")))
    log, base, cfg = d["log"], d["baselines"], d["args"]
    its = np.array([r["iteration"] for r in log])
    once = np.array([r["student_success_once"] for r in log])
    end = np.array([r["student_success_at_end"] for r in log])

    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor="white")
    ax.set_facecolor("white")

    # reference levels: the expert with the true state, and with the same corrupted observation as the student
    ref = [("expert, true state", base["expert_clean"][0], (0, (1, 2))),
           ("expert, noisy obs", base["expert_noisy"][0], (0, (5, 3)))]
    for label, y, ls in ref:
        ax.axhline(y, color=REF, lw=1.4, ls=ls, zorder=1)
        ax.text(its[-1] + 0.35, y, label, color=MUTED, fontsize=9.5, va="center", ha="left")

    ax.plot(its, once, color=BLUE, lw=2.4, marker="o", ms=6, mfc=BLUE, mec="white", mew=1.4, zorder=3)
    ax.plot(its, end, color=ORANGE, lw=2.4, marker="s", ms=6, mfc=ORANGE, mec="white", mew=1.4, ls=(0, (6, 2)), zorder=3)
    # direct labels at the right end, so the legend isn't needed to tell the series apart
    ax.text(its[-1] + 0.35, once[-1], f"student, any step\n{once[-1]:.2f}", color=BLUE, fontsize=10,
            fontweight="bold", va="center", ha="left", linespacing=1.15)
    ax.text(its[-1] + 0.35, end[-1], f"student, at end\n{end[-1]:.2f}", color=ORANGE, fontsize=10,
            fontweight="bold", va="center", ha="left", linespacing=1.15)

    ax.set_xlim(its[0] - 0.4, its[-1] + 0.3)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(its if len(its) <= 12 else np.r_[its[0], np.arange(5, its[-1] + 1, 5)])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("DAgger iteration")
    ax.set_ylabel("episodes succeeding")
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(length=0)

    title = args.title or "Recurrent student under noisy T-pose observations"
    sub = (f"offset up to ±{cfg['noise_xy'] * 100:g} cm and ±{cfg['noise_theta_deg']:g}°"
           if cfg.get("noise_scale", 1.0) else "no observation noise")
    fig.suptitle(title, x=0.075, y=0.985, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.075, 0.915, f"{sub}  ·  {cfg['num_envs']} eval episodes per point", ha="left", fontsize=10.5, color=MUTED)
    fig.subplots_adjust(left=0.085, right=0.76, top=0.86, bottom=0.14)
    out = args.out or os.path.join(args.run_dir, "success_curve.png")
    fig.savefig(out, dpi=170)
    print(out)


if __name__ == "__main__":
    main()
