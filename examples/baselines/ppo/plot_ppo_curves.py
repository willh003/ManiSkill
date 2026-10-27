"""PPO training curves (eval success vs environment steps) from TensorBoard logs.

usage: python plot_ppo_curves.py --runs pick-expert peg-expert-fast --out ppo_experts.png [--tag eval/success_once]
Runs are logs/runs/<name>/ (TensorBoard event files directly inside, as written by ppo.py / ppo_fast.py). Output: logs/renders/<out>."""
import glob
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tyro
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from run_logging import LOG_ROOT


@dataclass
class Args:
    runs: list[str] = field(default_factory=list)
    out: str = "ppo_curves.png"
    tag: str = "eval/success_once"


def main():
    args = tyro.cli(Args)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for r in args.runs:
        events = sorted(glob.glob(str(LOG_ROOT / "runs" / r / "events.out.tfevents.*")))
        ea = EventAccumulator(events[-1])
        ea.Reload()
        pts = ea.Scalars(args.tag)
        ax.plot([p.step / 1e6 for p in pts], [p.value for p in pts], lw=2, label=r)
    ax.set_xlabel("environment steps (M)")
    ax.set_ylabel(args.tag)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    out = LOG_ROOT / "renders" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
