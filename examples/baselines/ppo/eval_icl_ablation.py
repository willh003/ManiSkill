"""Eval-only ablation of a trained language student: normal / no language (no_y: the student's own episodes stay in the context, the command is
blanked) / random language (rand_y: random well-formed commands in the range of the real ones) / shuffled (another task's command) / no context,
over several eval seeds (each seed = fresh eval tasks).

usage: python eval_icl_ablation.py --run peg-language-hidden-strong
Writes logs/runs/icl-<run>/ablation.json and logs/renders/icl-peg/<out>."""
import json
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
import tyro

import icl_episodic as ie
from icl_tasks import TASKS
from ppo import Agent
from run_logging import LOG_ROOT
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


@dataclass
class Cfg:
    run: str = "peg-language-hidden-strong"
    expert: str = str(LOG_ROOT / "runs/peg-expert-r1/ckpt_3576.pt")
    seeds: list[int] = field(default_factory=lambda: [12345, 1, 2])
    modes: list[str] = field(default_factory=lambda: ["normal", "no_y", "rand_y", "shuf_y", "no_ctx"])
    out: str = "peg_language_ablation.png"


def main():
    cfg = tyro.cli(Cfg)
    run_dir = LOG_ROOT / "runs" / f"icl-{cfg.run}"
    raw = json.load(open(run_dir / "config.json"))["args"]
    args = ie.Args(**{k: v for k, v in raw.items() if k in ie.Args.__dataclass_fields__})
    assert args.feedback == "language"
    ie.TASK = TASKS[args.env_id](hide_hole=args.hide_hole, full_info=args.full_info)
    TASK, device = ie.TASK, torch.device("cuda")
    env = gym.make(args.env_id, num_envs=args.num_envs, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cuda")
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    base = env.unwrapped
    venv = ManiSkillVectorEnv(env, args.num_envs, ignore_terminations=True, record_metrics=True)
    obs_dim, act_dim = venv.single_observation_space.shape[0], venv.single_action_space.shape[0]

    class _Space:
        single_observation_space = type("S", (), {"shape": (TASK.expert_obs(torch.zeros(1, obs_dim)).shape[-1],)})
        single_action_space = venv.single_action_space

    expert = Agent(_Space).to(device)
    expert.load_state_dict(torch.load(cfg.expert, map_location=device))
    expert.eval()
    s_dim, y_dim = ie.student_obs(torch.zeros(1, obs_dim)).shape[-1], ie.feedback_dim(args)
    policy = ie.ICLPolicy(s_dim, act_dim, y_dim, args).to(device)
    policy.load_state_dict(torch.load(run_dir / "checkpoints" / "policy_final.pt", map_location=device))
    policy.eval()

    res = {m: [] for m in cfg.modes}  # mode -> seed -> success_once by h
    for seed in cfg.seeds:
        with torch.random.fork_rng():
            torch.manual_seed(seed + 1000)
            goals = TASK.sample(base, args.num_envs)
        for m in ["normal"] + [m for m in cfg.modes if m != "normal"]:  # normal first: it fills the range of real commands for rand_y
            torch.manual_seed(seed)
            eps, _ = ie.collect_task_episodes(venv, expert, policy, args, goals, seed, "eval", m)
            res[m].append([e["success_once"].float().mean().item() for e in eps])
            print(f"seed {seed} {m:8s}", " ".join(f"{x:.3f}" for x in res[m][-1]), flush=True)
    summary = {m: dict(mean=np.mean(v, 0).tolist(), std=np.std(v, 0).tolist(), per_seed=v) for m, v in res.items()}
    json.dump(dict(seeds=cfg.seeds, cmd_range=[r.tolist() for r in TASK.cmd_range], results=summary), open(run_dir / "ablation.json", "w"), indent=1)
    print("mean over seeds (std):")
    for m, v in summary.items():
        print(f"  {m:8s}", " ".join(f"{a:.2f}({b:.2f})" for a, b in zip(v["mean"], v["std"])))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = dict(normal="true commands", no_y="no language", rand_y="random language", shuf_y="another task's language", no_ctx="no context at all")
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for m, v in summary.items():
        ax.errorbar(range(len(v["mean"])), v["mean"], yerr=v["std"], marker="o", ms=5, lw=2, capsize=3, label=names.get(m, m))
    ax.set_xlabel("h = previous episodes in context")
    ax.set_ylabel(f"success_once (mean +- std over {len(cfg.seeds)} eval seeds)")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(len(summary["normal"]["mean"])))
    ax.set_title("Language student (hand-built vocab): eval ablation", loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    plt.tight_layout()
    out = LOG_ROOT / "renders" / "icl-peg" / cfg.out
    plt.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
