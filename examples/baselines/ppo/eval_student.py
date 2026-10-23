"""Re-evaluate a saved distill_recurrent.py student and report per-episode success (to check the overlay video).

Example:
    python eval_student.py --run-dir runs/pusht-distill-noisyR6
"""
import faulthandler
import json
import os
from dataclasses import dataclass, fields

import gymnasium as gym
import torch
import tyro

from distill_recurrent import Args, Student, run_episodes
from eval_obs_noise import render_video
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from ppo import Agent

faulthandler.cancel_dump_traceback_later()


@dataclass
class EvalArgs:
    run_dir: str
    ckpt: str = "student_latest.pt"
    repeats: int = 2
    """evaluate this many times with the same seed to check determinism"""
    video_successes: int = 0
    """render a video mixing this many successful episodes ..."""
    video_failures: int = 0
    """... and this many failed ones (the first of each by episode index, from one rollout of --video-pool episodes)"""
    video_pool: int = 64
    video_name: str = "student_overlay_mix"


def main():
    ea = tyro.cli(EvalArgs)
    cfg = json.load(open(os.path.join(ea.run_dir, "log.json")))["args"]
    args = Args(**{f.name: cfg[f.name] for f in fields(Args) if f.name in cfg})
    device = torch.device("cuda")
    env = gym.make(args.env_id, num_envs=args.num_envs, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cuda")
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    base = env.unwrapped
    venv = ManiSkillVectorEnv(env, args.num_envs, ignore_terminations=True, record_metrics=True)
    obs_dim, act_dim = venv.single_observation_space.shape[0], venv.single_action_space.shape[0]
    expert = Agent(venv).to(device)
    expert.load_state_dict(torch.load(args.expert_checkpoint, map_location=device))
    expert.eval()
    in_dim = obs_dim if args.memoryless else obs_dim + act_dim + (act_dim if args.expert_in_history else 0) + 1
    student = Student(in_dim, act_dim, args.hidden, args.memoryless).to(device)
    student.load_state_dict(torch.load(os.path.join(ea.run_dir, ea.ckpt), map_location=device))
    student.eval()
    never = torch.full((args.num_envs,), args.num_steps, device=device)
    runs = []
    for i in range(ea.repeats):
        r = run_episodes(venv, base, expert, student, args, args.eval_seed, never)
        runs.append(r["succ_once_env"])
        e = r["succ_once_env"]
        print(f"repeat {i}: success_once={r['success_once']:.3f} first8={int(e[:8].sum())}/8 first64={int(e[:64].sum())}/64 "
              f"first8_idx={e[:8].nonzero().flatten().tolist()}", flush=True)
    if len(runs) > 1:
        print(f"episodes whose success differs between repeats: {int((runs[0] != runs[1]).sum())} / {args.num_envs}", flush=True)
    # success rate of consecutive blocks of 8 episodes: how unusual is a block with <= 2 successes?
    if runs:
        blocks = runs[0].reshape(-1, 8).sum(1)
        print(f"blocks of 8 with <=2 successes: {int((blocks <= 2).sum())} / {len(blocks)}", flush=True)
    if ea.video_successes + ea.video_failures:
        import imageio
        import numpy as np
        r = run_episodes(venv, base, expert, student, args, args.eval_seed, never, rec_n=ea.video_pool)
        rec = r["rec"]
        ok = rec["success"].any(0)  # success_once per recorded episode
        idx = np.r_[np.flatnonzero(ok)[: ea.video_successes], np.flatnonzero(~ok)[: ea.video_failures]]
        idx.sort()
        print(f"video pool: {int(ok.sum())}/{len(ok)} succeeded; showing episodes {idx.tolist()} "
              f"({int(ok[idx].sum())} successes, {int((~ok[idx]).sum())} failures)", flush=True)
        sel = {k: v[:, idx] for k, v in rec.items()}
        path = os.path.join(ea.run_dir, ea.video_name + ".mp4")
        frames = render_video(sel, path, f"student only, noisy obs | {int(ok[idx].sum())} successes, {int((~ok[idx]).sum())} failures")
        imageio.imwrite(os.path.join(ea.run_dir, ea.video_name + "_final.png"), frames[-1])
        print(f"wrote {path}", flush=True)
    venv.close()


if __name__ == "__main__":
    main()
