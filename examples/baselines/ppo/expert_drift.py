"""What does the expert do after it first reaches the goal? Rolls out without termination and reports post-success statistics.
usage: python expert_drift.py <env_id> <checkpoint> <out.json>"""
import json
import sys

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from ppo import Agent

env_id, ckpt, out = sys.argv[1:4]
N, T = 256, 100
env = gym.make(env_id, num_envs=N, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cuda")
if isinstance(env.action_space, gym.spaces.Dict):
    env = FlattenActionSpaceWrapper(env)
base = env.unwrapped
venv = ManiSkillVectorEnv(env, N, ignore_terminations=True, record_metrics=True)
expert = Agent(venv).to(base.device)
expert.load_state_dict(torch.load(ckpt, map_location=base.device))
expert.eval()
obs, _ = venv.reset(seed=11)
inter, succ, rew, anorm, tee_speed = [], [], [], [], []
prev_tee = base.tee.pose.p[:, :2].clone()
for t in range(T):
    with torch.no_grad():
        a = expert.get_action(obs, deterministic=True)
    obs, r, _, _, info = venv.step(a)
    # on the last step the env auto-resets; read state before that only for t < T-1
    s = (info["final_info"] if "final_info" in info else info)["success"].bool()
    if t < T - 1:
        inter.append(base.pseudo_render_intersection().cpu().numpy())
        tee = base.tee.pose.p[:, :2]
        tee_speed.append((tee - prev_tee).norm(dim=1).cpu().numpy()); prev_tee = tee.clone()
    else:
        inter.append(inter[-1]); tee_speed.append(tee_speed[-1])
    succ.append(s.cpu().numpy()); rew.append(r.cpu().numpy()); anorm.append(a.norm(dim=1).cpu().numpy())
succ, inter, rew, anorm, tee_speed = (np.stack(x, 1) for x in (succ, inter, rew, anorm, tee_speed))  # (N, T)
ever = succ.any(1)
first = np.where(ever, succ.argmax(1), T)
after = np.arange(T)[None] > first[:, None]  # steps strictly after first success
exits = ((succ[:, :-1]) & (~succ[:, 1:])).sum(1)
enters = ((~succ[:, :-1]) & (succ[:, 1:])).sum(1) + succ[:, 0]
res = dict(
    env=env_id, ckpt=ckpt,
    success_once=float(ever.mean()), success_at_end=float(succ[:, -1].mean()),
    first_success_step_mean=float(first[ever].mean()), first_success_step_median=float(np.median(first[ever])),
    frac_post_success_steps_in_success=float(succ[after].mean()),
    mean_exits_per_episode=float(exits[ever].mean()), mean_entries_per_episode=float(enters[ever].mean()),
    frac_episodes_ending_out_given_success_once=float((~succ[:, -1])[ever].mean()),
    mean_intersection_post_success=float(inter[after].mean()),
    action_norm_pre_success=float(anorm[~after & (np.arange(T)[None] <= first[:, None])].mean()),
    action_norm_post_success=float(anorm[after].mean()),
    tee_speed_pre_success=float(tee_speed[np.arange(T)[None] <= first[:, None]].mean()),
    tee_speed_post_success=float(tee_speed[after].mean()),
    # normalized per-step reward: in success = 1.0 by construction; outside it is the dense shaping term
    mean_reward_in_success=float(rew[succ].mean()) if succ.any() else None,
    mean_reward_outside_success_post_first=float(rew[after & ~succ].mean()) if (after & ~succ).any() else None,
    max_reward_outside_success=float(rew[~succ].max()),
    # fraction of post-success time spent in each phase by time since first success (10-step bins)
    in_success_by_steps_since_first=[float(succ[(np.arange(T)[None] - first[:, None]) // 10 == b].mean()) for b in range(1, 6)],
)
print(json.dumps(res, indent=1))
json.dump(res, open(out, "w"), indent=1)
