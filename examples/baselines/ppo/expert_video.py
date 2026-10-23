"""Render the PushTGoal-v1 expert (acting on the true goal) in a 2D top-down view: red = T, grey = goal, black dot = tool tip."""
import os
import sys

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon

import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from eval_obs_noise import tee_polys, yaw_of
from ppo import Agent

ckpt, out = sys.argv[1], sys.argv[2]
E, T = 8, 100
env = gym.make("PushTGoal-v1", num_envs=E, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cuda")
if isinstance(env.action_space, gym.spaces.Dict):
    env = FlattenActionSpaceWrapper(env)
base = env.unwrapped
venv = ManiSkillVectorEnv(env, E, ignore_terminations=True, record_metrics=True)
expert = Agent(venv).to(base.device)
expert.load_state_dict(torch.load(ckpt, map_location=base.device))
expert.eval()
obs, _ = venv.reset(seed=7)
pose = lambda p: torch.cat([p[:, :2], yaw_of(p)[:, None]], -1).cpu().numpy()
rec = {k: [] for k in ("tee", "goal", "tcp", "succ")}
succ_once = torch.zeros(E, dtype=torch.bool, device=base.device)
for t in range(T):
    rec["tee"].append(pose(base.tee.pose.raw_pose)); rec["goal"].append(pose(base.goal_tee.pose.raw_pose))
    rec["tcp"].append(base.agent.tcp.pose.p[:, :2].cpu().numpy())
    with torch.no_grad():
        a = expert.get_action(obs, deterministic=True)
    obs, _, _, _, info = venv.step(a)
    s = (info["final_info"] if "final_info" in info else info)["success"].bool()
    succ_once |= s
    rec["succ"].append(succ_once.cpu().numpy().copy())
rec = {k: np.stack(v) for k, v in rec.items()}
print("success_once", succ_once.float().mean().item(), "at_end", s.float().mean().item())
allxy = np.concatenate([rec["tee"][..., :2], rec["goal"][..., :2], rec["tcp"]], 1).reshape(-1, 2)
lo, hi = allxy.min(0) - 0.2, allxy.max(0) + 0.2
fig, axs = plt.subplots(2, 4, figsize=(13, 6.8))
frames = []
for t in range(T):
    for e, ax in enumerate(axs.ravel()):
        ax.clear(); ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for p in tee_polys(*rec["goal"][t, e]): ax.add_patch(Polygon(p, closed=True, fc="#c9ced4", ec="none"))
        for p in tee_polys(*rec["tee"][t, e]): ax.add_patch(Polygon(p, closed=True, fc="#d2373b", ec="none", alpha=0.85))
        ax.plot(*rec["tcp"][t, e], "o", color="#1f2328", ms=5)
        ok = rec["succ"][t, e]
        ax.set_title(f"env {e}  {'reached goal' if ok else ''}", fontsize=9, color="#1a7f37" if ok else "#1f2328")
    fig.suptitle(f"PushTGoal-v1 expert (sees true goal) | step {t}   red = T, grey = goal", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
imageio.mimsave(out, frames, fps=15)
print("wrote", out)
