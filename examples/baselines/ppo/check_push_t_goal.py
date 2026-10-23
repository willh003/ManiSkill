"""Sanity checks for PushTGoal-v1: obs layout, goal randomization, success when the T is placed on the goal."""
import gymnasium as gym
import torch

import mani_skill.envs
from mani_skill.utils.structs import Pose
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

n = 16
env = gym.make("PushTGoal-v1", num_envs=n, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cuda")
base = env.unwrapped
obs, _ = env.reset(seed=0)
print("obs shape", obs.shape)
assert torch.allclose(obs[:, -7:], base.tee.pose.raw_pose, atol=1e-4), "last 7 obs != T pose"
assert torch.allclose(obs[:, 21:24], base.goal_tee.pose.p, atol=1e-4), "goal_pos not at 21:24"
assert torch.allclose(obs[:, 24], base.goal_yaw.cos(), atol=1e-5) and torch.allclose(obs[:, 25], base.goal_yaw.sin(), atol=1e-5), "goal cos/sin not at 24:26"
print("goal delta std", base.goal_delta.std(0).tolist())

# fixed goal via options: all envs the same goal, T spawns still differ
g = torch.tensor([[0.02, -0.01, 0.5]], device=base.device).repeat(n, 1)
obs, _ = env.reset(seed=1, options=dict(goal=g))
assert torch.allclose(base.goal_delta, g)
print("T spawn std (xy)", base.tee.pose.p[:, :2].std(0).tolist())

# put the T exactly on the goal -> success, zero error
p = base.goal_tee.pose.p.clone(); p[:, 2] = 0.021
base.tee.set_pose(Pose.create_from_pq(p=p, q=base.goal_tee.pose.q))
info = base.evaluate()
print("success with T on goal:", info["success"].float().mean().item(), "error:", base.tee_goal_error()[0].tolist())
# shifted T by 2cm should reduce the overlap
p[:, 0] += 0.02
base.tee.set_pose(Pose.create_from_pq(p=p, q=base.goal_tee.pose.q))
print("success with 2cm shift:", base.evaluate()["success"].float().mean().item())
# goal pose must also be respected when the goal is rotated: random goals, T on goal -> all succeed
obs, _ = env.reset(seed=2)
p = base.goal_tee.pose.p.clone(); p[:, 2] = 0.021
base.tee.set_pose(Pose.create_from_pq(p=p, q=base.goal_tee.pose.q))
print("random goals, T on goal success:", base.evaluate()["success"].float().mean().item())
r = base.compute_dense_reward(obs, None, base.evaluate())
print("reward at goal (should be 3):", r.mean().item())
