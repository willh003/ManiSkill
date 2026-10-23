"""PushT-v1 vs PushTGoal-v1 (zero goal range): inject identical T poses, compare reward / success / obs."""
import os
os.environ["PUSHT_GOAL_POS_RANGE"] = "0"; os.environ["PUSHT_GOAL_YAW_DEG"] = "0"
import gymnasium as gym, torch
import mani_skill.envs
from mani_skill.utils.structs import Pose
n = 64
def make(i): return gym.make(i, num_envs=n, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cuda")
a, b = make("PushT-v1"), make("PushTGoal-v1")
a.reset(seed=0); b.reset(seed=0)
A, B = a.unwrapped, b.unwrapped
print("goal pose equal:", torch.allclose(A.goal_tee.pose.raw_pose, B.goal_tee.pose.raw_pose, atol=1e-5), A.goal_tee.pose.raw_pose[0].tolist(), B.goal_tee.pose.raw_pose[0].tolist())
print("ee_goal pose equal:", torch.allclose(A.ee_goal_pos.pose.raw_pose, B.ee_goal_pos.pose.raw_pose, atol=1e-5))
print("tcp equal:", torch.allclose(A.agent.tcp.pose.raw_pose, B.agent.tcp.pose.raw_pose, atol=1e-5))
g = torch.Generator(device=A.device).manual_seed(3)
# T poses: random near goal and random anywhere
p = A.goal_tee.pose.p.clone(); p[:, 2] = 0.021
p[:, :2] += (torch.rand(n, 2, device=A.device, generator=g) - 0.5) * 0.08
yaw = A.goal_z_rot + (torch.rand(n, device=A.device, generator=g) - 0.5) * 1.5
q = torch.zeros(n, 4, device=A.device); q[:, 0] = (yaw / 2).cos(); q[:, 3] = (yaw / 2).sin()
for E in (A, B): E.tee.set_pose(Pose.create_from_pq(p=p, q=q))
ia, ib = A.evaluate(), B.evaluate()
ra = A.compute_dense_reward(None, None, ia); rb = B.compute_dense_reward(None, None, ib)
print("success equal:", (ia["success"] == ib["success"]).all().item(), "n success", ia["success"].sum().item(), ib["success"].sum().item())
print("max |reward diff|:", (ra - rb).abs().max().item(), "mean reward", ra.mean().item())
print("max |obs diff| T pose:", (A.get_obs()[:, 24:] - B.get_obs()[:, 26:]).abs().max().item())
# dynamics: step both with identical actions from identical state
act = torch.rand(n, a.action_space.shape[-1], device=A.device, generator=g) * 2 - 1
for t in range(20): oa, *_ = a.step(act); ob, *_ = b.step(act)
print("after 20 steps |obs diff| robot/T:", (oa[:, :21] - ob[:, :21]).abs().max().item(), (oa[:, 24:] - ob[:, 26:]).abs().max().item())
