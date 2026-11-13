"""Spatial success map of the offline BC N=500 policies (offline-vis-ckpt-s{1,2,3}/checkpoints/bc-500.pt) on fresh goals over the whole workspace."""
import sys
import numpy as np
import torch
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from icl_checker import Policy, ctx_index, empty_ctx, rollout
from icl_offline import Args
from icl_tasks import checker_sample
from ppo import Agent
from run_logging import LOG_ROOT

N, BATCHES, NAME = 512, 4, sys.argv[1] if len(sys.argv) > 1 else "bc-500"
dev = torch.device("cuda")
args = Args(sizes="500")
env = gym.make("PegInsertionSideBias-v1", num_envs=N, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cuda")
if isinstance(env.action_space, gym.spaces.Dict):
    env = FlattenActionSpaceWrapper(env)
venv = ManiSkillVectorEnv(env, N, ignore_terminations=True, record_metrics=True)
A = venv.single_action_space.shape[0]
class _S:
    single_observation_space = type("S", (), {"shape": (43,)})
    single_action_space = venv.single_action_space
expert = Agent(_S).to(dev); expert.load_state_dict(torch.load(args.expert_checkpoint, map_location=dev)); expert.eval()
Tc = len(ctx_index(args.num_steps, args.ctx_stride, dev))
g = torch.Generator(device=dev).manual_seed(777)
goals = [checker_sample(N, "all", dev, g, args.split) for _ in range(BATCHES)]
succ = []  # [seed][batch] -> (N,)
for s in (1, 2, 3):
    pol = Policy(A, Tc, args).to(dev)
    pol.load_state_dict(torch.load(LOG_ROOT / f"runs/offline-vis-ckpt-s{s}/checkpoints/{NAME}.pt", map_location=dev)); pol.eval()
    succ.append(torch.cat([rollout(venv, expert, pol, args, gl, 5000 + b, "student", empty_ctx(N, Tc, A, dev))["succ"] for b, gl in enumerate(goals)]).cpu().numpy())
exp = torch.cat([rollout(venv, expert, None, args, gl, 5000 + b, "expert")["succ"] for b, gl in enumerate(goals)]).cpu().numpy()
G = torch.cat(goals).cpu().numpy(); S = np.array(succ).astype(float)
print("pass@1 per seed", S.mean(1), "expert", exp.mean())
np.savez(LOG_ROOT / f"renders/offline/spatial_{NAME}.npz", goals=G, succ=S, expert=exp)

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for a, (m, ttl) in zip(ax[:2], [(S[0], "seed 1"), (S.mean(0), "mean over 3 seeds (success fraction)")]):
    sc = a.scatter(G[:, 0], G[:, 1], c=m, cmap="RdYlGn", vmin=0, vmax=1, s=14)
    a.set_xlabel("hole x"); a.set_ylabel("hole y"); a.set_title(f"{NAME}: {ttl}"); a.set_aspect("equal")
plt.colorbar(sc, ax=ax[:2], label="success")
# success rate vs each coordinate (binned), policy vs expert
for k, (j, lab) in enumerate([(0, "x"), (1, "y"), (2, "yaw")]):
    pass
b = ax[2]
for j, lab, c in [(0, "x", "C0"), (1, "y", "C1"), (2, "yaw", "C2")]:
    u = (G[:, j] - G[:, j].min()) / (G[:, j].max() - G[:, j].min()); bins = np.minimum((u * 8).astype(int), 7)
    b.plot(np.arange(8) / 8 + 1 / 16, [S.mean(0)[bins == i].mean() for i in range(8)], "o-", c=c, label=f"{lab}")
    b.plot(np.arange(8) / 8 + 1 / 16, [exp[bins == i].mean() for i in range(8)], "--", c=c, alpha=.5)
b.set_xlabel("normalised coordinate (low -> high)"); b.set_ylabel("success rate (solid: policy, dashed: expert)"); b.set_ylim(0, 1); b.legend(); b.set_title("success vs each goal coordinate")
fig.tight_layout(); out = LOG_ROOT / f"renders/offline/spatial_{NAME}.png"; fig.savefig(out, dpi=130); print(out)
