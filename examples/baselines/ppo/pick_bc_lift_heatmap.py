"""Success/failure map for lift >= 5 cm (cube max z over 50 steps) of the PickCube BC policies vs initial cube xy and goal height.
8 reset seeds x 512 resets, 3 policy seeds pooled; logs/renders/pick-bc/lift5_heatmap.png"""
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from run_logging import LOG_ROOT

N, T, THR, SEEDS = 512, 50, 0.05, [12345 + k for k in range(8)]
dev = torch.device("cuda")
env = gym.make("PickCube-v1", num_envs=N, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cuda")
if isinstance(env.action_space, gym.spaces.Dict):
    env = FlattenActionSpaceWrapper(env)
venv = ManiSkillVectorEnv(env, N, ignore_terminations=True, record_metrics=True)
base = venv.base_env
S, A = venv.single_observation_space.shape[0], venv.single_action_space.shape[0]
low, high = (torch.from_numpy(b).to(dev) for b in (venv.single_action_space.low, venv.single_action_space.high))


def run(pol, seed):
    obs, _ = venv.reset(seed=seed)
    cube0, goal = base.cube.pose.p[:, :2].clone(), base.goal_site.pose.p.clone()
    zmax = base.cube.pose.p[:, 2].clone()
    for _ in range(T):
        with torch.no_grad():
            obs, *_ = venv.step(torch.max(torch.min(pol(obs), high), low))
        zmax = torch.max(zmax, base.cube.pose.p[:, 2])
    return cube0.cpu().numpy(), goal.cpu().numpy(), (zmax >= THR).cpu().numpy()


sizes = [10, 50, 100, 200]
fig, ax = plt.subplots(2, 4, figsize=(19, 8), constrained_layout=True)
for i, n in enumerate(sizes):
    C, G, O = [], [], []
    for s in (1, 2, 3):
        ck = torch.load(LOG_ROOT / f"runs/pick-bc/checkpoints/bc-{n}-s{s}.pt", map_location=dev)
        net = nn.Sequential(nn.Linear(S, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, A)).to(dev)
        net.load_state_dict(ck["net"])
        for sd in SEEDS:
            c, g, o = run(lambda x: net((x - ck["mu"]) / ck["sd"]), sd)
            C.append(c), G.append(g), O.append(o)
    C, G, O = np.concatenate(C), np.concatenate(G), np.concatenate(O).astype(float)
    H, xe, ye = np.histogram2d(C[:, 0], C[:, 1], (8, 8), weights=O); Hc, _, _ = np.histogram2d(C[:, 0], C[:, 1], (8, 8))
    im = ax[0, i].imshow((H / np.maximum(Hc, 1)).T, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap="RdYlGn", vmin=0, vmax=1)
    ax[0, i].set_xlabel("initial cube x"); ax[0, i].set_ylabel("initial cube y"); ax[0, i].set_title(f"{n} demos: lift >= {THR*100:.0f} cm, rate {O.mean():.2f}")
    bins = np.linspace(G[:, 2].min(), G[:, 2].max(), 9); idx = np.clip(np.digitize(G[:, 2], bins) - 1, 0, 7)
    ax[1, i].plot([G[idx == b, 2].mean() for b in range(8)], [O[idx == b].mean() for b in range(8)], "o-"); ax[1, i].set_ylim(0, 1)
    ax[1, i].set_xlabel("goal height z (m)"); ax[1, i].set_ylabel("lift success rate"); ax[1, i].set_title("success vs goal height")
    print(n, f"rate {O.mean():.3f}; n={len(O)}", flush=True)
fig.colorbar(im, ax=ax[0, :], label="success rate (green = lifts, red = fails)", shrink=0.8)
fig.suptitle(f"PickCube BC, lift >= {THR*100:.0f} cm, 3 policy seeds x {len(SEEDS)*N} resets pooled")
fig.savefig(LOG_ROOT / "renders/pick-bc/lift5_heatmap.png", dpi=110)
