"""Re-eval of the PickCube BC checkpoints (logs/runs/pick-bc) with a LIFT criterion: the cube's max z over the episode >= threshold (ignores the goal).
Also reports the original goal-reaching success and the PPO expert as reference. 512 fixed resets (eval seed), 3 policy seeds."""
import json
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
from ppo import Agent
from run_logging import LOG_ROOT

N, T, EVAL_SEED = 512, 50, 12345
THRESH = [0.05, 0.10, 0.15, 0.20]  # cube z (m) above the table-top rest height 0.02
dev = torch.device("cuda")
env = gym.make("PickCube-v1", num_envs=N, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cuda")
if isinstance(env.action_space, gym.spaces.Dict):
    env = FlattenActionSpaceWrapper(env)
venv = ManiSkillVectorEnv(env, N, ignore_terminations=True, record_metrics=True)
base = venv.base_env
S, A = venv.single_observation_space.shape[0], venv.single_action_space.shape[0]
low, high = (torch.from_numpy(b).to(dev) for b in (venv.single_action_space.low, venv.single_action_space.high))


def run(pol, seed=EVAL_SEED):
    obs, _ = venv.reset(seed=seed)
    z0 = base.cube.pose.p[:, 2].clone()
    zmax, goal_ok = z0.clone(), torch.zeros(N, dtype=torch.bool, device=dev)
    for _ in range(T):
        with torch.no_grad():
            a = torch.max(torch.min(pol(obs), high), low)
        obs, _, _, _, info = venv.step(a)
        zmax = torch.max(zmax, base.cube.pose.p[:, 2])
        goal_ok |= (info["final_info"] if "final_info" in info else info)["success"].bool()
    return z0.cpu().numpy(), zmax.cpu().numpy(), goal_ok.cpu().numpy()


class _Space:
    single_observation_space, single_action_space = venv.single_observation_space, venv.single_action_space


expert = Agent(_Space).to(dev)
expert.load_state_dict(torch.load(LOG_ROOT / "runs/pick-expert/final_ckpt.pt", map_location=dev)); expert.eval()
z0, zm, g = run(lambda o: expert.get_action(o, deterministic=True))
res = {"expert": {"goal": float(g.mean()), **{f"lift{t}": float((zm >= t).mean()) for t in THRESH}}, "bc": {}}
print("cube rest z", z0.mean().round(3), "expert", res["expert"], flush=True)
sizes = [10, 50, 100, 200]
tab = {t: np.zeros((len(sizes), 3)) for t in THRESH}; gt = np.zeros((len(sizes), 3)); zmed = np.zeros((len(sizes), 3))
for i, n in enumerate(sizes):
    for j, s in enumerate((1, 2, 3)):
        ck = torch.load(LOG_ROOT / f"runs/pick-bc/checkpoints/bc-{n}-s{s}.pt", map_location=dev)
        net = nn.Sequential(nn.Linear(S, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, A)).to(dev)
        net.load_state_dict(ck["net"])
        _, zm, g = run(lambda o: net((o - ck["mu"]) / ck["sd"]))
        for t in THRESH:
            tab[t][i, j] = (zm >= t).mean()
        gt[i, j] = g.mean(); zmed[i, j] = np.median(zm)
        res["bc"][f"N{n}-s{s}"] = {"goal": float(g.mean()), "median_max_z": float(np.median(zm)), **{f"lift{t}": float((zm >= t).mean()) for t in THRESH}}
    print(f"N={n:3d}: " + "  ".join(f"lift>={t:.2f}m {tab[t][i].mean():.3f}" for t in THRESH) + f" | goal {gt[i].mean():.3f} | median max z {zmed[i].mean():.3f}", flush=True)
json.dump(res, open(LOG_ROOT / "runs/pick-bc/lift_results.json", "w"), indent=1)
fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
for t in THRESH:
    ax.errorbar(sizes, tab[t].mean(1), tab[t].std(1), marker="o", label=f"lift: cube z >= {t:.2f} m", lw=2 if t == 0.10 else 1)
ax.errorbar(sizes, gt.mean(1), gt.std(1), marker="s", c="k", ls="--", label="original: reach goal (+ static)")
ax.axhline(res["expert"]["lift0.1"], c="gray", ls=":", label="expert, lift >= 0.10")
ax.set_xscale("log"); ax.set_xticks(sizes); ax.set_xticklabels(sizes); ax.set_ylim(0, 1.02); ax.set_xlabel("number of demos"); ax.set_ylabel("success rate (512 resets, 3 seeds)"); ax.legend(fontsize=8); ax.set_title("PickCube BC: lift success vs demos")
fig.savefig(LOG_ROOT / "renders/pick-bc/lift_vs_demos.png", dpi=120)
