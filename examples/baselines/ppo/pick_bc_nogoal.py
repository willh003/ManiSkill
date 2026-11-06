"""Goal-blind BC on PickCube-v1: same expert demos as pick_bc.py (logs/runs/pick-bc/data-s{seed}.pt), but goal_pos (obs 26:29) and obj_to_goal_pos (39:42)
are removed from the student's observation (36-d). The demonstrations are still goal-directed (expert saw the goal).
Eval: 512 fixed resets; success = cube max z >= threshold (lift), plus goal-reaching success for reference."""
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from run_logging import LOG_ROOT, RunDir

N, T, EVAL_SEED, STEPS, BS, LR = 512, 50, 12345, 5000, 256, 3e-4
KEEP = [i for i in range(42) if not (26 <= i < 29 or 39 <= i < 42)]
THRESH = [0.05, 0.10, 0.15, 0.20]
dev = torch.device("cuda")


run = RunDir(str(LOG_ROOT / "runs/pick-bc-nogoal"), argparse.Namespace(steps=STEPS, bs=BS, lr=LR, keep=KEEP))
env = gym.make("PickCube-v1", num_envs=N, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cuda")
if isinstance(env.action_space, gym.spaces.Dict):
    env = FlattenActionSpaceWrapper(env)
venv = ManiSkillVectorEnv(env, N, ignore_terminations=True, record_metrics=True)
base = venv.base_env
A = venv.single_action_space.shape[0]
low, high = (torch.from_numpy(b).to(dev) for b in (venv.single_action_space.low, venv.single_action_space.high))


def evaluate(pol):
    obs, _ = venv.reset(seed=EVAL_SEED)
    zmax, goal_ok = base.cube.pose.p[:, 2].clone(), torch.zeros(N, dtype=torch.bool, device=dev)
    for _ in range(T):
        with torch.no_grad():
            a = torch.max(torch.min(pol(obs[:, KEEP]), high), low)
        obs, _, _, _, info = venv.step(a)
        zmax = torch.max(zmax, base.cube.pose.p[:, 2])
        goal_ok |= (info["final_info"] if "final_info" in info else info)["success"].bool()
    return {**{f"lift{t}": (zmax >= t).float().mean().item() for t in THRESH}, "goal": goal_ok.float().mean().item(), "median_max_z": zmax.median().item()}


res = {}
for seed in (1, 2, 3):
    torch.manual_seed(seed)
    d = torch.load(LOG_ROOT / f"runs/pick-bc/data-s{seed}.pt")
    obs, lab, alive = (d[k].to(dev) for k in ("obs", "lab", "alive"))
    for n in (10, 50, 100, 200):
        q, y = obs[:n][alive[:n]][:, KEEP], lab[:n][alive[:n]]
        mu, sd = q.mean(0), q.std(0).clamp_min(1e-3)
        net = nn.Sequential(nn.Linear(len(KEEP), 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, A)).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=LR * 0.02)
        for _ in range(STEPS):
            idx = torch.randint(0, len(q), (BS,), device=dev)
            loss = ((net((q[idx] - mu) / sd) - y[idx]) ** 2).mean()
            opt.zero_grad(), loss.backward(), opt.step(), sched.step()
        r = evaluate(lambda o: net((o - mu) / sd)); r["loss"] = loss.item()
        res[f"N{n}-s{seed}"] = r
        torch.save(dict(net=net.state_dict(), mu=mu, sd=sd, keep=KEEP), run.ckpt(f"bc-nogoal-{n}-s{seed}.pt"))
        print(f"seed {seed} N={n:3d} loss {loss.item():.5f} " + " ".join(f"lift{t} {r[f'lift{t}']:.3f}" for t in THRESH) + f" goal {r['goal']:.3f} medz {r['median_max_z']:.3f}", flush=True)
run.save_json("results.json", res)
old = json.load(open(LOG_ROOT / "runs/pick-bc/lift_results.json"))["bc"]
print("\nmean over 3 seeds: with goal -> without goal")
for n in (10, 50, 100, 200):
    f = lambda dct, k: np.mean([dct[f"N{n}-s{s}"][k] for s in (1, 2, 3)])
    print(f"N={n:3d}: " + " | ".join(f"{k}: {f(old, k):.2f} -> {f(res, k):.2f}" for k in ("lift0.05", "lift0.1", "lift0.15", "lift0.2", "goal")))
