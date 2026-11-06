"""Plain BC on PickCube-v1 from expert (PPO, logs/runs/pick-expert/final_ckpt.pt) demonstrations of varying size.

Data (per seed): successful expert episodes on random resets (full state obs, 42-d), the episode ends at the first success (steps after are masked).
Nested prefixes of size N use the first N demos. BC: MLP(obs -> action), MSE to the expert's deterministic action, no context, no online data.
Eval: 512 fixed resets, deterministic policy, success_once within 50 steps. One checkpoint per (N, seed).
"""
import time
from dataclasses import dataclass

import gymnasium as gym
import torch
import torch.nn as nn
import tyro

import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from ppo import Agent
from run_logging import LOG_ROOT, RunDir


@dataclass
class Args:
    expert_checkpoint: str = str(LOG_ROOT / "runs/pick-expert/final_ckpt.pt")
    out_dir: str = str(LOG_ROOT / "runs/pick-bc")
    sizes: str = "10,50,100,200"
    seeds: str = "1,2,3"
    num_envs: int = 512
    num_steps: int = 50
    steps: int = 5000
    batch_size: int = 256
    lr: float = 3e-4
    width: int = 256
    eval_seed: int = 12345


def rollout(venv, act_fn, num_steps, seed, expert=None):
    """One episode on every env, terminating at the first success. Returns obs (N,T,S), expert labels (N,T,A), alive (N,T), success_once (N,)."""
    obs, _ = venv.reset(seed=seed)
    N = obs.shape[0]
    done = torch.zeros(N, dtype=torch.bool, device=obs.device)
    xs, labs, alive = [], [], []
    low, high = (torch.from_numpy(b).to(obs.device) for b in (venv.single_action_space.low, venv.single_action_space.high))
    for _ in range(num_steps):
        with torch.no_grad():
            a = act_fn(obs)
            lab = expert.get_action(obs, deterministic=True) if expert is not None else a
        xs.append(obs), labs.append(torch.max(torch.min(lab, high), low)), alive.append(~done)
        obs, _, _, _, info = venv.step(torch.max(torch.min(a, high), low))
        done |= (info["final_info"] if "final_info" in info else info)["success"].bool()
    return dict(obs=torch.stack(xs, 1), lab=torch.stack(labs, 1), alive=torch.stack(alive, 1), succ=done)


def main():
    args = tyro.cli(Args)
    dev = torch.device("cuda")
    sizes, seeds = [int(x) for x in args.sizes.split(",")], [int(x) for x in args.seeds.split(",")]
    K = max(sizes)
    run = RunDir(args.out_dir, args)
    N = args.num_envs
    env = gym.make("PickCube-v1", num_envs=N, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cuda")
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    venv = ManiSkillVectorEnv(env, N, ignore_terminations=True, record_metrics=True)
    S, A = venv.single_observation_space.shape[0], venv.single_action_space.shape[0]

    class _Space:
        single_observation_space, single_action_space = venv.single_observation_space, venv.single_action_space

    expert = Agent(_Space).to(dev)
    expert.load_state_dict(torch.load(args.expert_checkpoint, map_location=dev))
    expert.eval()
    ref = rollout(venv, lambda o: expert.get_action(o, deterministic=True), args.num_steps, args.eval_seed)["succ"].float().mean().item()
    print(f"expert on eval resets: {ref:.3f}", flush=True)
    results = dict(expert_ref=ref, runs={})
    for seed in seeds:
        torch.manual_seed(seed)
        O, L, M, have, b = [], [], [], 0, 0
        while have < K:  # collect successful expert demos on fresh resets
            r = rollout(venv, lambda o: expert.get_action(o, deterministic=True), args.num_steps, seed * 100000 + b)
            ok = r["succ"].nonzero()[:, 0]
            O.append(r["obs"][ok]), L.append(r["lab"][ok]), M.append(r["alive"][ok])
            have += len(ok)
            b += 1
        obs, lab, alive = (torch.cat(x)[:K] for x in (O, L, M))
        torch.save(dict(obs=obs.cpu(), lab=lab.cpu(), alive=alive.cpu()), f"{run.path}/data-s{seed}.pt")
        print(f"seed {seed}: collected {K} successful demos in {b} batches; mean length {alive.sum(1).float().mean():.1f}", flush=True)
        for n in sizes:
            q, y = obs[:n][alive[:n]], lab[:n][alive[:n]]  # flatten the alive steps
            mu, sd = q.mean(0), q.std(0).clamp_min(1e-3)
            net = nn.Sequential(nn.Linear(S, args.width), nn.ReLU(), nn.Linear(args.width, args.width), nn.ReLU(), nn.Linear(args.width, args.width), nn.ReLU(), nn.Linear(args.width, A)).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=args.lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.02)
            t0 = time.time()
            for _ in range(args.steps):
                idx = torch.randint(0, len(q), (args.batch_size,), device=dev)
                loss = ((net((q[idx] - mu) / sd) - y[idx]) ** 2).mean()
                opt.zero_grad(), loss.backward(), opt.step(), sched.step()
            pol = lambda o: net((o - mu) / sd)
            succ = rollout(venv, pol, args.num_steps, args.eval_seed)["succ"].float().mean().item()
            torch.save(dict(net=net.state_dict(), mu=mu, sd=sd), run.ckpt(f"bc-{n}-s{seed}.pt"))
            results["runs"][f"bc-{n}-s{seed}"] = dict(loss=loss.item(), success=succ, frames=len(q))
            print(f"seed {seed} N={n:4d} frames {len(q):5d} loss {loss.item():.5f} success {succ:.3f} ({time.time()-t0:.0f}s)", flush=True)
            run.save_json("results.json", results)


if __name__ == "__main__":
    main()
