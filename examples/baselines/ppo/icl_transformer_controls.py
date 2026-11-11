"""Per-k controls for the noisy-correction transformer (kind hist_c_noisy) on its saved checkpoints.
 oracle  : exact correction (s=1) at every attempt (noise-free upper reference)
 noisy   : s~U(.9,1.1) fresh per attempt (as in the main eval; reproduced here for same-run comparison)
 shuf    : context (hist, f, c) of ANOTHER env at every attempt (should not help)
Reports pass@k, per-attempt and best-so-far peg-goal distance, k=1..8. Usage: python icl_transformer_controls.py [seeds=1,2,3] -> logs/runs/icl-tf-controls/controls.json"""
import json, sys
import gymnasium as gym
import torch
import tyro

import icl_transformer as T
from icl_checker import rollout
from icl_tasks import checker_sample
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from ppo import Agent
from run_logging import LOG_ROOT

seeds = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1,2,3").split(",")]
args = T.Args()
dev = torch.device("cuda")
N, K = args.num_envs, 8
env = gym.make("PegInsertionSideBias-v1", num_envs=N, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cuda")
env = FlattenActionSpaceWrapper(env) if isinstance(env.action_space, gym.spaces.Dict) else env
venv = ManiSkillVectorEnv(env, N, ignore_terminations=True, record_metrics=True)
A = venv.single_action_space.shape[0]


class _S:
    single_observation_space = type("S", (), {"shape": (43,)})
    single_action_space = venv.single_action_space


expert = Agent(_S).to(dev)
expert.load_state_dict(torch.load(args.expert_checkpoint, map_location=dev))
expert.eval()
g = torch.Generator(device=dev).manual_seed(args.eval_seed)
goals = checker_sample(N, "all", dev, g, args.split)
ex = rollout(venv, expert, None, args, goals, args.eval_seed, "expert")
ex_peg = ex["obs"][torch.arange(N), T.final_idx(ex["alive"])][:, T.PEG]
kind, L = "hist_c_noisy", T.LAYOUT["hist_c_noisy"]
dummy = dict(s=torch.zeros(N, 1, 43, device=dev), a=torch.zeros(N, 1, A, device=dev), y=torch.zeros(N, 1, L * 7, device=dev), valid=torch.zeros(N, 1, dtype=torch.bool, device=dev))


def run(pol, mode):
    out, r = [], None
    for k in range(1, K + 1):
        if k == 1:
            r = rollout(venv, expert, pol, args, goals, args.eval_seed, "student", dummy, "normal", mask_pose=True)
        else:
            fi = T.final_idx(r["alive"])
            fp = r["obs"][torch.arange(N), fi][:, T.PEG]
            s = torch.ones(N, device=dev) if mode == "oracle" else args.noise_lo + (args.noise_hi - args.noise_lo) * torch.rand(N, device=dev)
            c = T.mk(kind, T.hist_of(r["obs"], fi), fp, ex_peg, s)
            if mode == "shuf":
                c = torch.roll(c, 1, 0)
            ctx = dict(s=torch.zeros(N, 1, 43, device=dev), a=torch.zeros(N, 1, A, device=dev), y=c.reshape(N, 1, -1), valid=torch.ones(N, 1, dtype=torch.bool, device=dev))
            r = rollout(venv, expert, pol, args, goals, args.eval_seed, "student", ctx, "normal", mask_pose=True)
        fi = T.final_idx(r["alive"])
        fpk = r["obs"][torch.arange(N), fi][:, T.PEG]
        out.append(dict(succ=r["succ"], dist=(fpk[:, :3] - ex_peg[:, :3]).norm(dim=-1)))
    succ = torch.stack([o["succ"] for o in out], 1)
    dist = torch.stack([o["dist"] for o in out], 1)
    return dict(pass_k=succ.cummax(1).values.float().mean(0).tolist(), succ=succ.float().mean(0).tolist(), dist=dist.mean(0).tolist(), dist_best=dist.cummin(1).values.mean(0).tolist())


res = {}
for s in seeds:
    for n in (100, 1000, 10000):
        pol = T.TF(A, L, args.d, args.layers, args.heads).to(dev)
        pol.load_state_dict(torch.load(f"{LOG_ROOT}/runs/icl-tf-{kind}-s{s}/checkpoints/{kind}-N{n}.pt", map_location=dev))
        pol.eval()
        torch.manual_seed(s * 1000 + n)
        for mode in ("oracle", "noisy", "shuf"):
            res[f"s{s}-N{n}-{mode}"] = run(pol, mode)
            m = res[f"s{s}-N{n}-{mode}"]
            print(f"s{s} N={n} {mode}: pass@k {[round(x, 3) for x in m['pass_k']]} dist_best {[round(x, 3) for x in m['dist_best']]}", flush=True)
        json.dump(res, open(f"{LOG_ROOT}/runs/icl-tf-controls.json", "w"))
