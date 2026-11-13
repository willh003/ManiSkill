"""Measured BC baselines at k=1..8 (retries: same seed + deterministic policy). Usage: python icl_bc_passk.py 1,2,3"""
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




def run_k(pol, ctx, mask, kind_L):
    succ = []
    for k in range(K):
        r = rollout(venv, expert, pol, args, goals, args.eval_seed, "student", ctx, "normal", mask_pose=mask)
        succ.append(r["succ"])
    s = torch.stack(succ, 1)
    return dict(succ=s.float().mean(0).tolist(), pass_k=s.cummax(1).values.float().mean(0).tolist())


PFX = sys.argv[2] if len(sys.argv) > 2 else "icl-tf-fcfc"  # run-dir prefix holding checkpoints/tf-bc.pt; if given, BC only
OUTF = sys.argv[3] if len(sys.argv) > 3 else "icl-bc-passk.json"
res = {}
for s in seeds:
    pol = T.TF(A, 0, args.d, args.layers, args.heads, args.dropout).to(dev)
    pol.load_state_dict(torch.load(f"{LOG_ROOT}/runs/{PFX}-s{s}/checkpoints/tf-bc.pt", map_location=dev)); pol.eval()
    d0 = dict(s=torch.zeros(N, 1, 43, device=dev), a=torch.zeros(N, 1, A, device=dev), y=torch.zeros(N, 0, 0, device=dev), valid=torch.zeros(N, 1, dtype=torch.bool, device=dev))
    res[f"s{s}-tfbc_visible"] = run_k(pol, d0, False, 0)
    res[f"s{s}-tfbc_hidden"] = run_k(pol, d0, True, 0)
    for n in ((100, 1000, 10000) if len(sys.argv) <= 2 else ()):
        L = T.LAYOUT["fcfc"]
        p2 = T.TF(A, L, args.d, args.layers, args.heads, args.dropout).to(dev)
        p2.load_state_dict(torch.load(f"{LOG_ROOT}/runs/icl-tf-fcfc-s{s}/checkpoints/fcfc-N{n}.pt", map_location=dev)); p2.eval()
        dm = dict(s=torch.zeros(N, 1, 43, device=dev), a=torch.zeros(N, 1, A, device=dev), y=torch.zeros(N, 1, L * 7, device=dev), valid=torch.zeros(N, 1, dtype=torch.bool, device=dev))
        res[f"s{s}-T1_noctx-N{n}"] = run_k(p2, dm, True, L)
    for k, v in res.items():
        if k.startswith(f"s{s}-"): print(k, [round(x, 3) for x in v["pass_k"]], flush=True)
    json.dump(res, open(f"{LOG_ROOT}/runs/{OUTF}", "w"))
