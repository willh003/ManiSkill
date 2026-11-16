"""Fair BC baseline for pass@k. Usage: python icl_fair_bc.py PREFIX OUT.json [seeds]   (PREFIX-s{seed}/checkpoints/{tf-bc,fcfc-N1000}.pt, concat T1)
Conditions (k=1..8, hole visible for BC):
  bc_det        deterministic BC, same seed each retry (the earlier flat baseline)
  bc_noise{σ}   BC + Gaussian action noise σ, independent per retry
  t1_real       T1 recursive (context = previous attempt)
  t1_fixed      T1, context from attempt 1 reused for every retry (policy deterministic -> should be flat)
  t1_randf      T1, context [f',g-f',g] with f' = a random other task's failed final pose, fresh each retry (goal exact, f irrelevant)"""
import json, os, sys
import gymnasium as gym
import torch
import icl_transformer as T
from icl_checker import rollout
from icl_tasks import checker_sample
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from ppo import Agent
from run_logging import LOG_ROOT

PFX, OUTF = sys.argv[1], sys.argv[2]
seeds = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,2,3").split(",")]
args = T.Args(); dev = torch.device("cuda"); N, K = args.num_envs, 8
env = gym.make("PegInsertionSideBias-v1", num_envs=N, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cuda")
env = FlattenActionSpaceWrapper(env) if isinstance(env.action_space, gym.spaces.Dict) else env
venv = ManiSkillVectorEnv(env, N, ignore_terminations=True, record_metrics=True)
A = venv.single_action_space.shape[0]


class _S:
    single_observation_space = type("S", (), {"shape": (43,)}); single_action_space = venv.single_action_space


expert = Agent(_S).to(dev); expert.load_state_dict(torch.load(args.expert_checkpoint, map_location=dev)); expert.eval()
goals = checker_sample(N, "all", dev, torch.Generator(device=dev).manual_seed(args.eval_seed), args.split)
ex = rollout(venv, expert, None, args, goals, args.eval_seed, "expert")
ar = torch.arange(N, device=dev)
ex_peg = ex["obs"][ar, T.final_idx(ex["alive"])][:, T.PEG]
KIND = os.environ.get("KIND", "fcfc"); KPFX = os.environ.get("ICL_PFX", PFX)  # ICL variant (fcfc=T1, hist_c=B) and its run-dir prefix
L = T.LAYOUT[KIND]


def ctx_of(y, valid):
    return dict(s=torch.zeros(N, 1, 43, device=dev), a=torch.zeros(N, 1, A, device=dev), y=y.reshape(N, 1, -1), valid=torch.full((N, 1), valid, dtype=torch.bool, device=dev))


def fin(r):
    return r["obs"][ar, T.final_idx(r["alive"])][:, T.PEG]


def summ(succ):
    s = torch.stack(succ, 1)
    return dict(succ=s.float().mean(0).tolist(), pass_k=s.cummax(1).values.float().mean(0).tolist())


def bc_run(pol, sigma):
    d0 = dict(s=torch.zeros(N, 1, 43, device=dev), a=torch.zeros(N, 1, A, device=dev), y=torch.zeros(N, 0, 0, device=dev), valid=torch.zeros(N, 1, dtype=torch.bool, device=dev))
    return summ([rollout(venv, expert, pol, args, goals, args.eval_seed, "student", d0, "normal", mask_pose=False, act_noise=sigma)["succ"] for _ in range(K)])


def t1_run(pol, mode, bcpol=None):
    if bcpol is not None:  # attempt 1 = BC policy (hole visible); its rollout is the first context
        d0 = dict(s=torch.zeros(N, 1, 43, device=dev), a=torch.zeros(N, 1, A, device=dev), y=torch.zeros(N, 0, 0, device=dev), valid=torch.zeros(N, 1, dtype=torch.bool, device=dev))
        r = rollout(venv, expert, bcpol, args, goals, args.eval_seed, "student", d0, "normal", mask_pose=False)
    else:
        r = rollout(venv, expert, pol, args, goals, args.eval_seed, "student", ctx_of(torch.zeros(N, L * 7, device=dev), False), "normal", mask_pose=True)
    succ, f1, h1 = [r["succ"]], fin(r), T.hist_of(r["obs"], T.final_idx(r["alive"]))
    for k in range(1, K):
        if mode == "real":
            f, h = fin(r), T.hist_of(r["obs"], T.final_idx(r["alive"]))
        elif mode == "fixed":
            f, h = f1, h1
        else:
            pm = torch.randperm(N, device=dev)
            f, h = f1[pm], h1[pm]
        c = T.mk(KIND, h, f, ex_peg, torch.ones(N, device=dev))
        r = rollout(venv, expert, pol, args, goals, args.eval_seed, "student", ctx_of(c, True), "normal", mask_pose=True)
        succ.append(r["succ"])
    return summ(succ)


res = {}
for s in seeds:
    d = f"{LOG_ROOT}/runs/{PFX}-s{s}/checkpoints"
    bc = T.TF(A, 0, args.d, args.layers, args.heads, args.dropout).to(dev); bc.load_state_dict(torch.load(f"{LOG_ROOT}/runs/{os.environ.get('BC_PFX', PFX)}-s{s}/checkpoints/{os.environ.get('BC_FILE', 'tf-bc.pt')}", map_location=dev)); bc.eval()
    t1 = T.TF(A, L, args.d, args.layers, args.heads, args.dropout, True).to(dev); t1.load_state_dict(torch.load(f"{LOG_ROOT}/runs/{KPFX}-s{s}/checkpoints/{KIND}-N1000.pt", map_location=dev)); t1.eval()
    if os.environ.get("BCFIRST"):
        for m in ("real", "fixed", "randf"):
            res[f"s{s}-bf_{m}"] = t1_run(t1, m, bc)
    else:
        for sg in (0.0, 0.02, 0.05, 0.1):
            res[f"s{s}-bc_{'det' if sg == 0 else f'noise{sg}'}"] = bc_run(bc, sg)
        for m in ("real", "fixed", "randf"):
            res[f"s{s}-t1_{m}"] = t1_run(t1, m)
    for k, v in res.items():
        if k.startswith(f"s{s}-"): print(k, "succ", [round(x, 3) for x in v["succ"]][:4], "pass@k", [round(x, 3) for x in v["pass_k"]], flush=True)
    json.dump(res, open(f"{LOG_ROOT}/runs/{OUTF}", "w"))
